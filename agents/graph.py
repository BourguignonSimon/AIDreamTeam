"""LangGraph multi-agent graph assembly.

Topology
--------

    [START]
       |
       v
   PM_Agent  ──(error/circuit_open)──────────────────┐
       |                                              |
       v (spec ready)                                 |
   Dev_Agent ──(error/circuit_open)──────────────────┤
       |                                              |
       v (code ready)                                 |
   QA_Agent  ──(error/circuit_open)──────────────────┤
       |              |                               |
  (qa_passed)   (qa_failed, retry)                    |
       |              └──> Dev_Agent (with context)   |
       v                                              v
     [END]                                    Error_Handler
                                                      |
                                                      v
                                                    [END]

Circuit breaker: each node is wrapped with @circuit_breaker(max_iterations=3).
When a node's counter reaches MAX_ITERATIONS the decorator short-circuits to
status="circuit_open" before the LLM call, and the conditional router sends
the state directly to Error_Handler.
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.dev_agent import dev_agent_node
from agents.error_handler import error_handler_node
from agents.pm_agent import pm_agent_node
from agents.qa_agent import qa_agent_node
from agents.state import AgentState, MAX_ITERATIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

_FAILURE_STATUSES = {"error", "circuit_open"}


def route_after_pm(
    state: AgentState,
) -> Literal["Dev_Agent", "Error_Handler"]:
    """Route PM_Agent output."""
    if state.get("status") in _FAILURE_STATUSES:
        return "Error_Handler"
    return "Dev_Agent"


def route_after_dev(
    state: AgentState,
) -> Literal["QA_Agent", "Error_Handler"]:
    """Route Dev_Agent output."""
    if state.get("status") in _FAILURE_STATUSES:
        return "Error_Handler"
    return "QA_Agent"


def route_after_qa(
    state: AgentState,
) -> Literal["Dev_Agent", "Error_Handler", "__end__"]:
    """Route QA_Agent output.

    - qa_passed=True  → END  (pipeline complete)
    - qa_passed=False AND dev_iterations < MAX_ITERATIONS → Dev_Agent retry
    - qa_passed=False AND dev_iterations >= MAX_ITERATIONS → Error_Handler
    - status in failure statuses → Error_Handler
    """
    if state.get("status") in _FAILURE_STATUSES:
        return "Error_Handler"

    if state.get("qa_passed"):
        return END

    # QA failed — check if Dev_Agent has retries left
    if state.get("dev_iterations", 0) < MAX_ITERATIONS:
        logger.info(
            "QA failed — routing back to Dev_Agent (dev_iterations=%d)",
            state.get("dev_iterations", 0),
        )
        return "Dev_Agent"

    logger.warning("QA failed and Dev_Agent circuit open — routing to Error_Handler")
    return "Error_Handler"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph.

    Returns:
        A compiled LangGraph application ready for ``.invoke()`` or
        ``.stream()`` calls.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("PM_Agent", pm_agent_node)
    builder.add_node("Dev_Agent", dev_agent_node)
    builder.add_node("QA_Agent", qa_agent_node)
    builder.add_node("Error_Handler", error_handler_node)

    # Entry point
    builder.set_entry_point("PM_Agent")

    # Conditional edges
    builder.add_conditional_edges(
        "PM_Agent",
        route_after_pm,
        {"Dev_Agent": "Dev_Agent", "Error_Handler": "Error_Handler"},
    )
    builder.add_conditional_edges(
        "Dev_Agent",
        route_after_dev,
        {"QA_Agent": "QA_Agent", "Error_Handler": "Error_Handler"},
    )
    builder.add_conditional_edges(
        "QA_Agent",
        route_after_qa,
        {
            "Dev_Agent": "Dev_Agent",
            "Error_Handler": "Error_Handler",
            END: END,
        },
    )

    # Error_Handler always terminates
    builder.add_edge("Error_Handler", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def _initial_state(task: str) -> AgentState:
    return AgentState(
        task=task,
        messages=[],
        status="running",
        current_agent="",
        pm_iterations=0,
        dev_iterations=0,
        qa_iterations=0,
        error_iterations=0,
        spec=None,
        code=None,
        qa_report=None,
        qa_passed=False,
        error=None,
        result=None,
    )


def run_graph(task: str) -> AgentState:
    """Execute the full multi-agent pipeline for a given task.

    Args:
        task: Plain-text description of the feature or problem to solve.

    Returns:
        The final AgentState after the graph terminates.
    """
    graph = build_graph()
    final_state: AgentState = graph.invoke(_initial_state(task))
    return final_state
