"""Shared state definition for the multi-agent LangGraph workflow."""
from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator


def _add_messages(left: list[dict], right: list[dict]) -> list[dict]:
    """Reducer that appends new messages to the existing list."""
    return left + right


class AgentState(TypedDict):
    """Shared state flowing between all agent nodes.

    Circuit-breaker counters (pm_iterations, dev_iterations, qa_iterations,
    error_iterations) are each capped at MAX_ITERATIONS=3 per constraint R01.
    """

    # Core task payload
    task: str
    messages: Annotated[list[dict[str, Any]], _add_messages]

    # Current pipeline status
    status: str          # "running" | "done" | "error"
    current_agent: str   # name of the last agent that ran

    # Circuit-breaker counters — one per node
    pm_iterations: int
    dev_iterations: int
    qa_iterations: int
    error_iterations: int

    # Artefacts produced by nodes
    spec: Optional[str]        # PM output: user story / acceptance criteria
    code: Optional[str]        # Dev output: generated code
    qa_report: Optional[str]   # QA output: test results
    qa_passed: bool            # QA verdict

    # Error information
    error: Optional[str]
    result: Optional[str]      # Final answer returned to the caller


# Constraint from claude.md — R01
MAX_ITERATIONS: int = 3
