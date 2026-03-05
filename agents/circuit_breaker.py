"""Circuit-breaker decorator for LangGraph agent nodes.

Constraint R01 from claude.md: max_iterations=3 per node.
When a node exceeds MAX_ITERATIONS the circuit opens and routes to
Error_Handler by setting state["status"] = "circuit_open".
"""
from __future__ import annotations

import logging
from functools import wraps
from typing import Callable

from agents.state import AgentState, MAX_ITERATIONS

logger = logging.getLogger(__name__)


def circuit_breaker(counter_key: str) -> Callable:
    """Decorator factory that adds a per-node iteration guard.

    Args:
        counter_key: The AgentState key used to track this node's iterations
                     (e.g. "pm_iterations", "dev_iterations").

    Usage::

        @circuit_breaker("pm_iterations")
        def pm_agent_node(state: AgentState) -> AgentState:
            ...
    """

    def decorator(fn: Callable[[AgentState], AgentState]) -> Callable:
        @wraps(fn)
        def wrapper(state: AgentState) -> AgentState:
            current = state.get(counter_key, 0)

            if current >= MAX_ITERATIONS:
                logger.warning(
                    "Circuit breaker OPEN for %s: %d/%d iterations reached",
                    fn.__name__,
                    current,
                    MAX_ITERATIONS,
                )
                return {
                    **state,
                    "status": "circuit_open",
                    "error": (
                        f"Circuit breaker tripped for {fn.__name__}: "
                        f"{current}/{MAX_ITERATIONS} max iterations exceeded."
                    ),
                    "current_agent": fn.__name__,
                }

            logger.info(
                "%s iteration %d/%d", fn.__name__, current + 1, MAX_ITERATIONS
            )
            updated_state = {**state, counter_key: current + 1}
            return fn(updated_state)

        return wrapper

    return decorator
