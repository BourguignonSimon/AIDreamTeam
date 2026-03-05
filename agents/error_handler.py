"""Error Handler node — terminal sink for failures and open circuits.

Responsibilities:
- Receive any state where status == "error" or "circuit_open".
- Log structured error information.
- Produce a human-readable error summary in state["result"].
- Always terminates the graph (routes to END).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import litellm

from agents.circuit_breaker import circuit_breaker
from agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an incident-response engineer. Given error details from an AI pipeline,
produce a short (3-5 sentences) plain-English incident summary that includes:
1. Which agent failed and why.
2. What the user should do next.
3. Any actionable remediation steps.

Respond with plain text only — no JSON, no markdown.
"""


@circuit_breaker("error_iterations")
def error_handler_node(state: AgentState) -> AgentState:
    """LangGraph node — Error Handler (always terminates)."""
    error_msg = state.get("error") or "Unknown error"
    failed_agent = state.get("current_agent") or "unknown"
    status = state.get("status") or "error"

    logger.error(
        json.dumps(
            {
                "event": "pipeline_failure",
                "failed_agent": failed_agent,
                "status": status,
                "error": error_msg,
                "pm_iterations": state.get("pm_iterations", 0),
                "dev_iterations": state.get("dev_iterations", 0),
                "qa_iterations": state.get("qa_iterations", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    )

    # Attempt an LLM-assisted incident summary; fall back to plain text.
    try:
        response = litellm.completion(
            model=os.getenv("ERROR_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Failed agent: {failed_agent}\n"
                        f"Status: {status}\n"
                        f"Error: {error_msg}\n"
                        f"Task: {state.get('task', 'N/A')}"
                    ),
                },
            ],
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Error_Handler LLM call failed: %s", exc)
        summary = (
            f"Pipeline failure in {failed_agent} ({status}): {error_msg}. "
            "Please review the logs and retry the task."
        )

    return {
        **state,
        "status": "error",
        "result": summary,
        "current_agent": "Error_Handler",
        "messages": [
            {
                "role": "assistant",
                "agent": "Error_Handler",
                "content": summary,
            }
        ],
    }
