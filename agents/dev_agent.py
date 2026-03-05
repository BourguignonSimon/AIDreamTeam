"""Dev Agent node — generates code from the PM specification.

Responsibilities:
- Read state["spec"] produced by PM_Agent.
- Generate well-structured Python code satisfying the acceptance criteria.
- Store output in state["code"].
- Route to QA_Agent on success, Error_Handler on circuit-open.
"""
from __future__ import annotations

import logging
import os

import litellm

from agents.circuit_breaker import circuit_breaker
from agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior Python engineer. Given a product specification, write clean,
well-documented Python code that satisfies the acceptance criteria.

Rules:
- Use type annotations everywhere.
- Add docstrings to every public function and class.
- Do NOT include test code — QA handles that separately.
- Return ONLY the Python source code, no markdown fences.
"""


@circuit_breaker("dev_iterations")
def dev_agent_node(state: AgentState) -> AgentState:
    """LangGraph node — Dev Agent."""
    spec = state.get("spec") or ""
    if not spec:
        return {
            **state,
            "status": "error",
            "error": "Dev_Agent received empty spec from PM_Agent.",
            "current_agent": "Dev_Agent",
        }

    logger.info("Dev_Agent generating code for spec (len=%d)", len(spec))

    try:
        response = litellm.completion(
            model=os.getenv("DEV_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task description: {state['task']}\n\n"
                        f"Specification:\n{spec}"
                    ),
                },
            ],
            temperature=0.1,
        )
        code = response.choices[0].message.content.strip()

        return {
            **state,
            "code": code,
            "status": "running",
            "current_agent": "Dev_Agent",
            "messages": [
                {
                    "role": "assistant",
                    "agent": "Dev_Agent",
                    "content": f"Code generated ({len(code)} chars).",
                }
            ],
        }

    except Exception as exc:
        logger.error("Dev_Agent error: %s", exc, exc_info=True)
        return {
            **state,
            "status": "error",
            "error": f"Dev_Agent failed: {exc}",
            "current_agent": "Dev_Agent",
        }
