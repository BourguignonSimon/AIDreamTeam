"""PM Agent node — transforms a raw task into a structured specification.

Responsibilities:
- Analyse the incoming task description.
- Produce acceptance criteria and a user story (stored in state["spec"]).
- Route to Dev_Agent on success, Error_Handler on circuit-open.
"""
from __future__ import annotations

import json
import logging
import os

import litellm

from agents.circuit_breaker import circuit_breaker
from agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior Product Manager. Given a task description, produce a concise
specification in JSON with exactly these keys:
  - "user_story": one sentence in "As a … I want … So that …" format
  - "acceptance_criteria": list of 3-5 testable criteria (strings)
  - "out_of_scope": list of 1-3 items explicitly excluded

Respond with valid JSON only — no markdown fences, no extra text.
"""


@circuit_breaker("pm_iterations")
def pm_agent_node(state: AgentState) -> AgentState:
    """LangGraph node — PM Agent."""
    task = state["task"]
    logger.info("PM_Agent processing task: %s", task[:80])

    try:
        response = litellm.completion(
            model=os.getenv("PM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {task}"},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        spec_data = json.loads(raw)
        spec_text = json.dumps(spec_data, indent=2, ensure_ascii=False)

        return {
            **state,
            "spec": spec_text,
            "status": "running",
            "current_agent": "PM_Agent",
            "messages": [
                {
                    "role": "assistant",
                    "agent": "PM_Agent",
                    "content": f"Spec produced:\n{spec_text}",
                }
            ],
        }

    except Exception as exc:
        logger.error("PM_Agent error: %s", exc, exc_info=True)
        return {
            **state,
            "status": "error",
            "error": f"PM_Agent failed: {exc}",
            "current_agent": "PM_Agent",
        }
