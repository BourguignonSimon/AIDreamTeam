"""QA Agent node — verifies generated code against the specification.

Responsibilities (constraint R06 from claude.md):
- ALL code MUST pass QA before reaching any Sandbox.
- Read state["code"] and state["spec"].
- Produce a structured QA report in state["qa_report"].
- Set state["qa_passed"] = True only when ALL criteria pass.
- On failure: increment qa_iterations and route back to Dev_Agent (retry)
  or to Error_Handler when circuit opens.
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
You are a senior QA engineer and code reviewer. Given a product specification
and the code produced by a developer, evaluate the code and return a JSON
object with exactly these keys:
  - "passed": boolean — true only if ALL acceptance criteria are satisfied
  - "criteria_results": object mapping each criterion to true/false
  - "issues": list of strings describing what is wrong (empty if passed=true)
  - "suggestions": list of improvement suggestions (may be empty)

Respond with valid JSON only — no markdown fences, no extra text.
"""


@circuit_breaker("qa_iterations")
def qa_agent_node(state: AgentState) -> AgentState:
    """LangGraph node — QA Agent."""
    code = state.get("code") or ""
    spec = state.get("spec") or ""

    if not code:
        return {
            **state,
            "status": "error",
            "error": "QA_Agent received empty code from Dev_Agent.",
            "current_agent": "QA_Agent",
            "qa_passed": False,
        }

    logger.info("QA_Agent reviewing code (len=%d)", len(code))

    try:
        response = litellm.completion(
            model=os.getenv("QA_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Specification:\n{spec}\n\n"
                        f"Code to review:\n```python\n{code}\n```"
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        report = json.loads(raw)
        passed: bool = bool(report.get("passed", False))
        report_text = json.dumps(report, indent=2, ensure_ascii=False)

        logger.info("QA_Agent verdict: passed=%s", passed)

        return {
            **state,
            "qa_report": report_text,
            "qa_passed": passed,
            "status": "running",
            "current_agent": "QA_Agent",
            "messages": [
                {
                    "role": "assistant",
                    "agent": "QA_Agent",
                    "content": f"QA report (passed={passed}):\n{report_text}",
                }
            ],
        }

    except Exception as exc:
        logger.error("QA_Agent error: %s", exc, exc_info=True)
        return {
            **state,
            "status": "error",
            "error": f"QA_Agent failed: {exc}",
            "current_agent": "QA_Agent",
            "qa_passed": False,
        }
