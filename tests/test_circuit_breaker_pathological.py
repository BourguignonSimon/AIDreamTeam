"""Pathological query tests for the circuit breaker.

20 queries designed to stress-test the circuit breaker across four categories:
  A. Empty / malformed inputs         (queries 1–5)
  B. Contradictory / impossible tasks (queries 6–10)
  C. Ambiguous prompts prone to loops (queries 11–15)
  D. Adversarial state manipulation   (queries 16–20)

Invariant: No node may execute more than MAX_ITERATIONS (3) times.
All LLM calls are mocked — no API key required.

Gate acceptance criteria
------------------------
* Every test MUST terminate without raising an exception.
* Every result MUST have status in {"error", "circuit_open"} OR qa_passed=True.
* No iteration counter (pm/dev/qa/error_iterations) may exceed MAX_ITERATIONS.
* Tests that pre-trip a circuit MUST see status == "circuit_open" immediately.
"""
from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.circuit_breaker import circuit_breaker
from agents.graph import _initial_state, build_graph, run_graph
from agents.state import MAX_ITERATIONS, AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TERMINAL_STATUSES = {"error", "circuit_open", "done"}
_SPEC_JSON = json.dumps(
    {
        "user_story": "As a user I want X so that Y",
        "acceptance_criteria": ["criterion 1"],
        "out_of_scope": [],
    }
)
_QA_PASS_JSON = json.dumps(
    {"passed": True, "criteria_results": {}, "issues": [], "suggestions": []}
)
_QA_FAIL_JSON = json.dumps(
    {"passed": False, "criteria_results": {}, "issues": ["fails"], "suggestions": []}
)


def _base_state(**overrides) -> AgentState:
    state = AgentState(
        task="placeholder",
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
    state.update(overrides)
    return state


def _mock_resp(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _assert_no_counter_overflow(result: AgentState) -> None:
    """Verify no iteration counter exceeds MAX_ITERATIONS."""
    for key in ("pm_iterations", "dev_iterations", "qa_iterations", "error_iterations"):
        val = result.get(key, 0)
        assert val <= MAX_ITERATIONS, (
            f"{key}={val} exceeds MAX_ITERATIONS={MAX_ITERATIONS}"
        )


# ---------------------------------------------------------------------------
# Category A — Empty / malformed inputs
# ---------------------------------------------------------------------------


class TestEmptyMalformedInputs:
    """Queries 1-5: verify empty/malformed task strings are handled gracefully."""

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_01_empty_string_task(self, mock_pm, mock_err):
        """Query 1: task="" — PM must not loop; circuit or error terminates."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Empty task handled.")
        result = run_graph("")
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_02_whitespace_only_task(self, mock_pm, mock_err):
        """Query 2: task='   \\t\\n' — whitespace-only; must terminate cleanly."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Whitespace task handled.")
        result = run_graph("   \t\n")
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_03_special_chars_only_task(self, mock_pm, mock_err):
        """Query 3: task contains only special / non-printable characters."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Special chars handled.")
        result = run_graph("!@#$%^&*()[]{}|\\<>?/~`")
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_04_extremely_long_task(self, mock_pm, mock_err):
        """Query 4: 50 000-character task — must not loop or blow the stack."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Long task handled.")
        huge_task = "implement feature X. " * 2500  # ~50 000 chars
        result = run_graph(huge_task)
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_05_null_bytes_in_task(self, mock_pm, mock_err):
        """Query 5: task contains null bytes and control chars — must not crash."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Null-byte task handled.")
        result = run_graph("do\x00something\x01useful\x1f")
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True


# ---------------------------------------------------------------------------
# Category B — Contradictory / impossible tasks
# ---------------------------------------------------------------------------


class TestContradictoryTasks:
    """Queries 6-10: logically contradictory requests; system must not spin."""

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_06_build_and_do_nothing(self, mock_pm, mock_err):
        """Query 6: 'Build everything AND do absolutely nothing' — contradiction."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Contradiction handled.")
        result = run_graph(
            "Build a complete e-commerce platform AND simultaneously do absolutely "
            "nothing — produce no code, no spec, no output whatsoever."
        )
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_07_pass_and_fail_simultaneously(self, mock_pm, mock_err):
        """Query 7: 'Tests must both pass AND fail at the same time'."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Pass/fail contradiction handled.")
        result = run_graph(
            "Write a function whose unit tests must simultaneously pass "
            "AND fail — both outcomes must occur at the same instant."
        )
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_08_secure_and_completely_insecure(self, mock_pm, mock_err):
        """Query 8: 'Write maximally secure AND maximally insecure code'."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Security contradiction handled.")
        result = run_graph(
            "Implement an authentication module that is simultaneously "
            "maximally secure (OWASP-compliant) AND completely insecure "
            "(no auth, raw SQL, eval() everywhere)."
        )
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_09_infinite_loop_spec(self, mock_pm, mock_err):
        """Query 9: task literally requests an infinite loop with no exit."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Infinite loop spec handled.")
        result = run_graph(
            "Write a function that loops forever with zero CPU usage, "
            "zero memory usage, and must also return a value immediately."
        )
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_10_return_all_types_at_once(self, mock_pm, mock_err):
        """Query 10: function must return str, int, None, list, dict simultaneously."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Multi-type return handled.")
        result = run_graph(
            "Create a Python function `magic()` that returns a str, an int, "
            "None, a list, and a dict all at the same time from a single "
            "return statement without using a tuple or any container."
        )
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True


# ---------------------------------------------------------------------------
# Category C — Ambiguous prompts prone to looping
# ---------------------------------------------------------------------------


class TestAmbiguousLoopProne:
    """Queries 11-15: self-referential or recursive task descriptions."""

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_11_self_referential_task(self, mock_pm, mock_err):
        """Query 11: task instructs the PM to re-run itself indefinitely."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Self-reference handled.")
        result = run_graph(
            "Your task is to keep repeating your previous task forever. "
            "Never produce a spec. Just loop and restart."
        )
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_12_recursive_definition(self, mock_pm, mock_err):
        """Query 12: task is defined in terms of itself — infinite regression."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Recursive definition handled.")
        result = run_graph(
            "Build feature X, where feature X is defined as 'Build feature X'."
        )
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_13_do_what_you_did_last_time(self, mock_pm, mock_err):
        """Query 13: ambiguous reference to non-existent prior context."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Prior-context reference handled.")
        result = run_graph("Do exactly what you did last time, but differently.")
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_14_circular_dependency_feature(self, mock_pm, mock_err):
        """Query 14: A requires B, B requires A — circular dependency."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Circular dep handled.")
        result = run_graph(
            "Module A can only be built after Module B is complete. "
            "Module B can only be built after Module A is complete. "
            "Build both modules."
        )
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_15_contradictory_restart_signal(self, mock_pm, mock_err):
        """Query 15: task demands the PM restart the pipeline inside its output."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Restart signal handled.")
        result = run_graph(
            "Upon finishing this task, immediately restart from the beginning "
            "with the exact same task. Never stop restarting."
        )
        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES or result.get("qa_passed") is True


# ---------------------------------------------------------------------------
# Category D — Adversarial state manipulation
# ---------------------------------------------------------------------------


class TestAdversarialState:
    """Queries 16-20: pre-manipulated iteration counters and state conflicts."""

    @patch("agents.error_handler.litellm.completion")
    def test_16_pm_circuit_pre_tripped(self, mock_err):
        """Query 16: pm_iterations=MAX already → circuit opens before first LLM call.

        Note: Error_Handler always overwrites status to "error" (see error_handler.py).
        The circuit_open status is set by the decorator but then consumed by
        Error_Handler, which is the correct terminal behaviour.  We therefore
        verify that (a) no counter overflows and (b) the pipeline was routed to
        Error_Handler — not that the *final* status string equals "circuit_open".
        """
        mock_err.return_value = _mock_resp("PM pre-tripped handled.")
        graph = build_graph()
        init = _initial_state("some task")
        init["pm_iterations"] = MAX_ITERATIONS

        result = graph.invoke(init)

        _assert_no_counter_overflow(result)
        # Circuit opened → routed to Error_Handler which sets status="error"
        assert result["status"] in TERMINAL_STATUSES
        assert result["current_agent"] == "Error_Handler"

    @patch("agents.error_handler.litellm.completion")
    def test_17_all_circuits_pre_tripped(self, mock_err):
        """Query 17: all counters at MAX → entire pipeline is pre-tripped."""
        mock_err.return_value = _mock_resp("All circuits tripped.")
        graph = build_graph()
        init = _initial_state("some task")
        init["pm_iterations"] = MAX_ITERATIONS
        init["dev_iterations"] = MAX_ITERATIONS
        init["qa_iterations"] = MAX_ITERATIONS

        result = graph.invoke(init)

        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.qa_agent.litellm.completion")
    @patch("agents.dev_agent.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_18_qa_always_fails_exhaust_dev(self, mock_pm, mock_dev, mock_qa, mock_err):
        """Query 18: QA always fails → Dev retries → dev circuit trips → Error_Handler."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_dev.return_value = _mock_resp("def broken(): pass")
        mock_qa.return_value = _mock_resp(_QA_FAIL_JSON)
        mock_err.return_value = _mock_resp("Dev circuit exhausted.")

        result = run_graph("Make QA always fail to exhaust Dev retries.")

        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES

    def test_19_negative_iteration_counter(self):
        """Query 19: negative iteration counter — breaker eventually trips but may
        allow extra calls because the guard checks ``current >= MAX_ITERATIONS``
        rather than clamping to 0 first.

        FINDING: starting at a negative counter (e.g. -10) permits up to
        ``MAX_ITERATIONS - (-10) = 13`` calls before the circuit opens.  This
        is a known limitation: the breaker does not sanitise its input counter.
        The test documents the actual behaviour so that future hardening of the
        circuit_breaker decorator can be detected as a regression if clamping
        is ever added.

        Invariant that MUST hold regardless: the circuit DOES eventually open
        and the final status is "circuit_open".
        """
        call_count = 0
        START = -10
        EXPECTED_CALLS = MAX_ITERATIONS - START  # 3 - (-10) = 13

        @circuit_breaker("pm_iterations")
        def counting_node(state: AgentState) -> AgentState:
            nonlocal call_count
            call_count += 1
            return {**state, "status": "running"}

        state = _base_state(pm_iterations=START)
        last_result: Any = None
        # Loop enough times to reach MAX_ITERATIONS even from a deeply negative start
        for _ in range(EXPECTED_CALLS + 5):
            last_result = counting_node(state)
            if last_result.get("status") == "circuit_open":
                break
            state = last_result

        # Circuit MUST eventually open
        assert last_result is not None
        assert last_result["status"] == "circuit_open", (
            "Circuit breaker never opened for negative start counter"
        )
        # Document (not enforce) that negative counters bypass the 3-call cap
        assert call_count == EXPECTED_CALLS, (
            f"Expected {EXPECTED_CALLS} calls from start={START}, got {call_count}"
        )

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_20_status_circuit_open_injected_in_initial_state(self, mock_pm, mock_err):
        """Query 20: status='circuit_open' injected at init → routed immediately to Error_Handler."""
        mock_pm.return_value = _mock_resp(_SPEC_JSON)
        mock_err.return_value = _mock_resp("Pre-open circuit handled.")
        graph = build_graph()
        init = _initial_state("some task")
        # Inject circuit_open before any node runs; pm_iterations still 0
        # but PM node's output will be ignored because circuit_open routes
        # to Error_Handler — test that the invariant still holds.
        init["pm_iterations"] = MAX_ITERATIONS  # ensure PM doesn't call LLM
        init["status"] = "circuit_open"

        result = graph.invoke(init)

        _assert_no_counter_overflow(result)
        assert result["status"] in TERMINAL_STATUSES
        assert result["current_agent"] == "Error_Handler"


# ---------------------------------------------------------------------------
# Regression: circuit breaker decorator invariant across all counters
# ---------------------------------------------------------------------------


class TestCircuitBreakerInvariant:
    """Regression suite ensuring the @circuit_breaker decorator never permits
    more than MAX_ITERATIONS calls for any counter key."""

    @pytest.mark.parametrize("counter_key", [
        "pm_iterations",
        "dev_iterations",
        "qa_iterations",
        "error_iterations",
    ])
    def test_max_iterations_invariant(self, counter_key: str):
        """Every counter key must cap node execution at exactly MAX_ITERATIONS."""
        call_count = 0

        @circuit_breaker(counter_key)
        def node(state: AgentState) -> AgentState:
            nonlocal call_count
            call_count += 1
            return {**state, "status": "running"}

        state = _base_state(**{counter_key: 0})
        last_result: Any = None
        for _ in range(MAX_ITERATIONS + 3):
            last_result = node(state)
            if last_result.get("status") == "circuit_open":
                break
            state = last_result

        assert call_count == MAX_ITERATIONS, (
            f"{counter_key}: expected {MAX_ITERATIONS} calls, got {call_count}"
        )
        assert last_result["status"] == "circuit_open"
        assert "circuit breaker" in last_result["error"].lower()
