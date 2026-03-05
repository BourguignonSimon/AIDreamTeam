"""Unit tests for the LangGraph multi-agent architecture.

All LLM calls are mocked — no real API key required.
Tests cover:
  - Happy path: PM → Dev → QA (pass) → END
  - QA retry: QA fails once, Dev retries, QA passes on second attempt
  - Circuit breaker: node exceeds max_iterations → Error_Handler
  - Error propagation: node raises exception → Error_Handler
  - Router functions: direct unit tests of each routing helper
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.circuit_breaker import circuit_breaker
from agents.graph import (
    build_graph,
    route_after_dev,
    route_after_pm,
    route_after_qa,
)
from agents.state import MAX_ITERATIONS, AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> AgentState:
    state = AgentState(
        task="Build a hello-world API endpoint",
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


def _mock_litellm_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Router unit tests
# ---------------------------------------------------------------------------

class TestRouters:
    def test_route_after_pm_success(self):
        state = _base_state(status="running", spec='{"user_story": "..."}')
        assert route_after_pm(state) == "Dev_Agent"

    def test_route_after_pm_error(self):
        state = _base_state(status="error")
        assert route_after_pm(state) == "Error_Handler"

    def test_route_after_pm_circuit_open(self):
        state = _base_state(status="circuit_open")
        assert route_after_pm(state) == "Error_Handler"

    def test_route_after_dev_success(self):
        state = _base_state(status="running", code="def hello(): pass")
        assert route_after_dev(state) == "QA_Agent"

    def test_route_after_dev_error(self):
        state = _base_state(status="error")
        assert route_after_dev(state) == "Error_Handler"

    def test_route_after_qa_passed(self):
        from langgraph.graph import END
        state = _base_state(status="running", qa_passed=True)
        assert route_after_qa(state) == END

    def test_route_after_qa_failed_with_retries(self):
        state = _base_state(status="running", qa_passed=False, dev_iterations=1)
        assert route_after_qa(state) == "Dev_Agent"

    def test_route_after_qa_failed_no_retries(self):
        state = _base_state(
            status="running", qa_passed=False, dev_iterations=MAX_ITERATIONS
        )
        assert route_after_qa(state) == "Error_Handler"

    def test_route_after_qa_circuit_open(self):
        state = _base_state(status="circuit_open", qa_passed=False)
        assert route_after_qa(state) == "Error_Handler"


# ---------------------------------------------------------------------------
# Circuit-breaker unit tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_allows_under_limit(self):
        call_count = 0

        @circuit_breaker("pm_iterations")
        def dummy_node(state: AgentState) -> AgentState:
            nonlocal call_count
            call_count += 1
            return {**state, "status": "running"}

        state = _base_state(pm_iterations=0)
        result = dummy_node(state)
        assert call_count == 1
        assert result["status"] == "running"
        assert result["pm_iterations"] == 1

    def test_increments_counter(self):
        @circuit_breaker("dev_iterations")
        def dummy_node(state: AgentState) -> AgentState:
            return {**state, "status": "running"}

        state = _base_state(dev_iterations=2)
        result = dummy_node(state)
        assert result["dev_iterations"] == 3

    def test_trips_at_limit(self):
        called = False

        @circuit_breaker("qa_iterations")
        def dummy_node(state: AgentState) -> AgentState:
            nonlocal called
            called = True
            return {**state, "status": "running"}

        state = _base_state(qa_iterations=MAX_ITERATIONS)  # already at limit
        result = dummy_node(state)

        assert not called, "Node should not execute when circuit is open"
        assert result["status"] == "circuit_open"
        assert "circuit breaker" in result["error"].lower()

    def test_trips_one_before_limit_does_not_trip(self):
        @circuit_breaker("pm_iterations")
        def dummy_node(state: AgentState) -> AgentState:
            return {**state, "status": "running"}

        state = _base_state(pm_iterations=MAX_ITERATIONS - 1)
        result = dummy_node(state)
        assert result["status"] == "running"


# ---------------------------------------------------------------------------
# Node unit tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestPMAgent:
    @patch("agents.pm_agent.litellm.completion")
    def test_happy_path(self, mock_completion):
        from agents.pm_agent import pm_agent_node

        spec_json = '{"user_story": "As a dev...", "acceptance_criteria": ["A"], "out_of_scope": []}'
        mock_completion.return_value = _mock_litellm_response(spec_json)

        result = pm_agent_node(_base_state())

        assert result["status"] == "running"
        assert result["spec"] is not None
        assert result["current_agent"] == "PM_Agent"
        assert result["pm_iterations"] == 1

    @patch("agents.pm_agent.litellm.completion")
    def test_llm_exception_sets_error(self, mock_completion):
        from agents.pm_agent import pm_agent_node

        mock_completion.side_effect = RuntimeError("API timeout")
        result = pm_agent_node(_base_state())

        assert result["status"] == "error"
        assert "PM_Agent failed" in result["error"]

    def test_circuit_breaker_prevents_call(self):
        from agents.pm_agent import pm_agent_node

        state = _base_state(pm_iterations=MAX_ITERATIONS)
        result = pm_agent_node(state)

        assert result["status"] == "circuit_open"


class TestDevAgent:
    @patch("agents.dev_agent.litellm.completion")
    def test_happy_path(self, mock_completion):
        from agents.dev_agent import dev_agent_node

        mock_completion.return_value = _mock_litellm_response("def hello(): return 'world'")
        state = _base_state(spec='{"user_story": "...", "acceptance_criteria": []}')

        result = dev_agent_node(state)

        assert result["status"] == "running"
        assert "hello" in result["code"]
        assert result["current_agent"] == "Dev_Agent"

    def test_empty_spec_returns_error(self):
        from agents.dev_agent import dev_agent_node

        result = dev_agent_node(_base_state(spec=None))
        assert result["status"] == "error"

    def test_circuit_breaker_trips(self):
        from agents.dev_agent import dev_agent_node

        result = dev_agent_node(_base_state(dev_iterations=MAX_ITERATIONS))
        assert result["status"] == "circuit_open"


class TestQAAgent:
    @patch("agents.qa_agent.litellm.completion")
    def test_qa_passes(self, mock_completion):
        from agents.qa_agent import qa_agent_node

        report_json = '{"passed": true, "criteria_results": {}, "issues": [], "suggestions": []}'
        mock_completion.return_value = _mock_litellm_response(report_json)
        state = _base_state(spec="some spec", code="def hello(): pass")

        result = qa_agent_node(state)

        assert result["qa_passed"] is True
        assert result["status"] == "running"

    @patch("agents.qa_agent.litellm.completion")
    def test_qa_fails(self, mock_completion):
        from agents.qa_agent import qa_agent_node

        report_json = '{"passed": false, "criteria_results": {}, "issues": ["Missing docstring"], "suggestions": []}'
        mock_completion.return_value = _mock_litellm_response(report_json)
        state = _base_state(spec="some spec", code="def hello(): pass")

        result = qa_agent_node(state)

        assert result["qa_passed"] is False

    def test_empty_code_returns_error(self):
        from agents.qa_agent import qa_agent_node

        result = qa_agent_node(_base_state(code=None))
        assert result["status"] == "error"

    def test_circuit_breaker_trips(self):
        from agents.qa_agent import qa_agent_node

        result = qa_agent_node(_base_state(qa_iterations=MAX_ITERATIONS, code="x=1"))
        assert result["status"] == "circuit_open"


class TestErrorHandler:
    @patch("agents.error_handler.litellm.completion")
    def test_produces_summary(self, mock_completion):
        from agents.error_handler import error_handler_node

        mock_completion.return_value = _mock_litellm_response("Incident summary here.")
        state = _base_state(status="error", error="Some error", current_agent="Dev_Agent")

        result = error_handler_node(state)

        assert result["result"] == "Incident summary here."
        assert result["current_agent"] == "Error_Handler"

    @patch("agents.error_handler.litellm.completion")
    def test_fallback_summary_on_llm_error(self, mock_completion):
        from agents.error_handler import error_handler_node

        mock_completion.side_effect = RuntimeError("LLM down")
        state = _base_state(status="error", error="Something broke", current_agent="QA_Agent")

        result = error_handler_node(state)

        assert result["result"] is not None
        assert len(result["result"]) > 0


# ---------------------------------------------------------------------------
# Integration test — graph build
# ---------------------------------------------------------------------------

class TestGraphBuild:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    @patch("agents.qa_agent.litellm.completion")
    @patch("agents.dev_agent.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_full_happy_path(self, mock_pm, mock_dev, mock_qa):
        spec_json = '{"user_story": "As a user...", "acceptance_criteria": ["Works"], "out_of_scope": []}'
        mock_pm.return_value = _mock_litellm_response(spec_json)
        mock_dev.return_value = _mock_litellm_response("def run(): pass")
        qa_json = '{"passed": true, "criteria_results": {}, "issues": [], "suggestions": []}'
        mock_qa.return_value = _mock_litellm_response(qa_json)

        from agents.graph import run_graph

        result = run_graph("Create a run() function")

        assert result["qa_passed"] is True
        assert result["status"] == "running"
        assert result["code"] is not None
        assert result["spec"] is not None

    @patch("agents.error_handler.litellm.completion")
    @patch("agents.pm_agent.litellm.completion")
    def test_pm_circuit_open_routes_to_error_handler(self, mock_pm, mock_err):
        mock_pm.side_effect = RuntimeError("LLM unavailable")
        mock_err.return_value = _mock_litellm_response("PM failed, please retry.")

        from agents.graph import _initial_state, build_graph

        graph = build_graph()
        init = _initial_state("test task")
        init["pm_iterations"] = MAX_ITERATIONS  # pre-trip the circuit

        result = graph.invoke(init)

        assert result["status"] in {"error", "circuit_open"}
        assert result["current_agent"] == "Error_Handler"
