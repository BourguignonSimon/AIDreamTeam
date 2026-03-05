"""Tests for the React Sandbox WebSocket endpoint (/ws/codegen).

Coverage
--------
- Connection lifecycle (connect / disconnect).
- Streaming: valid prompt → sequence of token messages → done.
- Error handling: empty prompt, invalid JSON.
- Component selection: keyword-based dispatch.
- Token streaming latency: each chunk is non-empty.

Note: we build a *minimal* FastAPI app that mounts only the sandbox router so
that the transcription service (and its heavy numpy/faster-whisper deps) is
not imported during testing.
"""
from __future__ import annotations

import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.sandbox_ws import router as sandbox_router

# ---------------------------------------------------------------------------
# Minimal test app — avoids importing transcription dependencies
# ---------------------------------------------------------------------------

_test_app = FastAPI()
_test_app.include_router(sandbox_router)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def ws_client():
    """Yield a synchronous TestClient with WebSocket support."""
    with TestClient(_test_app) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_ws_messages(ws_client: TestClient, prompt: str) -> list[dict]:
    """Open a WS connection, send a prompt, and collect all server messages."""
    messages: list[dict] = []
    with ws_client.websocket_connect("/ws/codegen") as ws:
        ws.send_text(json.dumps({"prompt": prompt}))
        while True:
            raw = ws.receive_text()
            msg = json.loads(raw)
            messages.append(msg)
            if msg["type"] in ("done", "error"):
                break
    return messages


# ---------------------------------------------------------------------------
# Tests — basic streaming protocol
# ---------------------------------------------------------------------------

class TestCodegenWebSocket:
    def test_valid_prompt_yields_tokens_then_done(self, ws_client):
        msgs = collect_ws_messages(ws_client, "build a counter")

        types = [m["type"] for m in msgs]
        assert "token" in types, "At least one token should be emitted"
        assert types[-1] == "done", "Last message must be 'done'"

    def test_tokens_are_non_empty_strings(self, ws_client):
        msgs = collect_ws_messages(ws_client, "counter")
        tokens = [m for m in msgs if m["type"] == "token"]

        assert tokens, "Expected token messages"
        for tok in tokens:
            assert isinstance(tok["content"], str)
            assert len(tok["content"]) > 0

    def test_accumulated_code_is_non_empty(self, ws_client):
        msgs = collect_ws_messages(ws_client, "counter")
        code = "".join(m["content"] for m in msgs if m["type"] == "token")

        assert len(code) > 50, "Streamed code should be substantial"

    def test_accumulated_code_contains_jsx(self, ws_client):
        msgs = collect_ws_messages(ws_client, "counter")
        code = "".join(m["content"] for m in msgs if m["type"] == "token")

        assert "function App" in code, "Should contain App component definition"
        assert "return (" in code or "return(" in code, "Should contain JSX return"

    def test_done_message_has_no_extra_fields(self, ws_client):
        msgs = collect_ws_messages(ws_client, "counter")
        done = msgs[-1]

        assert done["type"] == "done"
        # 'done' should not carry a 'content' or 'detail' key
        assert "content" not in done


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------

class TestCodegenErrors:
    def test_empty_prompt_returns_error(self, ws_client):
        msgs = collect_ws_messages(ws_client, "")

        assert len(msgs) == 1
        assert msgs[0]["type"] == "error"
        assert "empty" in msgs[0]["detail"].lower()

    def test_whitespace_only_prompt_returns_error(self, ws_client):
        msgs = collect_ws_messages(ws_client, "   ")

        assert msgs[0]["type"] == "error"

    def test_invalid_json_returns_error(self, ws_client):
        with ws_client.websocket_connect("/ws/codegen") as ws:
            ws.send_text("not json at all")
            raw = ws.receive_text()
            msg = json.loads(raw)

        assert msg["type"] == "error"
        assert "json" in msg["detail"].lower() or "invalid" in msg["detail"].lower()

    def test_missing_prompt_key_returns_error(self, ws_client):
        with ws_client.websocket_connect("/ws/codegen") as ws:
            ws.send_text(json.dumps({"question": "build a counter"}))
            raw = ws.receive_text()
            msg = json.loads(raw)

        assert msg["type"] == "error"

    def test_multiple_requests_on_same_connection(self, ws_client):
        """The server should handle multiple prompts on a single WS connection."""
        with ws_client.websocket_connect("/ws/codegen") as ws:
            for prompt in ("counter", "todo"):
                ws.send_text(json.dumps({"prompt": prompt}))
                types: list[str] = []
                while True:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    types.append(msg["type"])
                    if msg["type"] in ("done", "error"):
                        break
                assert "token" in types
                assert types[-1] == "done"


# ---------------------------------------------------------------------------
# Tests — component selection
# ---------------------------------------------------------------------------

class TestComponentSelection:
    @pytest.mark.parametrize("keyword,expected_in_code", [
        ("counter", "setCount"),
        ("todo",    "setItems"),
        ("timer",   "setSeconds"),
    ])
    def test_keyword_dispatches_correct_component(
        self, ws_client, keyword: str, expected_in_code: str
    ):
        msgs = collect_ws_messages(ws_client, f"build a {keyword}")
        code = "".join(m["content"] for m in msgs if m["type"] == "token")

        assert expected_in_code in code, (
            f"Expected '{expected_in_code}' in code for keyword '{keyword}'"
        )

    def test_unknown_keyword_falls_back_to_default(self, ws_client):
        """Any unrecognised prompt should still stream valid code."""
        msgs = collect_ws_messages(ws_client, "something completely unknown")
        code = "".join(m["content"] for m in msgs if m["type"] == "token")

        assert "function App" in code


# ---------------------------------------------------------------------------
# Tests — streaming integrity
# ---------------------------------------------------------------------------

class TestStreamingIntegrity:
    def test_no_message_after_done(self, ws_client):
        """After 'done', the server should not emit further messages for that request."""
        with ws_client.websocket_connect("/ws/codegen") as ws:
            ws.send_text(json.dumps({"prompt": "counter"}))
            messages: list[dict] = []
            while True:
                raw = ws.receive_text()
                msg = json.loads(raw)
                messages.append(msg)
                if msg["type"] == "done":
                    break
            # Send a fresh prompt; the server must not replay previous tokens.
            ws.send_text(json.dumps({"prompt": "todo"}))
            first = json.loads(ws.receive_text())

        # First message of the new stream must be a token, not a stale 'done'.
        assert first["type"] == "token"

    def test_streamed_code_equals_expected_component(self, ws_client):
        """The concatenated tokens must equal the full component string."""
        from api.sandbox_ws import _pick_component  # noqa: PLC0415

        expected = _pick_component("counter")
        msgs = collect_ws_messages(ws_client, "counter")
        accumulated = "".join(m["content"] for m in msgs if m["type"] == "token")

        assert accumulated == expected
