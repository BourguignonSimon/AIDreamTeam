"""Unit tests for workers.audio_worker.

All Redis interactions and TranscriptionService calls are mocked so that
tests run without a live Redis instance or GPU.
"""
from __future__ import annotations

import base64
import json
import logging
from unittest.mock import MagicMock, call, patch

import pytest

# Patch heavy dependencies before importing the module under test
import sys

# Stub out faster_whisper so the module can be imported without the library
sys.modules.setdefault("faster_whisper", MagicMock())

from workers import audio_worker  # noqa: E402  (after sys.modules patch)
from workers.audio_worker import (  # noqa: E402
    AudioWorker,
    _delivery_count,
    _ensure_group,
    _handle_message,
    _JsonFormatter,
    _move_to_dlq,
    _process,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_redis():
    r = MagicMock()
    r.xpending_range.return_value = [{"times_delivered": 1}]
    return r


@pytest.fixture()
def mock_service():
    svc = MagicMock()
    result = MagicMock()
    result.language = "fr"
    result.duration = 5.0
    result.segments = [MagicMock(), MagicMock()]
    svc.transcribe_bytes.return_value = result
    return svc


@pytest.fixture()
def audio_b64():
    """Minimal valid base-64 payload (empty bytes — fine for mocked service)."""
    return base64.b64encode(b"fake-audio").decode()


# ---------------------------------------------------------------------------
# _JsonFormatter
# ---------------------------------------------------------------------------

class TestJsonFormatter:
    def test_format_basic(self):
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello %s", args=("world",), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["msg"] == "hello world"
        assert parsed["level"] == "INFO"
        assert "ts" in parsed

    def test_format_extra_fields(self):
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="oops", args=(), exc_info=None,
        )
        record.event = "message_error"
        record.msg_id = "1234-0"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["event"] == "message_error"
        assert parsed["msg_id"] == "1234-0"

    def test_format_with_exception(self):
        formatter = _JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="error", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exc" in parsed
        assert "ValueError" in parsed["exc"]


# ---------------------------------------------------------------------------
# _ensure_group
# ---------------------------------------------------------------------------

class TestEnsureGroup:
    def test_creates_group_when_missing(self, mock_redis):
        mock_redis.xgroup_create.return_value = True
        _ensure_group(mock_redis)
        mock_redis.xgroup_create.assert_called_once_with(
            audio_worker.STREAM, audio_worker.GROUP, id="0", mkstream=True
        )

    def test_ignores_busygroup_error(self, mock_redis):
        from redis.exceptions import ResponseError
        mock_redis.xgroup_create.side_effect = ResponseError("BUSYGROUP Consumer Group already exists")
        # Should not raise
        _ensure_group(mock_redis)

    def test_propagates_other_redis_errors(self, mock_redis):
        from redis.exceptions import ResponseError
        mock_redis.xgroup_create.side_effect = ResponseError("WRONGTYPE")
        with pytest.raises(ResponseError):
            _ensure_group(mock_redis)


# ---------------------------------------------------------------------------
# _delivery_count
# ---------------------------------------------------------------------------

class TestDeliveryCount:
    def test_returns_times_delivered(self, mock_redis):
        mock_redis.xpending_range.return_value = [{"times_delivered": 3}]
        assert _delivery_count(mock_redis, "1234-0") == 3

    def test_defaults_to_1_on_empty(self, mock_redis):
        mock_redis.xpending_range.return_value = []
        assert _delivery_count(mock_redis, "1234-0") == 1

    def test_defaults_to_1_on_exception(self, mock_redis):
        mock_redis.xpending_range.side_effect = Exception("oops")
        assert _delivery_count(mock_redis, "1234-0") == 1


# ---------------------------------------------------------------------------
# _move_to_dlq
# ---------------------------------------------------------------------------

class TestMoveToDlq:
    def test_xadd_and_xack(self, mock_redis):
        fields = {"task_id": "t1", "audio_b64": "abc"}
        _move_to_dlq(mock_redis, "1234-0", fields, "something went wrong")

        mock_redis.xadd.assert_called_once()
        xadd_call = mock_redis.xadd.call_args
        assert xadd_call.args[0] == audio_worker.DLQ_STREAM
        dlq_payload = xadd_call.args[1]
        assert dlq_payload["original_id"] == "1234-0"
        assert "something went wrong" in dlq_payload["error"]

        mock_redis.xack.assert_called_once_with(
            audio_worker.STREAM, audio_worker.GROUP, "1234-0"
        )

    def test_error_truncated_to_500_chars(self, mock_redis):
        _move_to_dlq(mock_redis, "1234-0", {}, "x" * 1000)
        dlq_payload = mock_redis.xadd.call_args.args[1]
        assert len(dlq_payload["error"]) == 500


# ---------------------------------------------------------------------------
# _process
# ---------------------------------------------------------------------------

class TestProcess:
    def test_calls_transcribe_bytes(self, mock_service, audio_b64):
        fields = {"task_id": "t1", "audio_b64": audio_b64, "language": "fr"}
        _process(mock_service, "1234-0", fields)
        mock_service.transcribe_bytes.assert_called_once()
        call_kwargs = mock_service.transcribe_bytes.call_args
        assert call_kwargs.kwargs.get("language") == "fr" or call_kwargs.args[1] == "fr"

    def test_empty_language_becomes_none(self, mock_service, audio_b64):
        fields = {"task_id": "t1", "audio_b64": audio_b64, "language": ""}
        _process(mock_service, "1234-0", fields)
        # language=None should have been passed
        call_args = mock_service.transcribe_bytes.call_args
        language_arg = call_args.kwargs.get("language") or call_args.args[1] if len(call_args.args) > 1 else None
        assert language_arg is None

    def test_raises_on_transcription_failure(self, mock_service, audio_b64):
        mock_service.transcribe_bytes.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(RuntimeError, match="GPU OOM"):
            _process(mock_service, "1234-0", {"audio_b64": audio_b64})


# ---------------------------------------------------------------------------
# _handle_message
# ---------------------------------------------------------------------------

class TestHandleMessage:
    def test_xack_on_success(self, mock_redis, mock_service, audio_b64):
        fields = {"task_id": "t1", "audio_b64": audio_b64}
        _handle_message(mock_redis, mock_service, "1234-0", fields)
        mock_redis.xack.assert_called_once_with(
            audio_worker.STREAM, audio_worker.GROUP, "1234-0"
        )

    def test_no_xack_on_failure_below_max_retries(self, mock_redis, mock_service, audio_b64):
        mock_service.transcribe_bytes.side_effect = RuntimeError("fail")
        mock_redis.xpending_range.return_value = [{"times_delivered": 1}]
        fields = {"audio_b64": audio_b64}
        _handle_message(mock_redis, mock_service, "1234-0", fields)
        mock_redis.xack.assert_not_called()
        mock_redis.xadd.assert_not_called()  # no DLQ yet

    def test_dlq_after_max_retries(self, mock_redis, mock_service, audio_b64):
        mock_service.transcribe_bytes.side_effect = RuntimeError("persistent fail")
        mock_redis.xpending_range.return_value = [
            {"times_delivered": audio_worker.MAX_RETRIES}
        ]
        fields = {"audio_b64": audio_b64}
        _handle_message(mock_redis, mock_service, "1234-0", fields)
        # Should XADD to DLQ and XACK
        mock_redis.xadd.assert_called_once()
        mock_redis.xack.assert_called_once()


# ---------------------------------------------------------------------------
# AudioWorker._autoclaim
# ---------------------------------------------------------------------------

class TestAudioWorkerAutoclaim:
    def test_processes_claimed_messages(self, mock_service, audio_b64):
        worker = AudioWorker(mock_service)
        r = MagicMock()
        claimed_msg = ("9999-0", {"audio_b64": audio_b64, "task_id": "t99"})
        r.xautoclaim.return_value = ("0-0", [claimed_msg], [])
        r.xpending_range.return_value = [{"times_delivered": 1}]

        worker._autoclaim(r)

        r.xautoclaim.assert_called_once()
        mock_service.transcribe_bytes.assert_called_once()

    def test_handles_autoclaim_exception_gracefully(self, mock_service):
        worker = AudioWorker(mock_service)
        r = MagicMock()
        r.xautoclaim.side_effect = Exception("NOSCRIPT")

        # Should not raise
        worker._autoclaim(r)

    def test_no_messages_claimed(self, mock_service):
        worker = AudioWorker(mock_service)
        r = MagicMock()
        r.xautoclaim.return_value = ("0-0", [], [])

        worker._autoclaim(r)
        mock_service.transcribe_bytes.assert_not_called()


# ---------------------------------------------------------------------------
# AudioWorker reconnect logic (integration-style)
# ---------------------------------------------------------------------------

class TestAudioWorkerReconnect:
    def test_stops_on_stop_signal(self, mock_service):
        """Worker exits cleanly when stop() is called before the first connect."""
        worker = AudioWorker(mock_service)

        with patch.object(audio_worker, "_build_client") as mock_build, \
             patch.object(audio_worker, "_ensure_group"), \
             patch("time.sleep"):
            mock_r = MagicMock()
            mock_r.ping.side_effect = lambda: worker.stop()  # stop on first ping
            mock_build.return_value = mock_r

            worker.run()  # should return quickly

        assert not worker._running

    def test_reconnects_after_connection_error(self, mock_service):
        """Worker retries after a RedisConnectionError with back-off sleep."""
        worker = AudioWorker(mock_service)
        call_count = {"n": 0}

        def fake_ping():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RedisConnectionError("connection refused")
            worker.stop()  # succeed on second attempt then stop

        with patch.object(audio_worker, "_build_client") as mock_build, \
             patch.object(audio_worker, "_ensure_group"), \
             patch("time.sleep") as mock_sleep:
            mock_r = MagicMock()
            mock_r.ping.side_effect = fake_ping
            mock_build.return_value = mock_r

            worker.run()

        # sleep was called at least once for the reconnect back-off
        mock_sleep.assert_called()
        assert call_count["n"] == 2
