"""Redis Streams worker — 'audio-tasks' queue.

Consumer group : audio-workers
Stream         : audio-tasks          (REDIS_STREAM)
DLQ            : audio-tasks:dlq      (REDIS_DLQ_STREAM)
Consumer ID    : audio-worker-<hostname>-<pid>

Message format (XADD fields)
----------------------------
  task_id   str  Opaque identifier for correlation / idempotency
  audio_b64 str  Base-64-encoded audio file bytes
  language  str  ISO 639-1 code, or "" / absent for auto-detect (optional)

Happy path
----------
  1. XREADGROUP reads one message at a time (blocking up to REDIS_BLOCK_MS).
  2. Audio bytes are decoded and fed to TranscriptionService.transcribe_bytes().
  3. XACK acknowledges the message on success.

Failure / DLQ
-------------
  On exception the message is left un-ACKed (stays in the Pending Entry List).
  A periodic XAUTOCLAIM pass reclaims messages idle > AUTOCLAIM_IDLE_MS and
  re-delivers them to this consumer.  Once a message has been delivered
  MAX_RETRIES times it is written to the DLQ stream and XACK-ed so it is
  removed from the PEL permanently.

Redis reconnection
------------------
  Any ConnectionError in the main loop triggers an exponential back-off
  reconnect sequence: 1 s → 2 s → 4 s … capped at RECONNECT_MAX_DELAY_S.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import signal
import socket
import sys
import time
from typing import Any, Optional

import redis
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.retry import Retry

from services.transcription import TranscriptionService

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------

REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

STREAM: str = os.getenv("REDIS_STREAM", "audio-tasks")
DLQ_STREAM: str = os.getenv("REDIS_DLQ_STREAM", "audio-tasks:dlq")
GROUP: str = os.getenv("REDIS_CONSUMER_GROUP", "audio-workers")
CONSUMER_ID: str = os.getenv(
    "REDIS_CONSUMER_ID",
    f"audio-worker-{socket.gethostname()}-{os.getpid()}",
)

MAX_RETRIES: int = int(os.getenv("AUDIO_WORKER_MAX_RETRIES", "3"))
BLOCK_MS: int = int(os.getenv("REDIS_BLOCK_MS", "5000"))
AUTOCLAIM_IDLE_MS: int = int(os.getenv("AUTOCLAIM_IDLE_MS", str(60_000)))  # 60 s
AUTOCLAIM_INTERVAL_S: float = float(os.getenv("AUTOCLAIM_INTERVAL_S", "30"))

RECONNECT_BASE_DELAY_S: float = 1.0
RECONNECT_MAX_DELAY_S: float = float(os.getenv("RECONNECT_MAX_DELAY_S", "60"))


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

_LOG_RESERVED = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "id", "levelname", "levelno", "lineno", "message",
    "module", "msecs", "msg", "name", "pathname", "process",
    "processName", "relativeCreated", "stack_info", "thread",
    "threadName", "taskName",
})


class _JsonFormatter(logging.Formatter):
    """Emit one compact JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "consumer": CONSUMER_ID,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _LOG_RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    root.handlers = [handler]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redis client factory
# ---------------------------------------------------------------------------

def _build_client() -> redis.Redis:
    """Return a Redis client with command-level retry on transient errors."""
    retry = Retry(ExponentialBackoff(cap=10, base=0.5), retries=6)
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=True,
        retry=retry,
        retry_on_error=[RedisConnectionError, TimeoutError],
        socket_connect_timeout=5,
        socket_timeout=BLOCK_MS / 1000 + 2,  # slightly above BLOCK timeout
    )


# ---------------------------------------------------------------------------
# Consumer group bootstrap
# ---------------------------------------------------------------------------

def _ensure_group(r: redis.Redis) -> None:
    """Create the consumer group if it does not already exist (idempotent)."""
    try:
        # id="0" means: the group will see all existing messages on first run.
        # Use id="$" if you only want messages arriving after the worker starts.
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        logger.info(
            "Consumer group created",
            extra={"event": "group_created", "stream": STREAM, "group": GROUP},
        )
    except ResponseError as exc:
        if "BUSYGROUP" in str(exc):
            logger.debug(
                "Consumer group already exists",
                extra={"event": "group_exists", "stream": STREAM, "group": GROUP},
            )
        else:
            raise


# ---------------------------------------------------------------------------
# PEL helpers
# ---------------------------------------------------------------------------

def _delivery_count(r: redis.Redis, msg_id: str) -> int:
    """Return how many times *msg_id* has been delivered; defaults to 1."""
    try:
        entries = r.xpending_range(STREAM, GROUP, min=msg_id, max=msg_id, count=1)
        if entries:
            return int(entries[0].get("times_delivered", 1))
    except Exception:
        pass
    return 1


def _move_to_dlq(
    r: redis.Redis, msg_id: str, fields: dict[str, str], error: str
) -> None:
    """Write a failed message to the DLQ and remove it from the source PEL."""
    dlq_fields: dict[str, str] = {
        **fields,
        "original_id": msg_id,
        "error": error[:500],
        "failed_at": str(time.time()),
        "consumer": CONSUMER_ID,
    }
    r.xadd(DLQ_STREAM, dlq_fields)
    r.xack(STREAM, GROUP, msg_id)
    logger.error(
        "Message moved to DLQ",
        extra={
            "event": "dlq",
            "stream": STREAM,
            "dlq_stream": DLQ_STREAM,
            "msg_id": msg_id,
            "error": error[:500],
        },
    )


# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------

def _process(
    service: TranscriptionService,
    msg_id: str,
    fields: dict[str, str],
) -> None:
    """Decode audio and transcribe; raises on any failure."""
    task_id = fields.get("task_id", msg_id)
    audio_b64 = fields.get("audio_b64", "")
    language = fields.get("language") or None

    logger.info(
        "Processing message",
        extra={
            "event": "message_received",
            "msg_id": msg_id,
            "task_id": task_id,
            "language": language,
            "audio_b64_len": len(audio_b64),
        },
    )

    t0 = time.perf_counter()
    audio_bytes = base64.b64decode(audio_b64)
    result = service.transcribe_bytes(audio_bytes, language=language)
    elapsed = round(time.perf_counter() - t0, 3)

    logger.info(
        "Message processed successfully",
        extra={
            "event": "message_done",
            "msg_id": msg_id,
            "task_id": task_id,
            "language": result.language,
            "audio_duration_s": result.duration,
            "processing_time_s": elapsed,
            "segments": len(result.segments),
        },
    )


def _handle_message(
    r: redis.Redis,
    service: TranscriptionService,
    msg_id: str,
    fields: dict[str, str],
) -> None:
    """Process one message: XACK on success, DLQ after MAX_RETRIES failures."""
    try:
        _process(service, msg_id, fields)
        r.xack(STREAM, GROUP, msg_id)
        logger.info(
            "Message ACK-ed",
            extra={"event": "xack", "msg_id": msg_id},
        )
    except Exception as exc:
        count = _delivery_count(r, msg_id)
        logger.warning(
            "Message processing failed",
            extra={
                "event": "message_error",
                "msg_id": msg_id,
                "error": str(exc),
                "delivery_count": count,
                "max_retries": MAX_RETRIES,
            },
            exc_info=True,
        )
        if count >= MAX_RETRIES:
            _move_to_dlq(r, msg_id, fields, str(exc))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class AudioWorker:
    """Long-running Redis Streams consumer for the 'audio-tasks' queue."""

    def __init__(self, service: TranscriptionService) -> None:
        self._service = service
        self._running = True
        self._r: Optional[redis.Redis] = None
        self._last_autoclaim_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def stop(self) -> None:
        logger.info("Stop signal received", extra={"event": "worker_stopping"})
        self._running = False

    def run(self) -> None:
        """Block and process messages until :meth:`stop` is called."""
        _configure_logging()
        logger.info(
            "Audio worker starting",
            extra={
                "event": "worker_start",
                "stream": STREAM,
                "group": GROUP,
                "consumer": CONSUMER_ID,
                "max_retries": MAX_RETRIES,
            },
        )

        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        delay = RECONNECT_BASE_DELAY_S
        while self._running:
            try:
                self._r = _build_client()
                self._r.ping()
                _ensure_group(self._r)
                delay = RECONNECT_BASE_DELAY_S  # reset on successful connect
                logger.info(
                    "Connected to Redis",
                    extra={
                        "event": "redis_connected",
                        "host": REDIS_HOST,
                        "port": REDIS_PORT,
                    },
                )
                self._loop()

            except RedisConnectionError as exc:
                logger.error(
                    "Redis connection lost — retrying",
                    extra={
                        "event": "redis_disconnected",
                        "error": str(exc),
                        "retry_in_s": delay,
                    },
                )
                time.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY_S)

            except Exception as exc:
                logger.exception(
                    "Unexpected worker error",
                    extra={"event": "worker_error", "error": str(exc)},
                )
                time.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY_S)

        logger.info("Audio worker stopped", extra={"event": "worker_stopped"})

    # ------------------------------------------------------------------
    # Inner processing loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        r = self._r
        assert r is not None

        while self._running:
            # Periodic XAUTOCLAIM: recover stale messages from dead consumers
            now = time.time()
            if now - self._last_autoclaim_ts >= AUTOCLAIM_INTERVAL_S:
                self._autoclaim(r)
                self._last_autoclaim_ts = now

            # Read the next new message (blocks up to BLOCK_MS)
            response = r.xreadgroup(
                GROUP,
                CONSUMER_ID,
                {STREAM: ">"},
                count=1,
                block=BLOCK_MS,
            )
            if not response:
                continue  # BLOCK timeout — loop and check SIGTERM

            for _stream_name, messages in response:
                for msg_id, fields in messages:
                    _handle_message(r, self._service, msg_id, fields)

    # ------------------------------------------------------------------
    # XAUTOCLAIM: recover messages from crashed consumers
    # ------------------------------------------------------------------

    def _autoclaim(self, r: redis.Redis) -> None:
        """Reclaim messages idle > AUTOCLAIM_IDLE_MS and re-process them."""
        try:
            # Returns (next_start_id, [(id, fields), ...], [deleted_ids])
            _next_id, claimed, _deleted = r.xautoclaim(
                STREAM,
                GROUP,
                CONSUMER_ID,
                min_idle_time=AUTOCLAIM_IDLE_MS,
                start_id="0-0",
                count=10,
            )
            if claimed:
                logger.info(
                    "XAUTOCLAIM reclaimed messages",
                    extra={"event": "autoclaim", "count": len(claimed)},
                )
            for msg_id, fields in claimed:
                _handle_message(r, self._service, msg_id, fields)

        except Exception as exc:
            logger.warning(
                "XAUTOCLAIM failed",
                extra={"event": "autoclaim_error", "error": str(exc)},
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    service = TranscriptionService()
    worker = AudioWorker(service)
    worker.run()


if __name__ == "__main__":
    main()
