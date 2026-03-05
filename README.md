# AIDreamTeam — Autonomous AI Platform (Phase 1)

A multi-agent LangGraph pipeline that transforms plain-text task descriptions into QA-verified code, backed by a GPU-accelerated audio transcription service and a Redis Streams worker.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Tech Stack](#tech-stack)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Services](#running-the-services)
8. [API Reference](#api-reference)
9. [Running Tests](#running-tests)
10. [Debugging Guide](#debugging-guide)
11. [Key Design Constraints](#key-design-constraints)
12. [Phase 1 Exit Gates](#phase-1-exit-gates)

---

## Architecture Overview

### Multi-Agent Pipeline (LangGraph)

```
[START]
   |
   v
PM_Agent  --(error / circuit_open)--> Error_Handler --> [END]
   |
   v (spec ready)
Dev_Agent --(error / circuit_open)--> Error_Handler --> [END]
   |
   v (code ready)
QA_Agent  --(error / circuit_open)--> Error_Handler --> [END]
   |              |
(qa_passed)  (qa_failed, dev retries left)
   |              |
 [END]       Dev_Agent (retry with QA context)
```

Each node is wrapped with a **circuit breaker** (`max_iterations = 3`). When a node is called more than 3 times the decorator short-circuits, sets `status = "circuit_open"`, and the router diverts to `Error_Handler`.

### Audio Transcription Pipeline

```
HTTP client / Redis producer
        |
        v
  FastAPI /transcribe  OR  AudioWorker (Redis Streams consumer)
        |
        v
  TranscriptionService (faster-whisper / CTranslate2)
        |
  chunk audio into 15 s windows (0.5 s overlap)
        |
  WhisperModel.transcribe() per chunk  (VAD filter on)
        |
  re-base timestamps -> TranscriptionResult
```

---

## Project Structure

```
AIDreamTeam/
├── agents/
│   ├── circuit_breaker.py   # @circuit_breaker decorator (max_iterations guard)
│   ├── dev_agent.py         # LangGraph node: generate Python code from spec
│   ├── error_handler.py     # LangGraph node: terminal failure sink
│   ├── graph.py             # StateGraph assembly, routing helpers, run_graph()
│   ├── pm_agent.py          # LangGraph node: produce JSON spec from task
│   ├── qa_agent.py          # LangGraph node: review code against spec
│   └── state.py             # AgentState TypedDict + MAX_ITERATIONS constant
│
├── api/
│   └── main.py              # FastAPI app — POST /transcribe (sync + SSE stream)
│
├── services/
│   └── transcription.py     # TranscriptionService, load_audio(), chunk_audio()
│
├── workers/
│   └── audio_worker.py      # Redis Streams consumer — audio-tasks queue
│
├── tests/
│   ├── test_graph.py                        # Unit + integration tests for agents
│   ├── test_transcription.py                # Unit tests for transcription service
│   ├── test_audio_worker.py                 # Unit tests for Redis worker
│   ├── test_circuit_breaker_pathological.py # 20 pathological circuit-breaker cases
│   ├── audio/generate.py                    # Helper to generate test audio files
│   └── CIRCUIT_BREAKER_PATHOLOGICAL_REPORT.md
│
├── .env.example    # All environment variables (copy to .env, never commit .env)
├── claude.md       # Architecture decisions and project constraints
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 0.2 |
| LLM gateway | LiteLLM (direct or via proxy) |
| Backend API | FastAPI + Uvicorn |
| Transcription engine | faster-whisper (CTranslate2) |
| Message broker | Redis Streams |
| Python runtime | Python 3.11 |
| Frontend (planned) | Next.js 14 + Tailwind CSS |
| GPU infra (planned) | OVH / Scaleway (sovereign cloud) |

---

## Prerequisites

- Python 3.11+
- Redis 7+ (for the audio worker)
- A CUDA-capable GPU is recommended for transcription (`WHISPER_DEVICE=cuda`); CPU mode works with `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8`
- An LLM API key (OpenAI, Anthropic, or a self-hosted LiteLLM proxy)

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd AIDreamTeam

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your actual values (see Configuration section)
```

---

## Configuration

All configuration is done through environment variables. Copy `.env.example` to `.env` and fill in the values.

### LLM Models

| Variable | Default | Description |
|---|---|---|
| `PM_MODEL` | `gpt-4o-mini` | LiteLLM model string for the PM agent |
| `DEV_MODEL` | `gpt-4o-mini` | LiteLLM model string for the Dev agent |
| `QA_MODEL` | `gpt-4o-mini` | LiteLLM model string for the QA agent |
| `ERROR_MODEL` | `gpt-4o-mini` | LiteLLM model string for the Error Handler |

### LiteLLM Proxy (optional)

| Variable | Description |
|---|---|
| `LITELLM_PROXY_URL` | URL of the LiteLLM proxy (e.g. `http://localhost:4000`) |
| `LITELLM_API_KEY` | API key for the proxy |

### API Keys (when calling providers directly)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models |

### Audio Transcription

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `WHISPER_DEVICE` | `cuda` | Inference device: `cuda`, `cpu`, or `auto` |
| `WHISPER_COMPUTE_TYPE` | `float16` | CTranslate2 type: `float16`, `int8_float16`, `int8` |

Recommended for RTX 3080 (10 GB VRAM): `WHISPER_MODEL_SIZE=large-v2`, `WHISPER_COMPUTE_TYPE=float16`.
For CPU-only: `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`.

### Redis Streams Worker

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | _(empty)_ | Redis password (optional) |
| `REDIS_DB` | `0` | Redis DB index |
| `REDIS_STREAM` | `audio-tasks` | Input stream name |
| `REDIS_DLQ_STREAM` | `audio-tasks:dlq` | Dead-letter queue stream name |
| `REDIS_CONSUMER_GROUP` | `audio-workers` | Consumer group name |
| `REDIS_CONSUMER_ID` | auto | Override consumer identity (`hostname-pid` by default) |
| `AUDIO_WORKER_MAX_RETRIES` | `3` | Deliveries before a message is sent to DLQ |
| `REDIS_BLOCK_MS` | `5000` | XREADGROUP blocking timeout (ms) |
| `AUTOCLAIM_IDLE_MS` | `60000` | Reclaim messages idle longer than this (ms) |
| `AUTOCLAIM_INTERVAL_S` | `30` | How often to run XAUTOCLAIM (seconds) |
| `RECONNECT_MAX_DELAY_S` | `60` | Max back-off between Redis reconnect attempts |

### Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Running the Services

### 1. FastAPI Transcription Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

### 2. Redis Streams Audio Worker

Make sure Redis is running first:

```bash
redis-server
```

Then start the worker:

```bash
python -m workers.audio_worker
```

The worker registers itself in the `audio-workers` consumer group on the `audio-tasks` stream and begins processing messages. Graceful shutdown is handled via `SIGINT` / `SIGTERM`.

### 3. Run the Multi-Agent Pipeline

The LangGraph pipeline is a library — invoke it from Python:

```python
from agents.graph import run_graph

result = run_graph("Build a REST endpoint that returns a health check JSON")

print(result["spec"])       # PM-produced specification
print(result["code"])       # Dev-generated Python code
print(result["qa_report"])  # QA review report
print(result["qa_passed"])  # True if QA approved
print(result["status"])     # "running" (success) or "error"
```

---

## API Reference

### `POST /transcribe`

Transcribe an audio file using faster-whisper.

**Request** — `multipart/form-data`:

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | Audio file (WAV, MP3, OGG, FLAC, etc.) |
| `language` | string | no | ISO 639-1 code (e.g. `fr`, `en`). Omit for auto-detection. |
| `stream` | bool | no | `true` for SSE streaming; `false` (default) for full JSON response. |

**Response (non-streaming)**:

```json
{
  "text": "Full transcript as a single string.",
  "language": "en",
  "duration": 12.345,
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "First segment text.",
      "chunk_index": 0
    }
  ],
  "processing_time_s": 1.23
}
```

**Response (streaming, `?stream=true`)** — Server-Sent Events:

```
data: {"start": 0.0, "end": 3.2, "text": "First segment.", "chunk_index": 0}

data: {"start": 3.5, "end": 6.1, "text": "Second segment.", "chunk_index": 0}

event: done
data: {}
```

On error during streaming:

```
event: error
data: {"detail": "error message"}

event: done
data: {}
```

**Example with curl**:

```bash
# Non-streaming
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=en"

# Streaming
curl -X POST "http://localhost:8000/transcribe?stream=true" \
  -F "file=@audio.wav"
```

### Redis Streams Message Format

Publish a message to the `audio-tasks` stream with these fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Opaque identifier for correlation / idempotency |
| `audio_b64` | string | Base64-encoded audio file bytes |
| `language` | string | ISO 639-1 code, or omit / leave empty for auto-detect |

**Example with redis-cli**:

```bash
python3 -c "
import base64, redis
r = redis.Redis()
with open('audio.wav', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode()
r.xadd('audio-tasks', {'task_id': 'job-001', 'audio_b64': audio_b64, 'language': 'en'})
"
```

---

## Running Tests

All tests use mocked LLM calls — no API key is needed.

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_graph.py -v

# Run a specific test class
pytest tests/test_graph.py::TestCircuitBreaker -v

# Run with coverage
pip install pytest-cov
pytest --cov=agents --cov=services --cov=workers --cov-report=term-missing
```

### Test Files

| File | What it covers |
|---|---|
| `tests/test_graph.py` | Routers, circuit breaker, every agent node, full graph integration |
| `tests/test_transcription.py` | `load_audio`, `chunk_audio`, `TranscriptionService` |
| `tests/test_audio_worker.py` | `AudioWorker` happy path, DLQ routing, reconnect logic |
| `tests/test_circuit_breaker_pathological.py` | 20 edge-case circuit breaker scenarios (Gate G4) |

---

## Debugging Guide

### Agent Pipeline Issues

**Problem: `circuit_open` status immediately**

Check the iteration counters in the state. If a counter starts at `MAX_ITERATIONS` (3), the circuit fires before the node body executes.

```python
# Inspect iteration counters
from agents.graph import _initial_state
state = _initial_state("my task")
print(state["pm_iterations"])   # should be 0
```

**Problem: LLM call fails with auth error**

Verify your API key is set:

```bash
echo $OPENAI_API_KEY      # or ANTHROPIC_API_KEY
```

Check which model each agent uses via the `PM_MODEL`, `DEV_MODEL`, `QA_MODEL`, `ERROR_MODEL` env vars.

**Problem: PM_Agent returns invalid JSON**

The PM agent uses `response_format={"type": "json_object"}`. Some models ignore this. Switch to a model that supports JSON mode or add retries. Check the raw LLM output in the logs at `DEBUG` level.

**Problem: QA keeps failing and routing back to Dev**

The pipeline retries Dev up to `MAX_ITERATIONS` (3) times. After that it routes to Error_Handler. Raise `LOG_LEVEL=DEBUG` to see the QA report with specific issues:

```bash
LOG_LEVEL=DEBUG python -c "from agents.graph import run_graph; run_graph('your task')"
```

### Audio Transcription Issues

**Problem: `faster-whisper` fails to load the model**

On first run, the model is downloaded from Hugging Face. Ensure internet access. On GPU:

```bash
# Verify CUDA is detected
python -c "import torch; print(torch.cuda.is_available())"
```

For CPU-only environments:

```bash
WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=int8 uvicorn api.main:app --port 8000
```

**Problem: `soundfile` cannot read the audio file**

`soundfile` reads WAV, FLAC, and OGG natively. For MP3, FFmpeg must be installed:

```bash
sudo apt-get install ffmpeg
pip install pydub   # optional — for format conversion pre-processing
```

**Problem: Timestamps are wrong / segments overlap**

The chunking uses 15 s windows with 0.5 s overlap. Timestamps are re-based by adding `time_offset` (the chunk start in seconds). If you see doubled segments, increase `OVERLAP_DURATION` in `services/transcription.py` or check for duplicate segment filtering.

### Redis Worker Issues

**Problem: Worker cannot connect to Redis**

```bash
redis-cli ping   # should return PONG
```

Check `REDIS_HOST`, `REDIS_PORT`, and `REDIS_PASSWORD`. The worker retries with exponential back-off (1s → 2s → 4s … capped at `RECONNECT_MAX_DELAY_S`).

**Problem: Messages stuck in the Pending Entry List (PEL)**

Messages that fail processing stay un-ACKed in the PEL. The `XAUTOCLAIM` loop (every `AUTOCLAIM_INTERVAL_S` seconds) re-delivers them. After `AUDIO_WORKER_MAX_RETRIES` deliveries the message is moved to the DLQ stream.

Inspect the PEL manually:

```bash
redis-cli XPENDING audio-tasks audio-workers - + 10
```

Inspect the DLQ:

```bash
redis-cli XRANGE audio-tasks:dlq - +
```

**Problem: Consumer group already exists error**

This is expected and harmless — the `BUSYGROUP` error is silently ignored. The worker is designed to be started multiple times safely.

### Structured Log Inspection

All workers emit structured JSON logs. Filter them with `jq`:

```bash
python -m workers.audio_worker 2>&1 | jq 'select(.level == "ERROR")'
python -m workers.audio_worker 2>&1 | jq 'select(.event == "dlq")'
```

---

## Key Design Constraints

These constraints come from `claude.md` and are enforced in code:

| ID | Constraint | Implementation |
|---|---|---|
| R01 | Circuit breaker mandatory — `max_iterations = 3` per node | `@circuit_breaker` decorator in `agents/circuit_breaker.py` |
| R06 | All code must pass QA before reaching any Sandbox | QA gate in `agents/graph.py` — `qa_passed` must be `True` to reach `END` |
| G7 | No GPU port exposed publicly — all via WireGuard VPN | Infrastructure-level constraint (not in application code) |

---

## Phase 1 Exit Gates

| Gate | Criterion | Status |
|---|---|---|
| G1 | Voice → spec in under 60 s | Target |
| G4 | 0 infinite loops across 20 pathological tests | Verified — see `tests/CIRCUIT_BREAKER_PATHOLOGICAL_REPORT.md` |
| G6 | LLaMA 70B P95 latency < 2 s | Requires GPU infra deployment |
