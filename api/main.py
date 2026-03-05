"""FastAPI application — Audio Transcription Service.

Endpoints
---------
POST /transcribe
    Upload an audio file; receive a JSON transcript with per-segment
    timestamps, detected language, audio duration, and processing time.

    Optional query parameter ``?stream=true`` switches to a Server-Sent Events
    stream where each event is one JSON segment delivered as it is recognised.

Structured JSON logging and FastAPI dependency injection are used throughout
so that the underlying :class:`~services.transcription.TranscriptionService`
can be replaced in tests via ``app.dependency_overrides``.
"""
from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from typing import Iterator, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.sandbox_ws import router as sandbox_router
from services.transcription import (
    TranscriptionResult,
    TranscriptionService,
    load_audio,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Dream Team — API",
    description="Audio transcription + React sandbox code streaming.",
    version="1.1.0",
)

# CORS — allow the Next.js dev server and production frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount WebSocket sandbox routes
app.include_router(sandbox_router)


# ---------------------------------------------------------------------------
# Dependency — single model instance per process
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _cached_service() -> TranscriptionService:
    return TranscriptionService()


def get_transcription_service() -> TranscriptionService:
    """FastAPI dependency that returns the singleton :class:`TranscriptionService`."""
    return _cached_service()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    chunk_index: int


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: list[SegmentOut]
    processing_time_s: float


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe an audio file",
    response_description="Full transcript with per-segment timestamps",
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file — WAV, MP3, OGG, FLAC, etc."),
    language: Optional[str] = Query(
        None,
        description="ISO 639-1 language code (e.g. 'fr', 'en'). Omit for auto-detection.",
    ),
    stream: bool = Query(
        False,
        description="Return a Server-Sent Events stream instead of waiting for the full result.",
    ),
    service: TranscriptionService = Depends(get_transcription_service),
) -> TranscribeResponse | StreamingResponse:
    """Transcribe the uploaded audio file.

    - **Non-streaming** (default): waits for the full transcript then returns
      a single JSON response including ``text``, ``language``, ``duration``,
      ``segments``, and ``processing_time_s``.

    - **Streaming** (``?stream=true``): returns a ``text/event-stream`` SSE
      response.  Each ``data:`` event contains one JSON segment as soon as it
      is recognised.  A final ``event: done`` signals completion.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if stream:
        return _build_sse_response(service, audio_bytes, language)

    # ---- synchronous full-result path ----
    t0 = time.perf_counter()
    try:
        result: TranscriptionResult = service.transcribe_bytes(audio_bytes, language=language)
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}") from exc

    elapsed = round(time.perf_counter() - t0, 3)
    ratio = result.duration / elapsed if elapsed > 0 else 0.0
    logger.info(
        '{"event":"transcription_complete","audio_duration":%.3f,"processing_time":%.3f,"rtf":%.2f}',
        result.duration, elapsed, ratio,
    )

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        duration=result.duration,
        segments=[
            SegmentOut(
                start=s.start,
                end=s.end,
                text=s.text,
                chunk_index=s.chunk_index,
            )
            for s in result.segments
        ],
        processing_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

def _build_sse_response(
    service: TranscriptionService,
    audio_bytes: bytes,
    language: Optional[str],
) -> StreamingResponse:
    """Wrap the streaming transcription generator in an SSE response."""

    def _event_generator() -> Iterator[str]:
        try:
            audio = load_audio(audio_bytes)
            for segment in service.transcribe_stream(audio, language=language):
                payload = json.dumps(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "chunk_index": segment.chunk_index,
                    }
                )
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.exception("Streaming transcription failed: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
        finally:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")
