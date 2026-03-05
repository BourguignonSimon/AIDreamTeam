"""Audio transcription service using faster-whisper (CTranslate2).

Performance target: < 3s for 30s of audio on GPU RTX 3080.

Strategy
--------
1. Load audio → numpy float32 mono array at 16 kHz (Whisper native rate).
2. Split into 15-second chunks with 0.5s overlap to avoid word boundary cuts.
3. Transcribe each chunk sequentially; faster-whisper streams VAD-filtered
   segments internally.
4. Re-base timestamps relative to the original audio timeline.
"""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import Generator, Iterator, List, Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults (overridable via environment variables)
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000          # Hz — Whisper native sample rate
CHUNK_DURATION: float = 15.0       # seconds per chunk
OVERLAP_DURATION: float = 0.5      # seconds of overlap between chunks

DEFAULT_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
DEFAULT_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
DEFAULT_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionSegment:
    start: float        # seconds from start of original audio
    end: float          # seconds from start of original audio
    text: str
    chunk_index: int = 0


@dataclass
class TranscriptionResult:
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(source: bytes | str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio from raw bytes or a file path.

    Returns a 1-D float32 numpy array resampled to *target_sr* Hz.
    Multi-channel audio is mixed down to mono by averaging channels.
    """
    if isinstance(source, bytes):
        buf = io.BytesIO(source)
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)
    else:
        audio, sr = sf.read(source, dtype="float32", always_2d=True)

    # Mix down to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample when necessary
    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, num_samples),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio.astype(np.float32)


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
    overlap_duration: float = OVERLAP_DURATION,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Split *audio* into overlapping windows.

    Yields
    ------
    (chunk_index, time_offset_seconds, audio_chunk)
        *time_offset_seconds* is the start position of the chunk in the
        original audio, used to re-base per-segment timestamps.
    """
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    step_samples = chunk_samples - overlap_samples
    total_samples = len(audio)

    chunk_index = 0
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        time_offset = start / sample_rate
        yield chunk_index, time_offset, audio[start:end]
        if end >= total_samples:
            break
        start += step_samples
        chunk_index += 1


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TranscriptionService:
    """Thin wrapper around :class:`faster_whisper.WhisperModel`.

    The model is loaded once at construction time and reused across requests.
    Use a DI framework or ``functools.lru_cache`` to keep a single instance
    per process.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            model_size, device, compute_type,
        )
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("faster-whisper model loaded successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_stream(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> Iterator[TranscriptionSegment]:
        """Transcribe *audio* chunk-by-chunk, yielding segments as they arrive.

        Each 15-second chunk is fed independently to the model; segment
        timestamps are shifted by the chunk's time offset so that they are
        relative to the original audio start.

        Args:
            audio: 1-D float32 array at :data:`SAMPLE_RATE` Hz.
            language: ISO 639-1 code (e.g. ``"fr"``).  ``None`` = auto-detect.
            beam_size: Beam search width.  Higher → more accurate but slower.

        Yields:
            :class:`TranscriptionSegment` objects in chronological order.
        """
        total_duration = len(audio) / SAMPLE_RATE
        logger.info(
            "Transcribing %.2fs of audio in %.0fs chunks", total_duration, CHUNK_DURATION
        )

        for chunk_idx, time_offset, chunk in chunk_audio(audio):
            segments_gen, info = self._model.transcribe(
                chunk,
                beam_size=beam_size,
                language=language,
                vad_filter=True,
            )
            if chunk_idx == 0:
                logger.info("Detected language: %s (prob=%.2f)", info.language, info.language_probability)

            for seg in segments_gen:
                yield TranscriptionSegment(
                    start=round(time_offset + seg.start, 3),
                    end=round(time_offset + seg.end, 3),
                    text=seg.text,
                    chunk_index=chunk_idx,
                )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> TranscriptionResult:
        """Load raw audio bytes and return a complete :class:`TranscriptionResult`.

        This is the synchronous, non-streaming path suitable for short clips
        or batch workloads where the full transcript is needed before returning.
        """
        audio = load_audio(audio_bytes)
        duration = round(len(audio) / SAMPLE_RATE, 3)

        all_segments: list[TranscriptionSegment] = []
        detected_language = language or ""

        for chunk_idx, time_offset, chunk in chunk_audio(audio):
            segments_gen, info = self._model.transcribe(
                chunk,
                beam_size=beam_size,
                language=language,
                vad_filter=True,
            )
            if chunk_idx == 0 and not detected_language:
                detected_language = info.language

            for seg in segments_gen:
                all_segments.append(
                    TranscriptionSegment(
                        start=round(time_offset + seg.start, 3),
                        end=round(time_offset + seg.end, 3),
                        text=seg.text,
                        chunk_index=chunk_idx,
                    )
                )

        return TranscriptionResult(
            segments=all_segments,
            language=detected_language,
            duration=duration,
        )
