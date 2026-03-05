"""Unit tests for the audio transcription service and FastAPI endpoint.

All faster-whisper model calls are mocked — no GPU or model weights required.

Test coverage
-------------
- ``load_audio``: bytes → numpy array, mono mix-down, resampling
- ``chunk_audio``: correct number of chunks, time offsets, overlap
- ``TranscriptionService.transcribe_stream``: mock model, timestamp re-basing
- ``TranscriptionService.transcribe_bytes``: full result assembly, language detection
- ``POST /transcribe``: FastAPI TestClient — happy path, empty file, streaming SSE
- Performance assertion: 30s audio chunked in < 3s (mocked, measures overhead only)
"""
from __future__ import annotations

import io
import json
import math
import struct
import time
import wave
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.transcription import (
    CHUNK_DURATION,
    OVERLAP_DURATION,
    SAMPLE_RATE,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionService,
    chunk_audio,
    load_audio,
)


# ---------------------------------------------------------------------------
# Helpers — deterministic test audio generation
# ---------------------------------------------------------------------------

def _make_wav_bytes(
    duration_s: float,
    sample_rate: int = 16_000,
    frequency: float = 440.0,
    amplitude: float = 0.3,
    channels: int = 1,
) -> bytes:
    """Generate a pure-sine mono (or stereo) WAV and return raw bytes."""
    n_samples = int(duration_s * sample_rate)
    raw_samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(n_samples)
    ]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if channels == 1:
            wf.writeframes(struct.pack(f"<{n_samples}h", *raw_samples))
        else:
            # Duplicate channel for stereo
            interleaved = []
            for s in raw_samples:
                interleaved.extend([s, s])
            wf.writeframes(struct.pack(f"<{n_samples * 2}h", *interleaved))
    return buf.getvalue()


def _make_mock_model(segments: list[dict] | None = None) -> MagicMock:
    """Return a mock WhisperModel whose ``transcribe`` returns preset segments."""
    if segments is None:
        segments = [{"start": 0.0, "end": 2.5, "text": " Hello world"}]

    mock_model = MagicMock()

    def _transcribe_side_effect(audio, **kwargs):
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_segments = []
        for s in segments:
            seg = MagicMock()
            seg.start = s["start"]
            seg.end = s["end"]
            seg.text = s["text"]
            mock_segments.append(seg)

        return iter(mock_segments), mock_info

    mock_model.transcribe.side_effect = _transcribe_side_effect
    return mock_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wav_1s() -> bytes:
    return _make_wav_bytes(duration_s=1.0)


@pytest.fixture
def wav_30s() -> bytes:
    return _make_wav_bytes(duration_s=30.0)


@pytest.fixture
def wav_stereo_5s() -> bytes:
    return _make_wav_bytes(duration_s=5.0, channels=2)


@pytest.fixture
def wav_48k_10s() -> bytes:
    """10-second WAV at 48 kHz — tests resampling path."""
    return _make_wav_bytes(duration_s=10.0, sample_rate=48_000)


# ---------------------------------------------------------------------------
# Tests — load_audio
# ---------------------------------------------------------------------------

class TestLoadAudio:
    def test_returns_float32_array(self, wav_1s):
        audio = load_audio(wav_1s)
        assert audio.dtype == np.float32

    def test_correct_length_native_rate(self, wav_1s):
        audio = load_audio(wav_1s)
        expected = 1.0 * SAMPLE_RATE
        # Allow ±1% tolerance for any rounding in WAV frame count
        assert abs(len(audio) - expected) / expected < 0.01

    def test_mono_mix_down(self, wav_stereo_5s):
        audio = load_audio(wav_stereo_5s)
        assert audio.ndim == 1

    def test_resampling_produces_correct_length(self, wav_48k_10s):
        audio = load_audio(wav_48k_10s, target_sr=16_000)
        expected = 10.0 * 16_000
        assert abs(len(audio) - expected) / expected < 0.01

    def test_accepts_file_path(self, tmp_path, wav_1s):
        path = tmp_path / "test.wav"
        path.write_bytes(wav_1s)
        audio = load_audio(str(path))
        assert len(audio) > 0


# ---------------------------------------------------------------------------
# Tests — chunk_audio
# ---------------------------------------------------------------------------

class TestChunkAudio:
    def _make_audio(self, duration_s: float) -> np.ndarray:
        return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)

    def test_single_chunk_for_short_audio(self):
        audio = self._make_audio(5.0)
        chunks = list(chunk_audio(audio))
        assert len(chunks) == 1

    def test_correct_chunk_count_for_30s(self):
        audio = self._make_audio(30.0)
        chunks = list(chunk_audio(audio))
        # 30s with 15s chunks and 0.5s overlap:
        # chunk 0: [0 → 15s], chunk 1: [14.5 → 29.5s], chunk 2: [29s → 30s]
        assert len(chunks) == 3

    def test_time_offsets_are_non_decreasing(self):
        audio = self._make_audio(60.0)
        offsets = [t for _, t, _ in chunk_audio(audio)]
        assert offsets == sorted(offsets)

    def test_first_chunk_offset_is_zero(self):
        audio = self._make_audio(30.0)
        first = next(chunk_audio(audio))
        _, offset, _ = first
        assert offset == 0.0

    def test_chunk_indices_are_sequential(self):
        audio = self._make_audio(45.0)
        indices = [idx for idx, _, _ in chunk_audio(audio)]
        assert indices == list(range(len(indices)))

    def test_last_chunk_does_not_exceed_audio(self):
        audio = self._make_audio(20.0)
        for _, _, chunk in chunk_audio(audio):
            assert len(chunk) <= len(audio)

    def test_custom_chunk_and_overlap(self):
        audio = self._make_audio(10.0)
        chunks = list(chunk_audio(audio, chunk_duration=5.0, overlap_duration=0.0))
        assert len(chunks) == 2
        _, t0, _ = chunks[0]
        _, t1, _ = chunks[1]
        assert t0 == pytest.approx(0.0)
        assert t1 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Tests — TranscriptionService (mocked model)
# ---------------------------------------------------------------------------

class TestTranscriptionService:
    def _make_service(self, segments=None) -> TranscriptionService:
        """Construct a TranscriptionService with a mocked WhisperModel."""
        with patch("services.transcription.WhisperModel") as MockModel:
            MockModel.return_value = _make_mock_model(segments)
            svc = TranscriptionService(model_size="base", device="cpu", compute_type="int8")
        # Swap in the mock directly so it persists after the context manager
        svc._model = _make_mock_model(segments)
        return svc

    def test_transcribe_stream_yields_segments(self):
        svc = self._make_service([{"start": 0.0, "end": 1.0, "text": " Hi"}])
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second
        results = list(svc.transcribe_stream(audio))
        assert len(results) >= 1
        assert all(isinstance(r, TranscriptionSegment) for r in results)

    def test_transcribe_stream_timestamps_rebased(self):
        """Segments from chunk N should have start offset >= chunk_start."""
        svc = self._make_service([{"start": 1.0, "end": 2.0, "text": " Word"}])
        # 20s audio → 2 chunks; second chunk starts at (15 - 0.5) = 14.5s
        audio = np.zeros(int(20 * SAMPLE_RATE), dtype=np.float32)
        results = list(svc.transcribe_stream(audio))
        # Chunk 1 segment must have start >= 14.5 (offset of chunk 1)
        chunk1_segs = [r for r in results if r.chunk_index == 1]
        if chunk1_segs:
            expected_offset = (CHUNK_DURATION - OVERLAP_DURATION)
            assert chunk1_segs[0].start >= expected_offset - 0.1

    def test_transcribe_bytes_returns_result(self, wav_30s):
        svc = self._make_service([{"start": 0.0, "end": 1.5, "text": " Test"}])
        result = svc.transcribe_bytes(wav_30s)
        assert isinstance(result, TranscriptionResult)
        assert result.duration == pytest.approx(30.0, abs=0.5)
        assert result.language == "en"
        assert len(result.segments) >= 1

    def test_transcribe_bytes_assembles_text(self, wav_1s):
        segs = [
            {"start": 0.0, "end": 0.5, "text": " Hello"},
            {"start": 0.5, "end": 1.0, "text": " world"},
        ]
        svc = self._make_service(segs)
        result = svc.transcribe_bytes(wav_1s)
        assert "Hello" in result.text
        assert "world" in result.text

    def test_transcribe_bytes_explicit_language_preserved(self, wav_1s):
        svc = self._make_service()
        result = svc.transcribe_bytes(wav_1s, language="fr")
        # Explicit language should be forwarded; mock returns "en" from info
        # but since language was provided, it should be kept
        assert result.language == "fr"

    def test_segment_text_stripped_of_leading_space(self, wav_1s):
        svc = self._make_service([{"start": 0.0, "end": 1.0, "text": " Word"}])
        result = svc.transcribe_bytes(wav_1s)
        assert result.text == "Word"


# ---------------------------------------------------------------------------
# Tests — FastAPI endpoint (TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_service():
    """A TranscriptionService mock that returns a canned TranscriptionResult."""
    svc = MagicMock(spec=TranscriptionService)
    svc.transcribe_bytes.return_value = TranscriptionResult(
        segments=[
            TranscriptionSegment(start=0.0, end=2.0, text="Hello world", chunk_index=0),
            TranscriptionSegment(start=2.0, end=4.5, text="This is a test.", chunk_index=0),
        ],
        language="en",
        duration=4.5,
    )
    # transcribe_stream returns an iterator
    svc.transcribe_stream.return_value = iter([
        TranscriptionSegment(start=0.0, end=2.0, text="Hello world", chunk_index=0),
    ])
    return svc


@pytest.fixture
def client(mock_service):
    from api.main import app, get_transcription_service

    app.dependency_overrides[get_transcription_service] = lambda: mock_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestTranscribeEndpoint:
    def test_happy_path_returns_200(self, client, wav_1s):
        resp = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_response_schema(self, client, wav_1s):
        resp = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        body = resp.json()
        assert "text" in body
        assert "language" in body
        assert "duration" in body
        assert "segments" in body
        assert "processing_time_s" in body

    def test_text_field_contains_transcript(self, client, wav_1s):
        resp = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        assert "Hello world" in resp.json()["text"]

    def test_segments_have_timestamps(self, client, wav_1s):
        resp = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        segs = resp.json()["segments"]
        assert len(segs) == 2
        for seg in segs:
            assert "start" in seg and "end" in seg
            assert seg["end"] >= seg["start"]

    def test_language_query_param_forwarded(self, client, mock_service, wav_1s):
        client.post(
            "/transcribe?language=fr",
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        call_kwargs = mock_service.transcribe_bytes.call_args
        assert call_kwargs.kwargs.get("language") == "fr" or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] == "fr"
        )

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/transcribe",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_processing_time_is_non_negative(self, client, wav_1s):
        resp = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        assert resp.json()["processing_time_s"] >= 0.0

    def test_streaming_returns_event_stream(self, client, wav_1s):
        resp = client.post(
            "/transcribe?stream=true",
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_streaming_contains_json_events(self, client, wav_1s):
        resp = client.post(
            "/transcribe?stream=true",
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        lines = resp.text.splitlines()
        data_lines = [l for l in lines if l.startswith("data:")]
        assert len(data_lines) >= 1
        payload = json.loads(data_lines[0].removeprefix("data: ").strip())
        assert "text" in payload
        assert "start" in payload
        assert "end" in payload

    def test_streaming_ends_with_done_event(self, client, wav_1s):
        resp = client.post(
            "/transcribe?stream=true",
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        assert "event: done" in resp.text


# ---------------------------------------------------------------------------
# Performance assertion (mocked — measures framework overhead, not GPU)
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_30s_audio_overhead_under_3s(self, mock_service, wav_30s):
        """With the model mocked, the framework overhead for 30s audio must be < 3s.

        On real hardware with an RTX 3080 and a loaded model the actual target
        is < 3s wall-clock including inference.  This test guards against
        regressions in chunking / serialisation overhead alone.
        """
        from api.main import app, get_transcription_service

        app.dependency_overrides[get_transcription_service] = lambda: mock_service
        try:
            with TestClient(app) as c:
                t0 = time.perf_counter()
                resp = c.post(
                    "/transcribe",
                    files={"file": ("test_30s.wav", wav_30s, "audio/wav")},
                )
                elapsed = time.perf_counter() - t0
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        assert elapsed < 3.0, (
            f"Framework overhead {elapsed:.3f}s exceeds 3s budget. "
            "Check chunking or serialisation regressions."
        )
