"""Unit and integration tests for src/transcription.py.

Unit tests mock ``mlx_whisper.transcribe`` so NO model download is needed.
Integration tests (marked slow) require the real model to be cached locally.

Run unit only:
    pytest tests/test_transcription.py -v

Run with integration:
    pytest tests/test_transcription.py -v -m slow
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Fixtures directory ────────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "audio_samples"


def _wav_bytes(samples: np.ndarray, sr: int = 16_000) -> bytes:
    """Encode float32 array as WAV bytes (in-memory)."""
    buf = io.BytesIO()
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_i16 = (pcm * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


def _tone(seconds: float = 3.0, amplitude: float = 0.3, sr: int = 16_000) -> np.ndarray:
    """440 Hz sine wave as float32 array."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


def _silence(seconds: float = 1.0, sr: int = 16_000) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ── Decode helpers ────────────────────────────────────────────────────────────

class TestDecodeRawInt16:
    def test_basic(self) -> None:
        from src.transcription import _decode_audio

        pcm_i16 = (np.sin(np.linspace(0, 6 * np.pi, 16_000)) * 10000).astype(np.int16)
        result = _decode_audio(pcm_i16.tobytes(), hint_sr=16_000)
        assert result is not None
        assert result.dtype == np.float32
        assert len(result) == 16_000

    def test_empty_returns_none(self) -> None:
        from src.transcription import _decode_audio

        assert _decode_audio(b"", hint_sr=16_000) is None

    def test_one_byte_returns_none(self) -> None:
        from src.transcription import _decode_audio

        assert _decode_audio(b"\x00", hint_sr=16_000) is None


class TestDecodeWAV:
    def test_wav_roundtrip(self) -> None:
        from src.transcription import _decode_audio

        tone = _tone(3.0)
        raw = _wav_bytes(tone)
        result = _decode_audio(raw, hint_sr=16_000)
        assert result is not None
        assert result.dtype == np.float32
        assert abs(len(result) - len(tone)) <= 10  # small rounding OK

    def test_from_fixture(self) -> None:
        from src.transcription import _decode_audio

        raw = (FIXTURES / "tone_3s.wav").read_bytes()
        result = _decode_audio(raw, hint_sr=16_000)
        assert result is not None
        assert len(result) > 40_000  # ~3 s at 16 kHz


class TestNanToNumHygiene:
    def test_nan_to_zero(self) -> None:
        from src.transcription import _decode_audio

        # Build a raw int16 blob that decodes cleanly, then inject NaN via mock
        pcm = np.ones(16_000, dtype=np.int16) * 1000
        raw = pcm.tobytes()
        with patch("numpy.frombuffer") as mock_fb:
            arr_with_nan = np.array([0.1, float("nan"), -float("inf"), float("inf")],
                                    dtype=np.float32)
            mock_fb.return_value = arr_with_nan.view(np.int16)
            result = _decode_audio(raw, hint_sr=16_000)
        # Even if frombuffer mock fails the decode, the returned array must have no NaN
        if result is not None:
            assert not np.any(np.isnan(result)), "NaN leaked into decoded audio"
            assert not np.any(np.isinf(result)), "Inf leaked into decoded audio"


class TestResample:
    def test_48k_to_16k(self) -> None:
        from src.transcription import _resample

        arr_48k = np.random.randn(48_000).astype(np.float32)
        arr_16k = _resample(arr_48k, orig_sr=48_000, target_sr=16_000)
        assert arr_16k.dtype == np.float32
        assert abs(len(arr_16k) - 16_000) <= 5

    def test_same_rate_passthrough(self) -> None:
        from src.transcription import _resample

        arr = np.ones(16_000, dtype=np.float32)
        result = _resample(arr, orig_sr=16_000, target_sr=16_000)
        assert result is arr  # exact same object returned


class TestWebmOpusDecode:
    def test_webm_fixture_decoded(self) -> None:
        webm_path = FIXTURES / "webm_tone.webm"
        if not webm_path.exists():
            pytest.skip("webm_tone.webm fixture not available")

        from src.transcription import _decode_audio

        raw = webm_path.read_bytes()
        # Verify EBML magic bytes
        assert raw[:4] == b"\x1a\x45\xdf\xa3", "fixture not a valid WebM file"
        result = _decode_audio(raw, hint_sr=16_000)
        assert result is not None
        assert len(result) > 20_000  # at least ~1.5 s

    def test_webm_routes_to_pyav_first(self) -> None:
        """EBML magic bytes should cause _decode_audio to try PyAV before soundfile."""
        from src.transcription import _decode_audio, _EBML_MAGIC

        fake_webm = _EBML_MAGIC + b"\x00" * 100

        with patch("src.transcription._decode_pyav") as mock_pyav, \
             patch("src.transcription._decode_soundfile") as mock_sf:
            mock_pyav.return_value = np.zeros(16_000, dtype=np.float32)
            mock_sf.return_value = None
            result = _decode_audio(fake_webm, hint_sr=16_000)
            mock_pyav.assert_called_once()
            # soundfile should NOT be called when PyAV succeeds
            mock_sf.assert_not_called()


# ── VAD / speech gate ─────────────────────────────────────────────────────────

class TestVADGate:
    def test_silence_rejected(self) -> None:
        from src.transcription import _has_speech_content

        assert _has_speech_content(_silence(2.0)) is False

    def test_short_audio_rejected(self) -> None:
        from src.transcription import _has_speech_content

        tiny = _tone(0.1)  # below MIN_AUDIO_SECONDS
        assert _has_speech_content(tiny) is False

    def test_tone_accepted(self) -> None:
        from src.transcription import _has_speech_content

        assert _has_speech_content(_tone(3.0)) is True

    def test_quiet_tone_rejected(self) -> None:
        from src.transcription import _has_speech_content

        quiet = _tone(3.0, amplitude=0.0005)
        # This should be well below RMS + peak thresholds
        assert _has_speech_content(quiet) is False


# ── Hallucination filter ──────────────────────────────────────────────────────

class TestFilterHallucinations:
    def test_blacklist_exact(self) -> None:
        from src.transcription import _filter_hallucinations

        assert _filter_hallucinations("Thank you.") == ""

    def test_blacklist_case_insensitive(self) -> None:
        from src.transcription import _filter_hallucinations

        assert _filter_hallucinations("THANK YOU.") == ""

    def test_blacklist_substring(self) -> None:
        from src.transcription import _filter_hallucinations

        assert _filter_hallucinations("This is great, thank you.") == ""

    def test_repetition_detection(self) -> None:
        from src.transcription import _filter_hallucinations

        # 3-gram "hello there how" repeated 3x
        text = "hello there how hello there how hello there how world"
        assert _filter_hallucinations(text) == ""

    def test_normal_text_passes(self) -> None:
        from src.transcription import _filter_hallucinations

        text = "What documents discuss the sugar trade in the Caribbean?"
        assert _filter_hallucinations(text) == text

    def test_empty_passthrough(self) -> None:
        from src.transcription import _filter_hallucinations

        assert _filter_hallucinations("") == ""


# ── Segment assembly ──────────────────────────────────────────────────────────

class TestAssembleText:
    def test_no_segments_fallback(self) -> None:
        from src.transcription import _assemble_text

        result = {"text": " Hello world "}
        assert _assemble_text(result) == "Hello world"

    def test_bad_segment_dropped(self) -> None:
        from src.transcription import _assemble_text

        result = {
            "text": "hello junk",
            "segments": [
                {"text": " hello", "no_speech_prob": 0.1, "avg_logprob": -0.3,
                 "compression_ratio": 1.2},
                {"text": " junk",  "no_speech_prob": 0.9, "avg_logprob": -0.3,
                 "compression_ratio": 1.2},  # dropped: high no_speech_prob
            ],
        }
        assert _assemble_text(result) == "hello"

    def test_logprob_filter(self) -> None:
        from src.transcription import _assemble_text

        result = {
            "text": "keep drop",
            "segments": [
                {"text": " keep", "no_speech_prob": 0.1, "avg_logprob": -0.5,
                 "compression_ratio": 1.0},
                {"text": " drop", "no_speech_prob": 0.1, "avg_logprob": -2.0,
                 "compression_ratio": 1.0},  # dropped: avg_logprob below threshold
            ],
        }
        assert _assemble_text(result) == "keep"

    def test_compression_ratio_filter(self) -> None:
        from src.transcription import _assemble_text

        result = {
            "segments": [
                {"text": " good", "no_speech_prob": 0.1, "avg_logprob": -0.5,
                 "compression_ratio": 1.5},
                {"text": " bad",  "no_speech_prob": 0.1, "avg_logprob": -0.5,
                 "compression_ratio": 3.5},  # dropped: high compression
            ],
        }
        assert _assemble_text(result) == "good"

    def test_empty_segments_fallback(self) -> None:
        from src.transcription import _assemble_text

        result = {"text": " fallback text ", "segments": []}
        assert _assemble_text(result) == "fallback text"


# ── Peak normalization ────────────────────────────────────────────────────────

class TestPeakNormalization:
    def test_disabled_by_default(self) -> None:
        from src.transcription import _maybe_normalize, ENABLE_PEAK_NORMALIZATION

        assert ENABLE_PEAK_NORMALIZATION is False
        arr = np.ones(100, dtype=np.float32) * 0.01
        result = _maybe_normalize(arr)
        assert result is arr  # exact same object, no copy

    def test_enabled_rescales_quiet_audio(self) -> None:
        import src.transcription as tc

        orig = tc.ENABLE_PEAK_NORMALIZATION
        try:
            tc.ENABLE_PEAK_NORMALIZATION = True
            arr = np.ones(100, dtype=np.float32) * 0.05
            result = tc._maybe_normalize(arr)
            assert float(np.max(np.abs(result))) == pytest.approx(tc.PEAK_NORM_TARGET, abs=1e-5)
        finally:
            tc.ENABLE_PEAK_NORMALIZATION = orig

    def test_loud_audio_not_normalized(self) -> None:
        import src.transcription as tc

        orig = tc.ENABLE_PEAK_NORMALIZATION
        try:
            tc.ENABLE_PEAK_NORMALIZATION = True
            # peak already above PEAK_NORM_TARGET — should be untouched
            arr = np.ones(100, dtype=np.float32) * tc.PEAK_NORM_TARGET
            result = tc._maybe_normalize(arr)
            np.testing.assert_array_equal(result, arr)
        finally:
            tc.ENABLE_PEAK_NORMALIZATION = orig


# ── Timing / observability ────────────────────────────────────────────────────

class TestTimingObservability:
    def test_timing_logged_on_speech(self, caplog: pytest.LogCaptureFixture) -> None:
        """transcription times: ... must appear in the log when speech is processed."""
        import logging
        import src.transcription as tc

        mock_result = {"text": "hello world", "segments": []}
        # Patch at the top-level module so the `import mlx_whisper` inside transcribe() gets the mock
        with patch("mlx_whisper.transcribe", return_value=mock_result), \
             patch.object(tc.WhisperTranscriber, "_ensure_loaded"), \
             patch.object(tc.WhisperTranscriber, "_resolve_model_path",
                          return_value=tc.WHISPER_MODEL_ID):
            transcriber = tc.WhisperTranscriber()
            transcriber._ready = True

            with caplog.at_level(logging.INFO, logger="src.transcription"):
                text = transcriber.transcribe(_tone(3.0))

        assert "transcription times:" in caplog.text
        assert text == "hello world"

    def test_silence_drop_counter_increments(self) -> None:
        import src.transcription as tc

        orig = tc._silence_drops
        transcriber = tc.WhisperTranscriber()
        transcriber._ready = True
        with patch.object(tc.WhisperTranscriber, "_ensure_loaded"):
            transcriber.transcribe(_silence(2.0))
        assert tc._silence_drops > orig


# ── Full transcribe_audio_bytes (mocked) ─────────────────────────────────────

class TestTranscribeAudioBytes:
    def test_silence_returns_empty(self) -> None:
        from src.transcription import transcribe_audio_bytes

        raw = _wav_bytes(_silence(2.0))
        result = transcribe_audio_bytes(raw)
        assert result == ""

    def test_decode_timing_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        from src.transcription import transcribe_audio_bytes

        raw = _wav_bytes(_silence(1.0))
        with caplog.at_level(logging.DEBUG, logger="src.transcription"):
            transcribe_audio_bytes(raw)
        assert "decode=" in caplog.text


# ── Integration (requires real model) ────────────────────────────────────────

@pytest.mark.slow
class TestIntegration:
    def test_silence_real_model(self) -> None:
        """Real model must return empty string for 2 s of silence."""
        from src.transcription import transcribe_audio_bytes

        raw = _wav_bytes(_silence(2.0))
        result = transcribe_audio_bytes(raw)
        assert result == ""
