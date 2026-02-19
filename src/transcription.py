"""Offline Speech-to-Text via MLX Whisper.

Architecture
~~~~~~~~~~~~
- A single ``WhisperTranscriber`` instance is lazily loaded on first use
  and kept resident until ``unload()`` is explicitly called (or the process
  exits).
- The model runs locally on Apple Silicon via ``mlx-whisper``.
- ``transcribe_audio_bytes()`` is the main entry point: it accepts raw PCM
  or WAV bytes from the frontend, resamples/converts as needed, and returns
  a transcript string.

WASM / browser approach deliberately avoided
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This backend approach keeps all inference local.
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

TARGET_SR = 16_000  # Whisper expects 16 kHz mono float audio
MAX_CHUNK_SECONDS = 10  # clamp long recordings so we never block for ages
WHISPER_MODEL_ID = "mlx-community/whisper-large-v3-4bit"
MIN_AUDIO_SECONDS = 0.30
SPEECH_RMS_THRESHOLD = 0.008
SPEECH_PEAK_THRESHOLD = 0.03
VAD_FRAME_MS = 30
VAD_MIN_VOICED_FRAMES = 3
VAD_MIN_VOICED_RATIO = 0.08


# ── Lazy singleton ────────────────────────────────────────────────────────────


class WhisperTranscriber:
    """Thin wrapper around the mlx-whisper package.

    Thread-safe: a module-level lock guards model loading so multiple
    concurrent requests won't each try to load the model.
    """

    def __init__(self) -> None:
        self._ready = False
        self._load_error: Optional[RuntimeError] = None
        self._model_path: Optional[str] = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _resolve_model_path(self) -> str:
        """Resolve HF repo to local path and normalize expected weight filename.

        mlx-whisper expects `weights.safetensors` or `weights.npz`, but some
        repos (including whisper-large-v3-4bit) ship `model.safetensors`.
        """
        if self._model_path is not None:
            return self._model_path

        from huggingface_hub import snapshot_download  # type: ignore[import]

        local = Path(snapshot_download(repo_id=WHISPER_MODEL_ID))
        model_sf = local / "model.safetensors"
        weights_sf = local / "weights.safetensors"

        if model_sf.exists() and not weights_sf.exists():
            try:
                weights_sf.symlink_to(model_sf.name)
            except Exception:
                # Filesystems without symlink support: copy once.
                import shutil

                shutil.copyfile(model_sf, weights_sf)

        self._model_path = str(local)
        return self._model_path

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._load_error is not None:
            raise self._load_error
        if self._ready:
            return
        with self._lock:
            if self._load_error is not None:
                raise self._load_error
            if self._ready:
                return
            logger.info("Loading Whisper model '%s' (this happens once)…", WHISPER_MODEL_ID)
            try:
                import mlx_whisper  # type: ignore[import]
                model_path = self._resolve_model_path()
                # Warm model cache/load once using a tiny silent sample.
                warmup = np.zeros((TARGET_SR,), dtype=np.float32)
                mlx_whisper.transcribe(
                    warmup,
                    path_or_hf_repo=model_path,
                    language="en",
                    condition_on_previous_text=False,
                    temperature=0.0,
                    no_speech_threshold=0.9,
                    fp16=False,
                )
                self._ready = True
                logger.info("Whisper model '%s' loaded successfully.", WHISPER_MODEL_ID)
            except Exception as exc:
                logger.exception("Failed to load Whisper model '%s': %s", WHISPER_MODEL_ID, exc)
                message = str(exc)
                if "load_npz" in message and "zip file" in message:
                    self._load_error = RuntimeError(
                        f"Whisper model '{WHISPER_MODEL_ID}' is incompatible with mlx-whisper "
                        "in this environment (weights format mismatch: expected MLX npz). "
                        "Use an MLX-converted Whisper repo such as 'mlx-community/distil-whisper-large-v3'."
                    )
                else:
                    self._load_error = RuntimeError(
                        f"Whisper model '{WHISPER_MODEL_ID}' could not be loaded. "
                        "Install dependencies with: pip install mlx-whisper"
                    )
                raise self._load_error from exc

    # ------------------------------------------------------------------
    def transcribe(self, audio_f32: np.ndarray) -> str:
        """Transcribe a 16 kHz mono float32 numpy array → text string."""
        self._ensure_loaded()

        # Clamp to MAX_CHUNK_SECONDS to keep latency bounded
        max_samples = TARGET_SR * MAX_CHUNK_SECONDS
        if len(audio_f32) > max_samples:
            audio_f32 = audio_f32[:max_samples]

        # Remove leading/trailing silence before VAD/ASR to reduce
        # end-of-utterance hallucinations on tail noise.
        audio_f32 = _trim_silence_edges(audio_f32)

        # Root-cause guard: do not run ASR on near-silent/noisy tail chunks.
        # This prevents hallucinated polite endings without brittle text hacks.
        if not _has_speech_content(audio_f32):
            return ""

        import mlx_whisper  # type: ignore[import]

        with self._infer_lock:
            result = mlx_whisper.transcribe(
                audio_f32,
                path_or_hf_repo=self._resolve_model_path(),
                language="en",
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.92,
                compression_ratio_threshold=2.4,
                logprob_threshold=-0.8,
                initial_prompt="Transcribe only spoken words from a short query. Ignore silence and do not add sign-offs.",
                fp16=False,
            )
        if isinstance(result, dict):
            text = result.get("text", "")
            return text.strip() if isinstance(text, str) else ""
        return ""

    # ------------------------------------------------------------------
    def unload(self) -> None:
        with self._lock:
            if self._ready:
                logger.info("Unloading Whisper transcriber runtime.")
                self._ready = False


# Module-level singleton — shared across requests
_transcriber: Optional[WhisperTranscriber] = None
_singleton_lock = threading.Lock()


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        with _singleton_lock:
            if _transcriber is None:
                _transcriber = WhisperTranscriber()
    return _transcriber


# ── Public helpers ────────────────────────────────────────────────────────────


def transcribe_audio_bytes(raw: bytes, sample_rate: int = TARGET_SR) -> str:
    """Accept raw audio bytes (WAV or raw 16-bit PCM) and return transcript.

    The frontend records via ``MediaRecorder`` which produces WebM/Opus or
    WAV depending on browser.  We use ``soundfile`` to decode any container
    that libsndfile supports; for browsers that emit raw PCM ArrayBuffers
    we handle the int16 → float32 conversion manually.
    """
    audio_f32 = _decode_audio(raw, sample_rate)
    if audio_f32 is None or len(audio_f32) == 0:
        return ""
    return get_transcriber().transcribe(audio_f32)


def _decode_audio(raw: bytes, hint_sr: int) -> Optional[np.ndarray]:
    """Decode audio bytes to a 16 kHz mono float32 numpy array.

    Tries (in order):
    1. soundfile – handles WAV, FLAC, OGG, WebM (with libsndfile ≥ 1.1)
    2. PyAV – robust decode for browser MediaRecorder chunks (WebM/OGG Opus)
    3. scipy.io.wavfile – pure-Python WAV fallback
    4. Treat as raw 16-bit little-endian PCM
    """
    # ── Attempt 1: soundfile ──────────────────────────────────────────────────
    try:
        import soundfile as sf  # type: ignore[import]
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        arr = _to_mono(arr)
        if sr != TARGET_SR:
            arr = _resample(arr, sr, TARGET_SR)
        return arr.astype(np.float32)
    except Exception:
        pass  # fall through

    # ── Attempt 2: PyAV (WebM/OGG/Opus robust path) ─────────────────────────
    try:
        import av  # type: ignore[import]

        chunks: list[np.ndarray] = []
        with av.open(io.BytesIO(raw), mode="r") as container:
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                raise ValueError("No audio stream found")

            resampler = av.audio.resampler.AudioResampler(
                format="fltp",
                layout="mono",
                rate=TARGET_SR,
            )

            for frame in container.decode(audio_stream):
                for out in resampler.resample(frame):
                    arr = out.to_ndarray()
                    # fltp mono: shape (1, samples)
                    if arr.ndim == 2:
                        arr = arr[0]
                    arr = arr.astype(np.float32)
                    if arr.size:
                        chunks.append(arr)

            for out in resampler.resample(None):
                arr = out.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                arr = arr.astype(np.float32)
                if arr.size:
                    chunks.append(arr)

        if chunks:
            return np.concatenate(chunks).astype(np.float32)
    except Exception:
        pass

    # ── Attempt 3: scipy WAV reader ───────────────────────────────────────────
    try:
        from scipy.io import wavfile  # type: ignore[import]
        sr, arr = wavfile.read(io.BytesIO(raw))
        arr = _to_mono(arr.astype(np.float32))
        if arr.max() > 1.0:  # int16 range
            arr = arr / 32768.0
        if sr != TARGET_SR:
            arr = _resample(arr, sr, TARGET_SR)
        return arr.astype(np.float32)
    except Exception:
        pass

    # ── Attempt 4: raw int16 PCM ──────────────────────────────────────────────
    try:
        if len(raw) < 2:
            return None
        usable = len(raw) - (len(raw) % 2)
        if usable <= 0:
            return None
        arr = np.frombuffer(raw[:usable], dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            return None
        if hint_sr != TARGET_SR:
            arr = _resample(arr, hint_sr, TARGET_SR)
        return arr
    except Exception:
        logger.warning(
            "All audio decoding strategies failed for chunk (%d bytes).",
            len(raw),
        )
        return None


def _to_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.mean(axis=1)
    return arr


def _has_speech_content(audio_f32: np.ndarray) -> bool:
    if audio_f32.size == 0:
        return False
    if (audio_f32.size / TARGET_SR) < MIN_AUDIO_SECONDS:
        return False

    if not _passes_voiced_gate(audio_f32):
        return False

    abs_audio = np.abs(audio_f32)
    rms = float(np.sqrt(np.mean(audio_f32 * audio_f32)))
    peak = float(np.max(abs_audio))

    return rms >= SPEECH_RMS_THRESHOLD and peak >= SPEECH_PEAK_THRESHOLD


def _trim_silence_edges(audio_f32: np.ndarray) -> np.ndarray:
    if audio_f32.size == 0:
        return audio_f32

    abs_audio = np.abs(audio_f32)
    edge_threshold = max(SPEECH_RMS_THRESHOLD * 0.75, 0.004)
    active = np.where(abs_audio >= edge_threshold)[0]
    if active.size == 0:
        return np.array([], dtype=np.float32)

    pad = int(0.12 * TARGET_SR)  # keep a small context margin
    start = max(0, int(active[0]) - pad)
    end = min(audio_f32.size, int(active[-1]) + pad + 1)
    return audio_f32[start:end].astype(np.float32, copy=False)


def _passes_voiced_gate(audio_f32: np.ndarray) -> bool:
    """Stable, pure-NumPy voicedness gate (no native-extension VAD)."""
    frame_len = int(TARGET_SR * (VAD_FRAME_MS / 1000.0))
    if frame_len <= 0 or audio_f32.size < frame_len:
        return False

    voiced = 0
    total = 0

    for start in range(0, audio_f32.size - frame_len + 1, frame_len):
        frame = audio_f32[start:start + frame_len]
        total += 1

        rms = float(np.sqrt(np.mean(frame * frame)))
        peak = float(np.max(np.abs(frame)))

        signs = np.signbit(frame)
        zcr = float(np.mean(signs[1:] != signs[:-1])) if frame.size > 1 else 0.0

        # Speech-ish heuristic: enough energy + plausible zero-crossing range.
        if rms >= 0.007 and peak >= 0.025 and 0.003 <= zcr <= 0.35:
            voiced += 1

    if total == 0:
        return False
    voiced_ratio = voiced / total
    return voiced >= VAD_MIN_VOICED_FRAMES and voiced_ratio >= VAD_MIN_VOICED_RATIO


def _resample(arr: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear-interpolation resample (good enough for speech, no extra deps)."""
    if orig_sr == target_sr:
        return arr
    # Try scipy first (higher quality)
    try:
        from scipy.signal import resample_poly  # type: ignore[import]
        from math import gcd
        g = gcd(orig_sr, target_sr)
        return resample_poly(arr, target_sr // g, orig_sr // g).astype(np.float32)
    except Exception:
        pass
    # Fallback: numpy linear interpolation
    n_out = int(len(arr) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(arr) - 1, n_out),
        np.arange(len(arr)),
        arr,
    ).astype(np.float32)
