"""Generate synthetic audio fixtures for transcription unit tests.

Run once (or whenever fixtures need regenerating):
    python tests/audio_samples/generate_fixtures.py

Produces:
    silence_1s.wav              — 1 s of 16 kHz silence (zeros)
    silence_5s.wav              — 5 s of 16 kHz silence
    tone_3s.wav                 — 3 s 440 Hz sine at moderate amplitude
    quiet_tone_3s.wav           — same tone at 1/30th amplitude (whisper-quiet)
    long_silence_then_tone.wav  — 3 s silence + 2 s 440 Hz tone
    webm_tone.webm              — WebM/Opus-encoded 3 s tone (requires PyAV)
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
SR = 16_000


# ── WAV writer ────────────────────────────────────────────────────────────────

def _write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> None:
    """Write a float32 numpy array as a 16-bit PCM mono WAV file."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ── Fixtures ──────────────────────────────────────────────────────────────────

def gen_silence_1s() -> None:
    _write_wav(OUT / "silence_1s.wav", np.zeros(SR, dtype=np.float32))


def gen_silence_5s() -> None:
    _write_wav(OUT / "silence_5s.wav", np.zeros(5 * SR, dtype=np.float32))


def gen_tone_3s(amplitude: float = 0.3, filename: str = "tone_3s.wav") -> None:
    t = np.linspace(0, 3.0, 3 * SR, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)
    _write_wav(OUT / filename, tone)


def gen_quiet_tone_3s() -> None:
    gen_tone_3s(amplitude=0.01, filename="quiet_tone_3s.wav")


def gen_long_silence_then_tone() -> None:
    silence = np.zeros(3 * SR, dtype=np.float32)
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    combined = np.concatenate([silence, tone]).astype(np.float32)
    _write_wav(OUT / "long_silence_then_tone.wav", combined)


def gen_webm_tone() -> None:
    """Encode the 3 s tone as WebM/Opus using PyAV (optional)."""
    try:
        import av  # type: ignore[import]
    except ImportError:
        print("PyAV not installed — skipping webm_tone.webm")
        return

    t = np.linspace(0, 3.0, 3 * SR, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    out_path = OUT / "webm_tone.webm"
    container = av.open(str(out_path), mode="w", format="webm")
    stream = container.add_stream("libopus", rate=SR)
    stream.layout = "mono"

    # PyAV expects fltp format for libopus
    frame = av.AudioFrame(format="fltp", layout="mono", samples=len(tone))
    frame.sample_rate = SR
    frame.pts = 0
    frame.planes[0].update(tone.tobytes())

    container.mux(stream.encode(frame))
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    print(f"Written {out_path}")


if __name__ == "__main__":
    gen_silence_1s()
    gen_silence_5s()
    gen_tone_3s()
    gen_quiet_tone_3s()
    gen_long_silence_then_tone()
    gen_webm_tone()
    print("Audio fixtures generated in", OUT)
