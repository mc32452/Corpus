"use client";

/**
 * useSpeechToText  —  backend-based, fully offline
 *
 * Architecture:
 *   The browser records audio via MediaRecorder and POSTs chunks every 2.5 s
 *   to POST /api/transcribe.  The Python backend transcribes each chunk with
 *   Moonshine Tiny locally — zero cloud/network calls.
 *
 *   An AnalyserNode watches RMS volume to detect silence; after 4 s of quiet
 *   the session auto-stops.  After 30 s with no transcription the
 *   onNoSpeech callback fires.
 */

import { useCallback, useEffect, useRef, useState } from "react";

// ─── Types ───────────────────────────────────────────────────────────────────

export type STTStatus = "idle" | "listening" | "stopping" | "transcribing" | "error";

export interface UseSpeechToTextOptions {
  /** Called with each transcript chunk from the backend */
  onTranscript: (chunk: string, isFinal: boolean) => void;
  /** Called when getUserMedia is denied */
  onPermissionDenied?: () => void;
  /** Called after NO_SPEECH_TIMEOUT_MS with no transcription */
  onNoSpeech?: () => void;
  /** Called on network/MediaRecorder errors */
  onError?: (msg: string) => void;
}

export interface UseSpeechToTextReturn {
  status: STTStatus;
  isListening: boolean;
  toggle: () => void;
  stop: () => void;
}

// ─── Config ───────────────────────────────────────────────────────────────────

const BACKEND_URL = "/api/transcribe";
/** MediaRecorder emits a chunk every CHUNK_MS milliseconds */
const CHUNK_MS = 2000;
/** Auto-stop after this many ms of silence */
const SILENCE_TIMEOUT_MS = 3_200;
/** Warn "no speech" if nothing transcribed within this window */
const NO_SPEECH_TIMEOUT_MS = 30_000;
/** RMS amplitude below which the mic is considered silent */
const SILENCE_THRESHOLD = 0.01;

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useSpeechToText({
  onTranscript,
  onPermissionDenied,
  onNoSpeech,
  onError,
}: UseSpeechToTextOptions): UseSpeechToTextReturn {
  const [status, setStatus] = useState<STTStatus>("idle");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const noSpeechTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const silenceRafRef = useRef<number | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordingMimeTypeRef = useRef("audio/webm");

  // Keep stable refs to the latest callbacks so inner closures never go stale
  const onTranscriptRef = useRef(onTranscript);
  const onPermissionDeniedRef = useRef(onPermissionDenied);
  const onNoSpeechRef = useRef(onNoSpeech);
  const onErrorRef = useRef(onError);
  useEffect(() => { onTranscriptRef.current = onTranscript; }, [onTranscript]);
  useEffect(() => { onPermissionDeniedRef.current = onPermissionDenied; }, [onPermissionDenied]);
  useEffect(() => { onNoSpeechRef.current = onNoSpeech; }, [onNoSpeech]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);

  const isListening = status === "listening";

  // ── Cleanup ────────────────────────────────────────────────────────────────
  const cleanup = useCallback(() => {
    if (silenceRafRef.current !== null) {
      cancelAnimationFrame(silenceRafRef.current);
      silenceRafRef.current = null;
    }
    if (silenceTimerRef.current) { clearTimeout(silenceTimerRef.current); silenceTimerRef.current = null; }
    if (noSpeechTimerRef.current) { clearTimeout(noSpeechTimerRef.current); noSpeechTimerRef.current = null; }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    mediaRecorderRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    analyserRef.current = null;
  }, []);

  // ── Stop ───────────────────────────────────────────────────────────────────
  // Stable ref so start() / silenceDetector can always call the latest stop
  const stopRef = useRef<() => void>(() => {});

  const stop = useCallback(() => {
    // Cancel monitoring timers immediately
    if (silenceRafRef.current !== null) { cancelAnimationFrame(silenceRafRef.current); silenceRafRef.current = null; }
    if (silenceTimerRef.current) { clearTimeout(silenceTimerRef.current); silenceTimerRef.current = null; }
    if (noSpeechTimerRef.current) { clearTimeout(noSpeechTimerRef.current); noSpeechTimerRef.current = null; }

    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      audioCtxRef.current?.close().catch(() => {});
      audioCtxRef.current = null;
      analyserRef.current = null;
      mediaRecorderRef.current = null;
      setStatus("idle");
      return;
    }

    setStatus("stopping");

    // Wire up onstop BEFORE calling stop() so we capture the final chunk.
    // cleanup() (unmount path) will NOT set onstop, so no spurious upload.
    recorder.onstop = () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      audioCtxRef.current?.close().catch(() => {});
      audioCtxRef.current = null;
      analyserRef.current = null;
      mediaRecorderRef.current = null;

      const chunks = recordedChunksRef.current;
      recordedChunksRef.current = [];
      const mimeType = recordingMimeTypeRef.current;
      const totalBytes = chunks.reduce((s, b) => s + b.size, 0);

      if (totalBytes < 100) {
        setStatus("idle");
        return;
      }

      setStatus("transcribing");

      const blob = new Blob(chunks, { type: mimeType });
      const form = new FormData();
      const ext = mimeType.includes("ogg") ? "ogg" : mimeType.includes("wav") ? "wav" : "webm";
      form.append("audio", blob, `recording.${ext}`);
      form.append("sample_rate", "16000");

      fetch(BACKEND_URL, { method: "POST", body: form })
        .then(async (res) => {
          if (!res.ok) {
            onErrorRef.current?.(
              res.status === 503
                ? "Whisper model unavailable — check backend logs."
                : `Transcription error (HTTP ${res.status})`,
            );
            return;
          }
          const data = (await res.json()) as { transcript: string; is_final: boolean };
          const text = data.transcript?.trim();
          if (text) onTranscriptRef.current(text, true);
        })
        .catch((err: unknown) => {
          console.error("[STT] fetch error:", err);
          onErrorRef.current?.("Could not reach transcription server.");
        })
        .finally(() => {
          setStatus("idle");
        });
    };

    try { recorder.requestData(); } catch { /* ignore */ }
    try { recorder.stop(); } catch { /* ignore */ }
  }, []);

  // Keep the ref in sync with the latest stop
  useEffect(() => { stopRef.current = stop; }, [stop]);

  // ── Silence detector ───────────────────────────────────────────────────────
  const startSilenceDetector = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const buf = new Float32Array(analyser.fftSize);

    const resetTimer = () => {
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = setTimeout(() => stopRef.current(), SILENCE_TIMEOUT_MS);
    };

    const tick = () => {
      if (!analyserRef.current) return;
      analyserRef.current.getFloatTimeDomainData(buf);
      const rms = Math.sqrt(buf.reduce((s, v) => s + v * v, 0) / buf.length);
      if (rms > SILENCE_THRESHOLD) resetTimer();
      silenceRafRef.current = requestAnimationFrame(tick);
    };

    resetTimer();
    silenceRafRef.current = requestAnimationFrame(tick);
  }, []);

  // ── Start ──────────────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    recordedChunksRef.current = [];

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (err) {
      console.error("[STT] getUserMedia error:", err);
      const msg = err instanceof Error ? err.message : String(err);
      if (
        msg.toLowerCase().includes("denied") ||
        msg.toLowerCase().includes("not allowed") ||
        msg.toLowerCase().includes("permission")
      ) {
        onPermissionDeniedRef.current?.();
      } else {
        onErrorRef.current?.(`Microphone unavailable: ${msg}`);
      }
      return;
    }

    streamRef.current = stream;

    try {
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;
      audioCtx.createMediaStreamSource(stream).connect(analyser);
    } catch (err) {
      console.error("[STT] AudioContext error:", err);
      // Non-fatal — silence detection just won't work
    }

    const mimeType = MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")
      ? "audio/ogg;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/ogg")
      ? "audio/ogg"
      : MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : "";

    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    } catch (err) {
      console.error("[STT] MediaRecorder init error:", err);
      onErrorRef.current?.("MediaRecorder not supported in this browser.");
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    mediaRecorderRef.current = recorder;
    recorder.ondataavailable = (e) => {
      if (e.data?.size > 0) {
        recordedChunksRef.current.push(e.data);
      }
    };
    recorder.onerror = (e) => {
      console.error("[STT] MediaRecorder error:", e);
      onErrorRef.current?.("MediaRecorder error.");
      stopRef.current();
    };
    recordingMimeTypeRef.current = mimeType || "audio/webm";
    recorder.start(CHUNK_MS);
    setStatus("listening");

    startSilenceDetector();

    noSpeechTimerRef.current = setTimeout(() => {
      onNoSpeechRef.current?.();
      stopRef.current();
    }, NO_SPEECH_TIMEOUT_MS);
  }, [startSilenceDetector]);

  // ── Toggle ─────────────────────────────────────────────────────────────────
  const toggle = useCallback(() => {
    if (isListening) { stop(); } else { start(); }
  }, [isListening, stop, start]);

  useEffect(() => () => { cleanup(); }, [cleanup]);

  return { status, isListening, toggle, stop };
}


