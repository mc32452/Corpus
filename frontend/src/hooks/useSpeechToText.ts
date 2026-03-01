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

export type STTStatus = "idle" | "listening" | "stopping" | "error";

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
  const gotSpeechRef = useRef(false);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordingMimeTypeRef = useRef("audio/webm");
  const flushInProgressRef = useRef(false);
  const chunkInFlightRef = useRef(false);
  const queuedChunkRef = useRef<{ blob: Blob; markFinal: boolean } | null>(null);

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
    queuedChunkRef.current = null;
    chunkInFlightRef.current = false;
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
  const flushFullRecordingRef = useRef<() => Promise<void>>(async () => {});

  const stop = useCallback(() => {
    try {
      mediaRecorderRef.current?.requestData();
    } catch {
      // ignore requestData errors
    }
    cleanup();
    void flushFullRecordingRef.current().finally(() => {
      onTranscriptRef.current("", true); // end-of-session signal
    });
    setStatus("idle");
  }, [cleanup]);

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

  // ── Send chunk to Moonshine backend ───────────────────────────────────────
  const sendChunkNow = useCallback(async (blob: Blob, markFinal = false) => {
    if (blob.size < 100) return;
    const form = new FormData();
    const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("wav") ? "wav" : "webm";
    form.append("audio", blob, `chunk.${ext}`);
    form.append("sample_rate", "16000");
    try {
      const res = await fetch(BACKEND_URL, { method: "POST", body: form });
      if (!res.ok) {
        onErrorRef.current?.(res.status === 503
          ? "Whisper model unavailable — check backend logs."
          : `Transcription error (HTTP ${res.status})`);
        return;
      }
      const data = (await res.json()) as { transcript: string; is_final: boolean };
      const text = data.transcript?.trim();
      if (text) { gotSpeechRef.current = true; onTranscriptRef.current(text, markFinal); }
    } catch (err) {
      console.error("[STT] fetch error:", err);
      onErrorRef.current?.("Could not reach transcription server.");
    }
  }, []);

  const processQueuedChunk = useCallback(async () => {
    if (chunkInFlightRef.current) return;
    const next = queuedChunkRef.current;
    if (!next) return;

    queuedChunkRef.current = null;
    chunkInFlightRef.current = true;
    try {
      await sendChunkNow(next.blob, next.markFinal);
    } finally {
      chunkInFlightRef.current = false;
      if (queuedChunkRef.current) {
        void processQueuedChunk();
      }
    }
  }, [sendChunkNow]);

  const enqueueChunk = useCallback((blob: Blob, markFinal = false) => {
    if (blob.size < 100) return;
    queuedChunkRef.current = { blob, markFinal }; // keep latest chunk only
    void processQueuedChunk();
  }, [processQueuedChunk]);

  const flushFullRecording = useCallback(async () => {
    if (flushInProgressRef.current) return;
    if (gotSpeechRef.current) return;
    if (recordedChunksRef.current.length === 0) return;

    flushInProgressRef.current = true;
    try {
      const fullBlob = new Blob(recordedChunksRef.current, {
        type: recordingMimeTypeRef.current,
      });
      await sendChunkNow(fullBlob, true);
    } finally {
      flushInProgressRef.current = false;
    }
  }, [sendChunkNow]);

  useEffect(() => {
    flushFullRecordingRef.current = flushFullRecording;
  }, [flushFullRecording]);

  // ── Start ──────────────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    gotSpeechRef.current = false;
    recordedChunksRef.current = [];
    flushInProgressRef.current = false;

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
        enqueueChunk(e.data, false);
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
      if (!gotSpeechRef.current) {
        onNoSpeechRef.current?.();
        void flushFullRecording().finally(() => stopRef.current());
      }
    }, NO_SPEECH_TIMEOUT_MS);
  }, [enqueueChunk, flushFullRecording, startSilenceDetector]);

  // ── Toggle ─────────────────────────────────────────────────────────────────
  const toggle = useCallback(() => {
    if (isListening) { stop(); } else { start(); }
  }, [isListening, stop, start]);

  useEffect(() => () => { cleanup(); }, [cleanup]);

  return { status, isListening, toggle, stop };
}


