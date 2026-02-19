"use client";

/**
 * SpeechToTextButton
 *
 * Toggleable mic button.  Delegates recording + transcription to
 * useSpeechToText, which POSTs audio chunks to the local Python backend.
 * Moonshine Tiny transcribes them server-side — fully offline.
 *
 * Props:
 *   inputRef  – ref to the <input> managed by ChatPanel
 *   value     – current controlled input value
 *   onChange  – setter for that value
 *   disabled  – true while the AI is streaming a response
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useSpeechToText } from "@/hooks/useSpeechToText";

interface SpeechToTextButtonProps {
  inputRef: React.RefObject<HTMLInputElement | null>;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

// ── Minimal toast (self-contained, no extra dependency) ──────────────────────

interface ToastState {
  message: string;
  type: "info" | "warning" | "error";
  id: number;
}

// ─────────────────────────────────────────────────────────────────────────────

export function SpeechToTextButton({
  inputRef,
  value,
  onChange,
  disabled = false,
}: SpeechToTextButtonProps) {
  const [toast, setToast] = useState<ToastState | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toastIdRef = useRef(0);

  // interimRef: in-progress (non-final) text already appended to the input
  const interimRef = useRef("");
  // sessionStartPosRef: string index where this dictation session started inserting
  const sessionStartPosRef = useRef<number | null>(null);

  const showToast = useCallback(
    (message: string, type: ToastState["type"] = "info", duration = 3000) => {
      const id = ++toastIdRef.current;
      setToast({ message, type, id });
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
      toastTimerRef.current = setTimeout(() => {
        setToast((prev) => (prev?.id === id ? null : prev));
      }, duration);
    },
    []
  );

  const dismissToast = useCallback(() => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    setToast(null);
  }, []);

  // ── Transcript handler (cursor-aware insertion) ───────────────────────────
  // We keep a ref to `value` so the callback always sees the current value
  // without needing to be recreated on every keystroke.
  const valueRef = useRef(value);
  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  const handleTranscript = useCallback(
    (chunk: string, isFinal: boolean) => {
      // End-of-session signal (empty final from stop())
      if (isFinal && !chunk) {
        interimRef.current = "";
        sessionStartPosRef.current = null;
        return;
      }

      const current = valueRef.current;

      if (sessionStartPosRef.current === null) {
        sessionStartPosRef.current =
          inputRef.current?.selectionStart ?? current.length;
      }

      const anchor = sessionStartPosRef.current;
      const before = current.slice(0, anchor);
      const after = current.slice(anchor + interimRef.current.length);
      const sep = before.length > 0 && !before.endsWith(" ") && chunk ? " " : "";
      const inserted = sep + chunk;
      const newValue = before + inserted + after;

      if (isFinal) {
        sessionStartPosRef.current = anchor + inserted.length;
        interimRef.current = "";
      } else {
        interimRef.current = inserted;
      }

      onChange(newValue);
      requestAnimationFrame(() => {
        const pos = sessionStartPosRef.current ?? newValue.length;
        inputRef.current?.setSelectionRange(pos, pos);
      });
    },
    [inputRef, onChange]
  );

  const handlePermissionDenied = useCallback(() => {
    showToast(
      "Mic access needed. Allow it in your browser settings.",
      "error",
      8000
    );
  }, [showToast]);

  const handleNoSpeech = useCallback(() => {
    showToast("No speech heard. Tap to try again.", "info", 4000);
  }, [showToast]);

  const handleError = useCallback(
    (msg: string) => { showToast(msg, "error", 5000); },
    [showToast]
  );

  const { isListening, toggle } = useSpeechToText({
    onTranscript: handleTranscript,
    onPermissionDenied: handlePermissionDenied,
    onNoSpeech: handleNoSpeech,
    onError: handleError,
  });

  // Show / dismiss the "Listening…" toast
  useEffect(() => {
    if (isListening) {
      showToast("Listening…", "info", 120_000); // dismissed on stop
      interimRef.current = "";
      sessionStartPosRef.current = null;
    } else {
      dismissToast();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isListening]);

  const handleClick = useCallback(() => {
    if (disabled) {
      showToast("Wait for the current response to finish.", "warning", 2500);
      return;
    }
    toggle();
  }, [disabled, toggle, isListening, showToast]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="relative z-20 flex items-center pointer-events-auto">
      <button
        type="button"
        aria-label={isListening ? "Stop listening" : "Voice input"}
        onClick={handleClick}
        title={isListening ? "Stop listening" : "Voice input (offline)"}
        className={[
          "relative z-20 w-10 h-10 flex items-center justify-center rounded-full transition-all duration-150 shrink-0 pointer-events-auto",
          isListening
            ? "bg-red-600 hover:bg-red-700 text-white shadow-lg shadow-red-900/40"
            : disabled
            ? "bg-gray-800 text-gray-600 cursor-not-allowed"
            : "bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white",
        ].join(" ")}
      >
        {/* Pulsing ring when listening */}
        {isListening && (
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-full bg-red-500 opacity-25 animate-ping"
          />
        )}

        {/* Mic icon */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          className={isListening ? "animate-pulse" : ""}
        >
          <rect x="9" y="2" width="6" height="13" rx="3" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="22" />
          <line x1="8" y1="22" x2="16" y2="22" />
        </svg>
      </button>

      {/* Toast */}
      {toast && (
        <div
          role="status"
          aria-live="polite"
          className={[
            "absolute bottom-12 right-0 z-50 px-3 py-1.5 rounded-lg text-xs whitespace-nowrap shadow-lg pointer-events-none",
            "transition-opacity duration-200",
            toast.type === "error"
              ? "bg-red-900 text-red-200 border border-red-700"
              : toast.type === "warning"
              ? "bg-yellow-900 text-yellow-200 border border-yellow-700"
              : "bg-gray-700 text-gray-100 border border-gray-600",
          ].join(" ")}
        >
          {toast.message}
        </div>
      )}
    </div>
  );
}
