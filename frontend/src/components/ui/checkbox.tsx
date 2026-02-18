"use client";

import * as React from "react";

interface CheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
  id?: string;
}

/**
 * Custom dark-theme checkbox. Renders a styled <button> + hidden <input>
 * so native appearance CSS conflicts can't affect the visual.
 */
export function Checkbox({
  checked,
  onChange,
  disabled = false,
  className = "",
  id,
}: CheckboxProps) {
  return (
    <span className={`relative inline-flex shrink-0 ${className}`}>
      {/* Hidden real input for accessibility / form submission */}
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        aria-hidden="true"
        tabIndex={-1}
        style={{ position: "absolute", opacity: 0, width: 0, height: 0, margin: 0, padding: 0, border: 0 }}
      />
      {/* Custom visual */}
      <button
        type="button"
        role="checkbox"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={[
          "w-4 h-4 rounded-[4px] shrink-0 flex items-center justify-center",
          "transition-colors duration-100",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/30",
          checked
            ? "bg-white border border-white"
            : "bg-transparent border border-[#555] hover:border-[#888]",
          disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer",
        ].join(" ")}
      >
        {checked && (
          <svg
            viewBox="0 0 24 24"
            className="w-3 h-3 text-black"
            fill="none"
            stroke="currentColor"
            strokeWidth={2.8}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M5 13l4 4L19 7" />
          </svg>
        )}
      </button>
    </span>
  );
}
