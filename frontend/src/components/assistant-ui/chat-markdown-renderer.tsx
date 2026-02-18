"use client";

import type { FC } from "react";
import { INTERNAL } from "@assistant-ui/react";
import { StreamdownTextPrimitive } from "@assistant-ui/react-streamdown";
import type { Citation } from "@/lib/event-parser";
import { useAppDispatch, useAppState } from "@/context/app-context";

function addCitationLinks(content: string, citations?: Citation[]): string {
  const withChunkMarkersLinked = content.replace(
    /\[CHUNK\s+(\d+)(?:\s*\|[^\]]*)?\](?!\()/gi,
    (_fullMatch, numberText) => `[${numberText}](/citation/${numberText})`,
  );

  if (!citations || citations.length === 0) {
    return withChunkMarkersLinked;
  }

  return withChunkMarkersLinked.replace(/\[(\d+)\](?!\()/g, (fullMatch, numberText) => {
    const index = Number.parseInt(numberText, 10) - 1;
    if (!Number.isInteger(index) || index < 0 || index >= citations.length) {
      return fullMatch;
    }
    return `[${numberText}](/citation/${numberText})`;
  });
}

function extractCitationIndex(href?: string): number | null {
  if (!href) return null;

  const decodedHref = (() => {
    try {
      return decodeURIComponent(href);
    } catch {
      return href;
    }
  })();

  const citationMatch =
    decodedHref.match(/(?:^|\/)citation[:/](\d+)(?:$|[/?#])/i) ??
    decodedHref.match(/\/citation\/(\d+)(?:$|[/?#])/i) ??
    decodedHref.match(/citation[:/](\d+)/i);
  if (!citationMatch) return null;

  const index = Number.parseInt(citationMatch[1], 10) - 1;
  return Number.isInteger(index) && index >= 0 ? index : null;
}

export const ChatMarkdownRenderer: FC = () => {
  const { citations } = useAppState();
  const dispatch = useAppDispatch();

  return (
    <StreamdownTextPrimitive
      mode="streaming"
      containerClassName="text-base leading-[1.65] min-h-[1.5em] break-words"
      preprocess={(raw) => addCitationLinks(raw, citations)}
      components={{
        a: ({ href, children, ...props }) => {
          const index = extractCitationIndex(href);
          if (index !== null) {
            const citation: Citation | undefined =
              citations.find((c) => c.number === index + 1) ?? citations[index];
            return (
              <span className="inline-flex items-center mx-0.5 align-middle">
                <button
                  type="button"
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    if (citation) {
                      dispatch({ type: "SET_ACTIVE_CITATION", citation });
                    }
                  }}
                  className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 text-[10px] font-semibold text-black rounded-full cursor-pointer transition-colors disabled:opacity-60 disabled:cursor-not-allowed bg-white/90 border border-white/80 hover:bg-white"
                  title={citation ? `View source: ${citation.source_id}` : "Source (no metadata)"}
                  disabled={!citation}
                >
                  {children}
                </button>
              </span>
            );
          }

          return (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-foreground hover:text-white underline underline-offset-2 decoration-white/40"
              {...props}
            >
              {children}
            </a>
          );
        },
      }}
    />
  );
};

export const ChatMarkdownRendererWithSmooth = INTERNAL.withSmoothContextProvider(ChatMarkdownRenderer);
