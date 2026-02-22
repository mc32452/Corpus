"use client";

import { useState, useMemo, useCallback } from "react";
import { useAuiState } from "@assistant-ui/react";
import { useAppState, useAppDispatch } from "@/context/app-context";
import { groupCitations } from "@/lib/group-citations";
import { sourceApi } from "@/lib/api-client";
import { ChevronDownIcon, ChevronRightIcon } from "lucide-react";

export function MessageReferences() {
    const messageId = useAuiState((s) => s.message.id);
    const isRunning = useAuiState((s) => s.message.status?.type === "running");
    const { citationsByMessage } = useAppState();
    const dispatch = useAppDispatch();
    const citations = citationsByMessage[messageId] || [];

    const [isDrawerOpen, setIsDrawerOpen] = useState(false);
    const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
    const [chunkCache, setChunkCache] = useState<Record<string, string>>({});
    const [loadingSources, setLoadingSources] = useState<Set<string>>(new Set());

    const grouped = useMemo(() => groupCitations(citations), [citations]);

    const toggleDrawer = () => setIsDrawerOpen((prev) => !prev);

    const toggleSource = useCallback(
        async (sourceId: string, chunkIds: string[]) => {
            const isCurrentlyExpanded = expandedSources.has(sourceId);

            setExpandedSources((prev) => {
                const next = new Set(prev);
                if (isCurrentlyExpanded) {
                    next.delete(sourceId);
                } else {
                    next.add(sourceId);
                }
                return next;
            });

            if (isCurrentlyExpanded) return;

            // Fetch missing chunks
            const missingChunkIds = chunkIds.filter((id) => !chunkCache[id]);
            if (missingChunkIds.length > 0) {
                setLoadingSources((prev) => new Set(prev).add(sourceId));
                try {
                    const resp = await sourceApi.getChunks(sourceId, missingChunkIds);
                    setChunkCache((prev) => {
                        const next = { ...prev };
                        for (const chunk of resp.chunks) {
                            next[chunk.chunk_id] = chunk.chunk_text;
                        }
                        return next;
                    });
                } catch (error) {
                    console.error("Failed to fetch chunks for overview:", error);
                } finally {
                    setLoadingSources((prev) => {
                        const next = new Set(prev);
                        next.delete(sourceId);
                        return next;
                    });
                }
            }
        },
        [chunkCache, expandedSources]
    );

    const totalChunks = citations.length;
    const totalSources = grouped.length;

    if (grouped.length === 0 || isRunning) {
        return null;
    }

    return (
        <div className="mt-2 mb-1 w-full flex flex-col items-start border-t border-white/10 pt-2 pb-1 text-sm transition-all duration-200">
            <button
                onClick={toggleDrawer}
                className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground outline-none group w-fit transition-colors"
            >
                {isDrawerOpen ? (
                    <ChevronDownIcon className="size-4 shrink-0 transition-transform" />
                ) : (
                    <ChevronRightIcon className="size-4 shrink-0 transition-transform" />
                )}
                <span className="font-medium">
                    References ({totalSources} source{totalSources !== 1 ? "s" : ""},{" "}
                    {totalChunks} chunk{totalChunks !== 1 ? "s" : ""})
                </span>
            </button>

            {isDrawerOpen && (
                <div className="w-full mt-3 flex flex-col gap-3 pl-2 sm:pl-4">
                    {grouped.map((group) => {
                        const isExpanded = expandedSources.has(group.sourceId);
                        const isLoading = loadingSources.has(group.sourceId);
                        const chunkIds = group.citations.map((c) => c.chunkId);

                        return (
                            <div key={group.sourceId} className="flex flex-col text-sm text-foreground/90 w-full mb-1">
                                <div className="flex flex-col mb-1 border-l-2 border-[#333] pl-3 py-0.5">
                                    <div className="font-semibold text-foreground break-words line-clamp-2">
                                        {group.displayName}
                                    </div>
                                    <div className="flex items-center gap-1.5 flex-wrap mt-1">
                                        {group.citations.map((cit, idx) => (
                                            <div key={cit.index} className="flex items-center gap-1.5 whitespace-nowrap">
                                                <button
                                                    onClick={() => {
                                                        const matchingCitation = citations.find((c) => c.number === cit.index);
                                                        if (matchingCitation) {
                                                            dispatch({ type: "SET_ACTIVE_CITATION", citation: matchingCitation });
                                                        }
                                                    }}
                                                    className="inline-flex items-center justify-center min-w-[20px] h-[20px] px-1 text-[11px] font-bold text-black rounded-full cursor-pointer transition-colors bg-white/90 hover:bg-white"
                                                    title={`View inline citation [${cit.index}]`}
                                                >
                                                    [{cit.index}]
                                                </button>
                                                {cit.page !== null && (
                                                    <span className="text-muted-foreground text-xs font-mono">p.{cit.page}</span>
                                                )}
                                                {idx < group.citations.length - 1 && (
                                                    <span className="text-muted-foreground/30 text-xs mx-0.5">&middot;</span>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="mt-1 pl-3">
                                    <button
                                        onClick={() => toggleSource(group.sourceId, chunkIds)}
                                        className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors text-xs font-semibold"
                                    >
                                        {isExpanded ? (
                                            <ChevronDownIcon className="size-3.5" />
                                        ) : (
                                            <ChevronRightIcon className="size-3.5" />
                                        )}
                                        {isExpanded ? "Hide retrieved passages" : "Show retrieved passages"}
                                    </button>

                                    {isExpanded && (
                                        <div className="flex flex-col gap-3 mt-3 w-full max-w-full overflow-hidden">
                                            {group.citations.map((cit) => {
                                                const text = cit.text || chunkCache[cit.chunkId];
                                                return (
                                                    <div
                                                        key={cit.index}
                                                        className="flex flex-col bg-[#1e1e1e]/60 border border-white/10 rounded-md p-3 max-w-[95%]"
                                                    >
                                                        <div className="text-xs font-mono text-muted-foreground mb-1.5 flex justify-between items-center group-hover:text-foreground">
                                                            <span>[{cit.index}]{cit.page !== null ? ` p.${cit.page}` : ""}</span>
                                                        </div>
                                                        {isLoading && !text ? (
                                                            <div className="h-4 w-1/3 bg-white/10 animate-pulse rounded"></div>
                                                        ) : text ? (
                                                            <div className="break-words leading-relaxed text-[13px] text-white/80 line-clamp-none max-h-60 overflow-y-auto pr-1">
                                                                {text.length > 500 ? (
                                                                    <>
                                                                        {text.substring(0, 500)}...
                                                                        <span className="ml-1 text-xs text-blue-400 opacity-80">(Continues in raw passage)</span>
                                                                    </>
                                                                ) : (
                                                                    text
                                                                )}
                                                            </div>
                                                        ) : (
                                                            <div className="text-red-400 text-xs italic">Failed to load snippet.</div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
