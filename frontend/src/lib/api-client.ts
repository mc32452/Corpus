/**
 * API client for source management endpoints.
 *
 * Provides typed wrappers around fetch calls to the FastAPI backend.
 */

import { getBackendApiBase as _getBackendApiBase } from "./backend-url";

export interface SourceInfo {
  source_id: string;
  summary: string | null;
  source_path: string | null;
  snapshot_path: string | null;
  source_size_bytes?: number | null;
  content_size_bytes?: number | null;
}

export interface SourceListResponse {
  sources: SourceInfo[];
}

export interface IngestResponse {
  source_id: string;
  parents_count: number;
  children_count: number;
  summarized: boolean;
}

export interface SourceContentResponse {
  source_id: string;
  content: string;
  content_source: "original" | "snapshot" | "summary";
  format: "pdf" | "markdown" | "text";
}

export interface SourceDeleteResponse {
  source_id: string;
  deleted: boolean;
}

/** A single citation entry emitted via the CITATIONS: stream line. */
export interface CitationEntry {
  index: number;
  source_id: string;
  chunk_id: string;
  page_number?: number | null;
  display_page?: string | null;
  header_path?: string;
  chunk_text: string;
  /** Corrected highlight passage from the parent chunk, present only when
   *  the cited content falls outside the child chunk boundaries. */
  highlight_text?: string;
}

/** Payload passed to CitationViewerModal on citation click. */
export interface CitationPayload {
  source_id: string;
  chunk_id?: string;
  page_number?: number | null;
  display_page?: string | null;
  header_path?: string;
  chunk_text?: string;
  /** Corrected highlight passage from parent chunk (post-hoc verification). */
  highlight_text?: string;
}

export interface ChunkDetailResponse {
  source_id: string;
  chunk_id: string;
  chunk_text: string;
  parent_text?: string | null;
  page_number?: number | null;
  display_page?: string | null;
  header_path: string;
  format: "pdf" | "markdown" | "text";
  source_path?: string | null;
}

export interface ChunkBatchItem {
  source_id: string;
  chunk_id: string;
  chunk_text: string;
  page_number?: number | null;
  display_page?: string | null;
  header_path: string;
  format: "pdf" | "markdown" | "text";
  source_path?: string | null;
}

export interface ChunkBatchResponse {
  chunks: ChunkBatchItem[];
}

export interface ApiError {
  error: {
    code: string;
    message: string;
  };
}

export type StreamEvent = {
  event: "status" | "intent" | "sources" | "citations" | "token" | "error" | "complete";
  data: Record<string, unknown>;
};

export interface QueryStreamingOptions {
  sourceIds?: string[];
  citationsEnabled?: boolean;
  mode?: string;
  signal?: AbortSignal;
}

/**
 * Legacy stream helper used by ChatPanel.
 *
 * Consumes the AI SDK UI message stream from `/api/query` and maps it into
 * simple event records (`status`, `intent`, `sources`, `citations`, `token`,
 * `error`, `complete`) expected by the legacy chat renderer.
 */
export async function* queryStreaming(
  query: string,
  options: QueryStreamingOptions = {}
): AsyncGenerator<StreamEvent, void, unknown> {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      source_ids: options.sourceIds,
      citations_enabled: options.citationsEnabled ?? true,
      stream: true,
      mode: options.mode,
    }),
    signal: options.signal,
  });

  if (!res.ok || !res.body) {
    let message = `Query failed: HTTP ${res.status}`;
    try {
      const body = (await res.json()) as ApiError;
      if (body?.error?.message) message = body.error.message;
    } catch {
      try {
        const text = await res.text();
        if (text) message = text;
      } catch {
        // keep fallback message
      }
    }
    throw new Error(message);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const parseBlock = (block: string): string | null => {
    const lines = block.split(/\r?\n/);
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!dataLines.length) return null;
    return dataLines.join("\n");
  };

  const mapPayload = (payload: unknown): StreamEvent[] => {
    if (payload === "[DONE]") {
      return [{ event: "complete", data: {} }];
    }
    if (!payload || typeof payload !== "object") return [];

    const frame = payload as Record<string, unknown>;
    const type = frame.type;
    if (typeof type !== "string") return [];

    switch (type) {
      case "text-delta": {
        const text = typeof frame.delta === "string" ? frame.delta : "";
        return text ? [{ event: "token", data: { text } }] : [];
      }
      case "data-status": {
        const data = (frame.data ?? {}) as Record<string, unknown>;
        const status = typeof data.status === "string" ? data.status : "Working...";
        return [{ event: "status", data: { message: status } }];
      }
      case "data-intent": {
        const data = (frame.data ?? {}) as Record<string, unknown>;
        return [{
          event: "intent",
          data: {
            intent: data.intent,
            confidence: data.confidence,
            method: data.method,
          },
        }];
      }
      case "data-sources": {
        const data = (frame.data ?? {}) as Record<string, unknown>;
        const sourceIds = Array.isArray(data.sourceIds)
          ? data.sourceIds.filter((x): x is string => typeof x === "string")
          : [];
        return [{ event: "sources", data: { source_ids: sourceIds } }];
      }
      case "data-citations": {
        const data = (frame.data ?? {}) as Record<string, unknown>;
        const citations = Array.isArray(data.citations) ? data.citations : [];
        return [{ event: "citations", data: { citations } }];
      }
      case "data-error": {
        const data = (frame.data ?? {}) as Record<string, unknown>;
        const err = (data.error ?? {}) as Record<string, unknown>;
        const message = typeof err.message === "string" ? err.message : "Streaming error";
        return [{ event: "error", data: { error: message, code: err.code } }];
      }
      case "error": {
        const message = typeof frame.error === "string" ? frame.error : "Streaming error";
        return [{ event: "error", data: { error: message } }];
      }
      case "finish": {
        return [{ event: "complete", data: {} }];
      }
      default:
        return [];
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const rawData = parseBlock(block);
      if (!rawData) continue;

      if (rawData === "[DONE]") {
        yield { event: "complete", data: {} };
        continue;
      }

      try {
        const parsed = JSON.parse(rawData) as unknown;
        const mapped = mapPayload(parsed);
        for (const evt of mapped) {
          yield evt;
        }
      } catch {
        // ignore malformed frames and keep streaming
      }
    }
  }

  if (buffer.trim()) {
    const rawData = parseBlock(buffer.trim());
    if (rawData === "[DONE]") {
      yield { event: "complete", data: {} };
    } else if (rawData) {
      try {
        const parsed = JSON.parse(rawData) as unknown;
        const mapped = mapPayload(parsed);
        for (const evt of mapped) {
          yield evt;
        }
      } catch {
        // ignore malformed trailing frame
      }
    }
  }
}

class SourceApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = "/api") {
    this.baseUrl = baseUrl;
  }

  /**
   * Parse an error response that may or may not be JSON.
   * Returns a human-readable error message.
   */
  private async parseErrorResponse(res: Response): Promise<string> {
    try {
      const body = await res.json();
      return body?.error?.message ?? `HTTP ${res.status}`;
    } catch {
      // Response body is not valid JSON (e.g. plain "Internal Server Error")
      try {
        const text = await res.text();
        return text || `HTTP ${res.status}`;
      } catch {
        return `HTTP ${res.status}`;
      }
    }
  }

  /**
   * Resolve the backend URL for long-running operations that must
   * bypass the Next.js dev proxy to avoid its ~30s timeout.
   */
  private getDirectBackendUrl(path: string): string {
    return `${_getBackendApiBase()}${path}`;
  }

  async listSources(): Promise<SourceInfo[]> {
    const res = await fetch(`${this.baseUrl}/sources`);
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    const data: SourceListResponse = await res.json();
    return data.sources;
  }

  async getContent(sourceId: string): Promise<SourceContentResponse> {
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}/content`
    );
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    return res.json();
  }

  async deleteSource(sourceId: string): Promise<SourceDeleteResponse> {
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}`,
      { method: "DELETE" }
    );
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    return res.json();
  }

  async ingest(
    filePath: string,
    sourceId: string,
    summarize: boolean = true
  ): Promise<IngestResponse> {
    const res = await fetch(`${this.baseUrl}/sources/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_path: filePath,
        source_id: sourceId,
        summarize,
      }),
    });
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    return res.json();
  }

  async uploadDocument(
    file: File,
    sourceId: string,
    summarize: boolean = true
  ): Promise<IngestResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("source_id", sourceId);
    formData.append("summarize", String(summarize));

    // Call the backend directly to bypass the Next.js dev proxy
    // which has a ~30s timeout — upload+ingest can take minutes.
    const url = this.getDirectBackendUrl("/sources/upload");
    const res = await fetch(url, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async getChunk(
    sourceId: string,
    chunkId: string
  ): Promise<ChunkDetailResponse> {
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}/chunk/${encodeURIComponent(chunkId)}`
    );
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    return res.json();
  }

  async getChunks(
    sourceId: string,
    chunkIds: string[]
  ): Promise<ChunkBatchResponse> {
    if (!chunkIds.length) return { chunks: [] };
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}/chunks?ids=${encodeURIComponent(chunkIds.join(","))}`
    );
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    return res.json();
  }
}

export const sourceApi = new SourceApiClient();
