import type { Citation } from "./event-parser";

export type GroupedReferenceCitation = {
  index: number;
  chunkId: string;
  page: number | null;
  text?: string;
};

export type GroupedReference = {
  sourceId: string;
  displayName: string;
  pages: number[];
  citations: GroupedReferenceCitation[];
};

export function groupCitations(citations: Citation[]): GroupedReference[] {
  if (!citations || citations.length === 0) return [];

  const groups = new Map<string, GroupedReference>();

  for (const citation of citations) {
    if (!groups.has(citation.source_id)) {
      groups.set(citation.source_id, {
        sourceId: citation.source_id,
        displayName: citation.source_id, // For display we just use the ID if we don't have a resolve map
        pages: [],
        citations: [],
      });
    }

    const group = groups.get(citation.source_id)!;
    
    // Check if we already added a citation with this index to avoid duplicates 
    // (though in theory the backend shouldn't send duplicates for the same chunk/index)
    const exists = group.citations.some(c => c.index === citation.number && c.chunkId === citation.chunk_id);
    if (!exists) {
      group.citations.push({
        index: citation.number,
        chunkId: citation.chunk_id,
        page: citation.page ?? null,
        text: citation.chunk_text,
      });

      if (citation.page !== undefined && citation.page !== null && !group.pages.includes(citation.page)) {
        group.pages.push(citation.page);
      }
    }
  }

  const result = Array.from(groups.values());

  // Sort pages numerically
  for (const group of result) {
    group.pages.sort((a, b) => a - b);
    group.citations.sort((a, b) => a.index - b.index);
  }

  // Sort groups by the lowest citation index in each group
  result.sort((a, b) => {
    const minA = a.citations[0]?.index ?? Infinity;
    const minB = b.citations[0]?.index ?? Infinity;
    return minA - minB;
  });

  return result;
}
