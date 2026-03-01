"use client";

import { useEffect, useState } from "react";

const BACKEND_BASE = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

/**
 * Fetches system RAM from the backend health endpoint once on mount.
 * Returns `null` until the value is available (or if the fetch fails).
 */
export function useSystemRam(): number | null {
  const [ramGb, setRamGb] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${BACKEND_BASE}/api/health`)
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled && typeof data.system_ram_gb === "number") {
          setRamGb(data.system_ram_gb);
        }
      })
      .catch(() => {
        /* health endpoint unavailable — leave as null */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return ramGb;
}
