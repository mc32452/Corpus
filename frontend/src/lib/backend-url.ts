/**
 * Single source of truth for the backend API base URL.
 *
 * Set the `NEXT_PUBLIC_BACKEND_URL` env var to override (e.g. in `.env.local`).
 * Defaults to `http://127.0.0.1:8000` for local development.
 */

const _DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

/**
 * Returns the bare backend origin (no trailing slash, no `/api`).
 * Example: `"http://127.0.0.1:8000"`
 */
export function getBackendBase(): string {
  const env =
    typeof process !== "undefined" &&
    typeof process.env?.NEXT_PUBLIC_BACKEND_URL === "string"
      ? process.env.NEXT_PUBLIC_BACKEND_URL
      : "";
  return (env || _DEFAULT_BACKEND_URL).replace(/\/$/, "");
}

/**
 * Returns the backend API base (origin + `/api`).
 * Example: `"http://127.0.0.1:8000/api"`
 */
export function getBackendApiBase(): string {
  return `${getBackendBase()}/api`;
}
