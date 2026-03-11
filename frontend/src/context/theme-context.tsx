"use client";

import React, { createContext, useCallback, useContext, useEffect, useState } from "react";

export type BackgroundTheme = "stars" | "meteors" | "rain" | "mesh" | "starfield" | "particles";

const STORAGE_KEY = "dh-background-theme";

interface ThemeContextValue {
  theme: BackgroundTheme;
  setTheme: (t: BackgroundTheme) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "stars",
  setTheme: () => {},
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<BackgroundTheme>("stars");

  // Hydrate from localStorage on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      // "beams" and "none" were removed; fall back to "stars"
      if (raw && raw !== "beams" && raw !== "none") setThemeState(raw as BackgroundTheme);
    } catch {}
  }, []);

  const setTheme = useCallback((t: BackgroundTheme) => {
    setThemeState(t);
    try {
      localStorage.setItem(STORAGE_KEY, t);
    } catch {}
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
