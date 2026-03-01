"use client";

import React, { createContext, useCallback, useContext, useEffect, useState } from "react";

export type BackgroundTheme = "none" | "beams" | "meteors" | "rain" | "mesh" | "paths" | "starfield" | "particles";

const STORAGE_KEY = "dh-background-theme";

interface ThemeContextValue {
  theme: BackgroundTheme;
  setTheme: (t: BackgroundTheme) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "none",
  setTheme: () => {},
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<BackgroundTheme>("none");

  // Hydrate from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY) as BackgroundTheme | null;
      if (stored) setThemeState(stored);
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
