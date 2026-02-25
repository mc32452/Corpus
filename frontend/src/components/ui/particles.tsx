"use client";

import React from "react";
import { GL } from "@/components/gl";
import { cn } from "@/lib/utils";

export interface ParticleBackgroundProps {
  className?: string;
  style?: React.CSSProperties;
  hovering?: boolean;
}

export const ParticleBackground = React.memo(({
  className,
  style,
  hovering = false,
}: ParticleBackgroundProps) => {
  return (
    <div
      className={cn("pointer-events-none absolute inset-0 h-full w-full", className)}
      style={style}
    >
      <GL hovering={hovering} />
    </div>
  );
});

ParticleBackground.displayName = "ParticleBackground";

export default ParticleBackground;
