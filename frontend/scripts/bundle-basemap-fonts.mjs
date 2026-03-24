#!/usr/bin/env node

import { mkdir, access, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");

const targetRoot = path.join(projectRoot, "public", "basemap-assets", "fonts");
const baseUrl = "https://protomaps.github.io/basemaps-assets/fonts";

// Keep this intentionally tight for offline startup and current style usage.
const fontStacks = ["Noto Sans Regular", "Noto Sans Medium", "Noto Sans Italic"];
const ranges = [
  "0-255",
  "256-511",
  "512-767",
  "768-1023",
  "1024-1279",
  "1536-1791",
  "11520-11775",
];

async function fileExists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function download(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} for ${url}`);
  }
  const arrayBuffer = await res.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

async function ensureFontGlyph(fontStack, range) {
  const fontDir = path.join(targetRoot, fontStack);
  const filePath = path.join(fontDir, `${range}.pbf`);

  if (await fileExists(filePath)) {
    return { status: "exists", filePath };
  }

  await mkdir(fontDir, { recursive: true });
  const encodedStack = encodeURIComponent(fontStack).replace(/%20/g, "%20");
  const url = `${baseUrl}/${encodedStack}/${range}.pbf`;
  const data = await download(url);
  await writeFile(filePath, data);
  return { status: "downloaded", filePath };
}

async function main() {
  await mkdir(targetRoot, { recursive: true });

  let downloaded = 0;
  let existing = 0;

  for (const stack of fontStacks) {
    for (const range of ranges) {
      const result = await ensureFontGlyph(stack, range);
      if (result.status === "downloaded") {
        downloaded += 1;
        console.log(`downloaded ${result.filePath}`);
      } else {
        existing += 1;
      }
    }
  }

  console.log(`basemap font bundle complete: ${downloaded} downloaded, ${existing} existing`);
}

main().catch((err) => {
  console.error(`failed to bundle basemap fonts: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
