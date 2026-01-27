from __future__ import annotations

import re

_BULLET_RE = re.compile(r"^\s*[-•]\s+")


def build_prompt(context: str, question: str) -> str:
    instruction = (
        "You are a helpful research assistant.\n"
        "Task: Summarize the retrieved context to answer the user's question.\n"
        "Constraints:\n"
        "1. Start directly with bullet points. Do NOT write an introduction paragraph.\n"
        "2. Provide exactly 3-5 distinct bullet points.\n"
        "3. Stop writing immediately after the last bullet point.\n"
        "4. Do not repeat the same point in different words."
    )

    return (
        f"System: {instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


def postprocess_bullets(text: str) -> str:
    if not text:
        return text

    lines = text.splitlines()
    bullets: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Assistant:") or stripped.startswith("Human:"):
            continue
        if stripped.startswith("Note:"):
            continue
        if _BULLET_RE.match(stripped):
            if current:
                bullets.append(" ".join(current).strip())
            current = [_BULLET_RE.sub("", stripped).strip()]
        else:
            if current:
                current.append(stripped)

    if current:
        bullets.append(" ".join(current).strip())

    bullets = [b for b in bullets if b]
    if len(bullets) < 3:
        inline_matches = re.findall(
            r"(?:^|\s)-\s+(.+?)(?=(?:\s-\s+|$))",
            text,
            flags=re.DOTALL,
        )
        bullets = [
            re.sub(r"\s+", " ", b).strip()
            for b in inline_matches
            if b.strip()
        ]

    if not bullets:
        return text.strip()

    if len(bullets) >= 3:
        bullets = bullets[:5]

    return "\n".join(f"- {bullet}" for bullet in bullets)
