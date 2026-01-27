from __future__ import annotations


def build_prompt(context: str, question: str) -> str:
    instruction = (
        "Answer the user's question concisely in 3-4 sentences. "
        "Avoid repetition. If the answer involves multiple distinct points, "
        "use a bulleted list."
    )

    return (
        f"System: {instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
