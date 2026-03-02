"""
Callback Tool-Use Prototype Validation
=======================================
This script validates four critical assumptions before implementing the
callback tool-use system.  Run it directly:

    python tests/test_callback_prototype.py

It loads the raw model via mlx_lm (no MlxGenerator wrapper) and tests:
  1. Does tokenizer.apply_chat_template(..., tools=...) work?
  2. Does the model produce parseable <tool_call> output when context is weak?
  3. Does the model produce a clean decline when context is strong?
  4. How long does each callback generation take?
"""

from __future__ import annotations

import json
import re
import time
import sys
from typing import Any

# ── Model path (same as src/config.py) ───────────────────────────────
MODEL_PATH = "NexVeridian/Qwen3.5-35B-A3B-4bit"

# ── Tool definition (single search tool from the plan) ───────────────
SEARCH_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the ingested knowledge base for relevant passages. "
            "Use this when your initial answer lacks sufficient evidence, "
            "contains uncertain claims, or missed key information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant passages.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["sparse", "dense", "hybrid"],
                    "description": (
                        "Search mode. 'sparse' for keyword/BM25 (facts, names, dates). "
                        "'dense' for semantic similarity. "
                        "'hybrid' for combined retrieval with reranking (default)."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum passages to return (1-8, default 5).",
                },
            },
            "required": ["query"],
        },
    },
}

# ── Callback instruction (the prompt the model sees after its answer) ─
CALLBACK_INSTRUCTION = """\
Review the answer you just gave against the retrieved context passages above.

Assess whether:
1. Every factual claim is supported by the provided context
2. Key information from the context was not missed or overlooked
3. The answer would benefit from additional evidence on any specific point

If the answer is well-supported and complete, respond with exactly: ANSWER_SUFFICIENT

If the answer needs better evidence, call the search tool ONCE with a targeted query to retrieve additional relevant passages. After receiving the search results, provide a revised and improved answer that incorporates the new evidence. Maintain the same citation format [N] used in the original answer, continuing the numbering sequence.

You may call the search tool at most once. Do not explain your reasoning — either call the tool or respond with ANSWER_SUFFICIENT."""

# ── Tool call parser ─────────────────────────────────────────────────
# Qwen3.5 produces TWO possible tool-call formats depending on the
# chat template version:
#
# Format A (XML-parameter, observed with this model):
#   <tool_call>
#   <function=search>
#   <parameter=query>value</parameter>
#   <parameter=max_results>5</parameter>
#   </function>
#   </tool_call>
#
# Format B (JSON, seen in some Qwen3 variants):
#   <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>

# Format A: XML-parameter style
_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)

# Format B: JSON style
_TOOL_CALL_JSON_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Parse a Qwen3.5 tool call from either XML-parameter or JSON format.

    Returns {"name": str, "arguments": dict} or None.
    """
    # Try Format A (XML-parameter) first — this is what the current model produces
    match_xml = _TOOL_CALL_XML_RE.search(text)
    if match_xml is not None:
        func_name = match_xml.group(1)
        params_block = match_xml.group(2)
        arguments: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(params_block):
            param_name = pm.group(1)
            param_value = pm.group(2).strip()
            # Attempt numeric coercion for integer-typed params
            try:
                param_value = int(param_value)
            except (ValueError, TypeError):
                pass
            arguments[param_name] = param_value
        if func_name and arguments:
            return {"name": func_name, "arguments": arguments}

    # Try Format B (JSON)
    match_json = _TOOL_CALL_JSON_RE.search(text)
    if match_json is not None:
        try:
            payload = json.loads(match_json.group(1))
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        name = payload.get("name")
        args = payload.get("arguments")
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        return {"name": name, "arguments": args}

    return None


# ── Helpers ──────────────────────────────────────────────────────────
def separator(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


def timed_generate(model, tokenizer, prompt: str, max_tokens: int = 1024) -> tuple[str, float]:
    """Run mlx_lm.generate and return (output_text, elapsed_seconds)."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.generate import make_sampler

    sampler = make_sampler(temp=0.3, top_p=0.8)
    t0 = time.perf_counter()
    output = mlx_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    elapsed = time.perf_counter() - t0
    return output, elapsed


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    # ----------------------------------------------------------------
    # Load model + tokenizer
    # ----------------------------------------------------------------
    separator("LOADING MODEL")
    print(f"Model: {MODEL_PATH}")
    t0 = time.perf_counter()
    from mlx_lm import load
    model, tokenizer = load(MODEL_PATH)
    load_time = time.perf_counter() - t0
    print(f"Loaded in {load_time:.1f}s")

    # ----------------------------------------------------------------
    # TEST 1: Does tools= work with apply_chat_template?
    # ----------------------------------------------------------------
    separator("TEST 1: apply_chat_template with tools= parameter")

    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is photosynthesis?"},
    ]
    tools = [SEARCH_TOOL_DEF]

    # Try with tools= kwarg
    tools_supported = False
    rendered_with_tools = None
    try:
        rendered_with_tools = tokenizer.apply_chat_template(
            test_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tools_supported = True
        print("✅ tools= parameter ACCEPTED (no TypeError)")
        print(f"\nRendered prompt ({len(rendered_with_tools)} chars):")
        print("-" * 60)
        print(rendered_with_tools[:3000])
        if len(rendered_with_tools) > 3000:
            print(f"\n... ({len(rendered_with_tools) - 3000} more chars)")
        print("-" * 60)

        # Check if the tool schema actually appears in the output
        if "search" in rendered_with_tools and "query" in rendered_with_tools:
            print("✅ Tool schema IS present in rendered template")
        else:
            print("⚠️  tools= was accepted but tool schema NOT found in output")
            print("    The template may have silently ignored the tools parameter")
    except TypeError as e:
        print(f"❌ tools= parameter REJECTED with TypeError: {e}")
        print("   → Native tool-call path is dead. Need prompt-engineering fallback.")
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")

    # Also try without enable_thinking (in case the combination fails)
    if not tools_supported:
        print("\nRetrying without enable_thinking...")
        try:
            rendered_with_tools = tokenizer.apply_chat_template(
                test_messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            tools_supported = True
            print("✅ tools= works WITHOUT enable_thinking")
            print(f"\nRendered prompt ({len(rendered_with_tools)} chars):")
            print("-" * 60)
            print(rendered_with_tools[:3000])
            if len(rendered_with_tools) > 3000:
                print(f"\n... ({len(rendered_with_tools) - 3000} more chars)")
            print("-" * 60)
        except Exception as e:
            print(f"❌ Still fails: {type(e).__name__}: {e}")

    if not tools_supported:
        print("\n🛑 BLOCKING: Cannot proceed with native tool-call path.")
        print("   The implementation plan needs revision for prompt-engineering fallback.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # TEST 2: Callback with WEAK context (should trigger tool call)
    # ----------------------------------------------------------------
    separator("TEST 2: Callback — weak context → model should call search tool")

    weak_context = (
        "[CHUNK 1 | SOURCE: battery_review | PAGE: 3]\n"
        "Lithium-ion batteries have become the dominant technology for energy storage.\n"
        "[CHUNK END]\n"
    )

    weak_messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer questions using ONLY the provided context passages. "
                "Cite sources using [N] notation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{weak_context}\n\n"
                "Question: What are the specific energy density improvements in solid-state "
                "batteries compared to lithium-ion, and what manufacturing challenges remain?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Based on the available context, lithium-ion batteries are the dominant energy "
                "storage technology [1]. However, the provided passages do not contain specific "
                "information about solid-state battery energy density comparisons or their "
                "manufacturing challenges. I cannot provide detailed figures on these topics "
                "from the available evidence."
            ),
        },
        {
            "role": "user",
            "content": CALLBACK_INSTRUCTION,
        },
    ]

    # Build prompt with tools
    callback_prompt_weak = tokenizer.apply_chat_template(
        weak_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print(f"Prompt length: {len(callback_prompt_weak)} chars")
    print("\nGenerating callback response (weak context)...")

    output_weak, time_weak = timed_generate(model, tokenizer, callback_prompt_weak, max_tokens=512)

    print(f"\n🕐 Generation time: {time_weak:.1f}s")
    print(f"\nRaw model output ({len(output_weak)} chars):")
    print("-" * 60)
    print(output_weak)
    print("-" * 60)

    # Parse tool call
    parsed = parse_tool_call(output_weak)
    if parsed is not None:
        print(f"\n✅ Tool call DETECTED and PARSED:")
        print(f"   Name: {parsed['name']}")
        print(f"   Arguments: {json.dumps(parsed['arguments'], indent=2)}")
        query_arg = parsed["arguments"].get("query")
        mode_arg = parsed["arguments"].get("mode", "(default)")
        max_r_arg = parsed["arguments"].get("max_results", "(default)")
        print(f"   → query={query_arg!r}, mode={mode_arg}, max_results={max_r_arg}")
    else:
        if "ANSWER_SUFFICIENT" in output_weak.upper():
            print("\n⚠️  Model declined (ANSWER_SUFFICIENT) even with weak context")
            print("   The callback instruction may need strengthening")
        else:
            print("\n❌ No parseable tool call AND no ANSWER_SUFFICIENT signal")
            print("   Raw output didn't match expected format")
            # Try to find any JSON-like structure
            json_matches = re.findall(r'\{[^}]+\}', output_weak)
            if json_matches:
                print(f"   Found {len(json_matches)} JSON-like block(s):")
                for i, m in enumerate(json_matches):
                    print(f"   [{i}] {m}")

    # ----------------------------------------------------------------
    # TEST 3: Callback with STRONG context (should decline)
    # ----------------------------------------------------------------
    separator("TEST 3: Callback — strong context → model should decline")

    strong_context = (
        "[CHUNK 1 | SOURCE: battery_review | PAGE: 3]\n"
        "Lithium-ion batteries have become the dominant technology for energy storage. "
        "Current lithium-ion cells achieve energy densities of 250-300 Wh/kg at the cell "
        "level, a significant improvement from the 90 Wh/kg of early commercial cells in 1991.\n"
        "[CHUNK END]\n\n"
        "[CHUNK 2 | SOURCE: battery_review | PAGE: 7]\n"
        "The global battery storage market reached 45 GWh of annual deployments in 2023, "
        "with projections suggesting 200 GWh by 2030. Cost reductions have been dramatic, "
        "falling from $1,100/kWh in 2010 to approximately $139/kWh in 2023.\n"
        "[CHUNK END]\n\n"
        "[CHUNK 3 | SOURCE: battery_review | PAGE: 12]\n"
        "Lithium iron phosphate (LFP) batteries have gained market share due to their "
        "superior thermal stability and lower cost, despite offering lower energy density "
        "(~170 Wh/kg) compared to nickel-manganese-cobalt (NMC) chemistries (~250 Wh/kg). "
        "LFP now represents over 40% of global EV battery production.\n"
        "[CHUNK END]\n"
    )

    strong_messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer questions using ONLY the provided context passages. "
                "Cite sources using [N] notation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{strong_context}\n\n"
                "Question: What has been the trajectory of lithium-ion battery costs and "
                "energy density improvements?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Lithium-ion battery technology has seen substantial improvements in both cost "
                "and energy density since its commercialization. Early cells in 1991 offered "
                "approximately 90 Wh/kg, while current cells achieve 250-300 Wh/kg at the "
                "cell level [1]. This represents roughly a threefold improvement in energy density.\n\n"
                "Cost reductions have been equally dramatic. Battery pack costs fell from "
                "$1,100/kWh in 2010 to approximately $139/kWh in 2023 — an 87% reduction [2]. "
                "This cost decline has been a key driver of the growing market, which reached "
                "45 GWh of annual deployments in 2023 with projections of 200 GWh by 2030 [2].\n\n"
                "Within the lithium-ion family, chemistry choices involve trade-offs. LFP "
                "batteries offer lower energy density (~170 Wh/kg) but superior thermal "
                "stability and lower cost, while NMC chemistries achieve higher densities "
                "(~250 Wh/kg). LFP now accounts for over 40% of global EV battery production [3]."
            ),
        },
        {
            "role": "user",
            "content": CALLBACK_INSTRUCTION,
        },
    ]

    callback_prompt_strong = tokenizer.apply_chat_template(
        strong_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print(f"Prompt length: {len(callback_prompt_strong)} chars")
    print("\nGenerating callback response (strong context)...")

    output_strong, time_strong = timed_generate(model, tokenizer, callback_prompt_strong, max_tokens=512)

    print(f"\n🕐 Generation time: {time_strong:.1f}s")
    print(f"\nRaw model output ({len(output_strong)} chars):")
    print("-" * 60)
    print(output_strong)
    print("-" * 60)

    parsed_strong = parse_tool_call(output_strong)
    if parsed_strong is not None:
        print(f"\n⚠️  Model called tool even with strong context:")
        print(f"   {json.dumps(parsed_strong, indent=2)}")
        print("   The callback instruction may need to be more conservative")
    elif "ANSWER_SUFFICIENT" in output_strong.upper():
        print("\n✅ Model correctly declined: ANSWER_SUFFICIENT detected")
    else:
        print("\n⚠️  No tool call and no ANSWER_SUFFICIENT signal")
        print("   Decline detection may need fuzzy matching or different signal token")

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    separator("SUMMARY")

    print(f"tools= parameter accepted:     {'YES' if tools_supported else 'NO'}")
    print(f"Tool schema in rendered prompt: {'YES' if (rendered_with_tools and 'search' in rendered_with_tools) else 'NO/UNKNOWN'}")
    print(f"Weak-context tool call parsed:  {'YES' if parsed is not None else 'NO'}")
    print(f"Strong-context decline signal:  {'YES' if 'ANSWER_SUFFICIENT' in output_strong.upper() else 'NO'}")
    print(f"Weak-context generation time:   {time_weak:.1f}s")
    print(f"Strong-context generation time: {time_strong:.1f}s")
    print(f"Model load time:                {load_time:.1f}s")

    # Overall gate verdict
    print()
    all_pass = (
        tools_supported
        and parsed is not None
        and "ANSWER_SUFFICIENT" in output_strong.upper()
    )
    if all_pass:
        print("✅ ALL GATES PASSED — proceed with implementation plan as-is")
    else:
        failures = []
        if not tools_supported:
            failures.append("tools= not supported → need prompt-engineering fallback")
        if parsed is None:
            failures.append("tool call not parseable → check model output format, adjust _parse_tool_call or instruction")
        if "ANSWER_SUFFICIENT" not in output_strong.upper():
            failures.append("decline signal not produced → adjust CALLBACK_INSTRUCTION wording")
        print("⚠️  SOME GATES FAILED — plan needs revision:")
        for f in failures:
            print(f"   • {f}")


if __name__ == "__main__":
    main()
