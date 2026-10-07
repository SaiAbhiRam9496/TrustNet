"""
llm_analyzer.py
---------------
Phase 1 & 3 | Gemini Flash LLM Client.

Responsibilities:
  - Provides a low-level `query_gemini_json()` helper used by all reasoning modules.
  - Provides `synthesize_evidence()` — Phase 3 Stage 5 synthesis that takes
    subclaims + NLI stance results and produces a structured verdict used by
    the Scoring Engine (Phase 4).
"""

import os
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.core.config import settings


# -----------------------------------------------------------------
# Synthesis system prompt for structured verdict output
# -----------------------------------------------------------------
_SYNTHESIS_SYSTEM_PROMPT = """
You are a professional fact-checking analyst producing a final structured verdict.
You will receive:
  1. The original claim.
  2. A list of atomic subclaims with their NLI stance results (supports/contradicts/neutral).
  3. The top supporting/contradicting evidence snippets.

Your tasks:
  - Summarize the overall factual standing of the claim in 2-3 sentences.
  - Identify which subclaims are well-supported, which are refuted, and which lack evidence.
  - Produce factor justifications for: evidence_strength, factual_consistency,
    source_provenance, bias_framing, and incentive_conflict.
  - Do NOT hardcode scores. Base all justifications purely on the evidence provided.

Respond ONLY in valid JSON with this exact structure:
{
  "summary": "<2-3 sentence plain English explanation of the verdict>",
  "supported_subclaims": ["<subclaim text>", ...],
  "refuted_subclaims": ["<subclaim text>", ...],
  "insufficient_subclaims": ["<subclaim text>", ...],
  "factor_justifications": {
    "evidence_strength":     "<justification>",
    "factual_consistency":   "<justification>",
    "source_provenance":     "<justification>",
    "bias_framing":          "<justification>",
    "incentive_conflict":    "<justification>"
  }
}
""".strip()


def get_llm_client() -> OpenAI:
    """
    Returns an OpenAI client configured for Google Gemini API via Google AI Studio's
    OpenAI-compatible endpoint.
    """
    api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in environment or .env file.")

    return OpenAI(
        api_key=api_key,
        base_url=settings.GEMINI_BASE_URL,
        timeout=settings.DEFAULT_LLM_TIMEOUT,
        max_retries=1,
    )


def query_gemini_json(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sends a structured prompt to Gemini API and expects a clean JSON response.

    Args:
        system_prompt: System-level instruction for the model.
        user_prompt:   The user-level content / data to analyze.
        model:         Optional model override (defaults to settings.LLM_MODEL).

    Returns:
        Parsed JSON dict from the model response.
    """
    client = get_llm_client()
    target_model = model or settings.LLM_MODEL

    response = client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    
    # Strip thinking tags <thought>...</thought> if present
    import re
    content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL).strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Failed to parse LLM response as JSON: {content}")


def synthesize_evidence(
    claim: str,
    subclaims_with_stances: List[Dict[str, Any]],
    top_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Phase 3 core synthesis call. Sends subclaim NLI results + evidence passages
    to Gemini Flash for structured verdict generation.

    Args:
        claim:
            The original user-inputted claim string.
        subclaims_with_stances:
            List of dicts, each containing:
              - "text"    : str  — the atomic subclaim
              - "stance"  : str  — "supports" | "contradicts" | "neutral"
              - "nli_conf": float — NLI model confidence score
        top_evidence:
            List of dicts (EvidenceItem-compatible) containing:
              - "snippet"              : str
              - "source_domain"        : str
              - "domain_authority_tier": str
              - "stance"               : str

    Returns:
        Dict with keys:
          - "summary"                 : str
          - "supported_subclaims"     : List[str]
          - "refuted_subclaims"       : List[str]
          - "insufficient_subclaims"  : List[str]
          - "factor_justifications"   : Dict[str, str]
    """
    # Build compact user prompt with all relevant data
    subclaim_lines = "\n".join(
        f"  [{i+1}] \"{sc['text']}\" → stance={sc.get('stance','?')}, nli_conf={sc.get('nli_conf', 0):.3f}"
        for i, sc in enumerate(subclaims_with_stances)
    )

    evidence_lines = "\n".join(
        f"  [{i+1}] [{ev.get('domain_authority_tier','?')} authority | {ev.get('stance','?')}] "
        f"{ev.get('source_domain','?')}: \"{ev.get('snippet','')[:200]}\""
        for i, ev in enumerate(top_evidence[:8])  # cap at 8 snippets to stay within token limits
    )

    user_prompt = (
        f"ORIGINAL CLAIM:\n\"{claim}\"\n\n"
        f"SUBCLAIM NLI RESULTS:\n{subclaim_lines}\n\n"
        f"TOP EVIDENCE PASSAGES:\n{evidence_lines}"
    )

    try:
        result = query_gemini_json(
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        if isinstance(result, list):
            result = result[0] if (len(result) > 0 and isinstance(result[0], dict)) else {}
    except Exception as e:
        raise ValueError(f"[LLMAnalyzer] Synthesis call failed: {e}") from e

    # Ensure all expected keys exist with safe defaults
    return {
        "summary": result.get("summary", "Synthesis unavailable."),
        "supported_subclaims": result.get("supported_subclaims", []),
        "refuted_subclaims": result.get("refuted_subclaims", []),
        "insufficient_subclaims": result.get("insufficient_subclaims", []),
        "factor_justifications": result.get("factor_justifications", {}),
    }
