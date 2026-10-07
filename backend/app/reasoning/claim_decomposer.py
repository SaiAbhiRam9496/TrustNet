"""
claim_decomposer.py
-------------------
Phase 3 | Stage 1 of the Live Pipeline Inspector.

Responsibilities:
  1. Takes a raw user-inputted claim string.
  2. Uses Gemini Flash to decompose it into a list of atomic subclaims
     (single, independently verifiable factual assertions).
  3. Generates optimized Tavily search queries for each subclaim.

The output is fed directly into the retrieval engine (search_provider.py)
so that each subclaim gets its own targeted evidence set.
"""

from typing import List
from app.reasoning.llm_analyzer import query_gemini_json


# -----------------------------------------------------------------
# System Prompt: Strict decomposition rules for reliable JSON output
# -----------------------------------------------------------------
_DECOMPOSE_SYSTEM_PROMPT = """
You are a professional fact-checking analyst. Your only job is to decompose
complex or compound claims into atomic subclaims and generate precise web
search queries for each subclaim.

Rules:
- An "atomic subclaim" is a single, independently verifiable factual statement.
- Do NOT add opinions, caveats, or qualifications — only factual assertions.
- Generate 1-3 focused search queries per subclaim (short, keyword-rich).
- Respond ONLY in valid JSON with this exact structure:

{
  "subclaims": [
    {
      "text": "<atomic subclaim text>",
      "search_queries": ["<query 1>", "<query 2>"]
    }
  ]
}
""".strip()


def decompose_claim(claim: str) -> List[dict]:
    """
    Decomposes a user claim into atomic subclaims with search queries.

    Args:
        claim: The raw claim text entered by the user.

    Returns:
        A list of dicts, each containing:
          - "text"           : str — the atomic subclaim text
          - "search_queries" : List[str] — optimised search queries

    Raises:
        ValueError: If Gemini does not return parseable JSON or is misconfigured.
    """
    user_prompt = f'Decompose the following claim:\n\n"{claim}"'

    try:
        result = query_gemini_json(
            system_prompt=_DECOMPOSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        raise ValueError(f"[ClaimDecomposer] LLM call failed: {e}") from e

    subclaims = result.get("subclaims", [])

    # Validate and sanitize
    cleaned = []
    for item in subclaims:
        text = item.get("text", "").strip()
        queries = item.get("search_queries", [])
        if not text:
            continue
        cleaned.append({
            "text": text,
            "search_queries": [q.strip() for q in queries if q.strip()],
        })

    # If LLM returns nothing useful, fall back to treating the whole claim as one subclaim
    if not cleaned:
        cleaned = [{
            "text": claim.strip(),
            "search_queries": [claim.strip()],
        }]

    return cleaned


def get_all_search_queries(subclaims: List[dict]) -> List[str]:
    """
    Flattens all search queries across all subclaims into a single deduplicated list.
    Useful for batching Tavily API calls.

    Args:
        subclaims: Output of decompose_claim().

    Returns:
        A flat deduplicated list of search query strings.
    """
    seen = set()
    queries = []
    for sc in subclaims:
        for q in sc.get("search_queries", []):
            if q not in seen:
                seen.add(q)
                queries.append(q)
    return queries
