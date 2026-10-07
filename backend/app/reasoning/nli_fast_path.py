"""
nli_fast_path.py
----------------
Phase 3 | Stage 4 of the Live Pipeline Inspector.

Responsibilities:
  - Loads DeBERTa-v3-base NLI model locally (no API cost, offline capable).
  - Given a (premise, hypothesis) pair, classifies the stance as:
      * "supports"     — evidence supports the subclaim
      * "contradicts"  — evidence directly contradicts the subclaim
      * "neutral"      — evidence is unrelated / neither supports nor refutes

  - Returns the raw softmax probabilities alongside the label so that the
    Scoring Engine (Phase 4) can use P(NLI Softmax) in the dynamic
    confidence formula:

        Confidence = w1 * AgreementRatio
                   + w2 * DomainAuthorityScore
                   + w3 * P(NLI Softmax)

Implementation Notes:
  - Model: cross-encoder/nli-deberta-v3-base (HuggingFace Hub)
  - Framework: sentence-transformers CrossEncoder
  - The CrossEncoder returns raw logits for [contradiction, neutral, entailment].
    We map:  entailment   → "supports"
             contradiction → "contradicts"
             neutral      → "neutral"
  - Model is loaded once at module level (singleton) to avoid reloading on
    every API request.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Singleton model loader — loaded once, reused for every request
# -----------------------------------------------------------------
_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_nli_model: CrossEncoder | None = None


def _get_nli_model() -> CrossEncoder:
    """
    Lazy-loads the DeBERTa-v3 NLI model the first time it is called.
    Subsequent calls return the cached instance.
    """
    global _nli_model
    if _nli_model is None:
        logger.info("[NLI] Loading DeBERTa-v3 NLI model: %s", _MODEL_NAME)
        _nli_model = CrossEncoder(
            _MODEL_NAME,
            num_labels=3,
            device="cpu",  # CPU-safe — works on any machine without GPU
        )
        logger.info("[NLI] Model loaded successfully.")
    return _nli_model


# -----------------------------------------------------------------
# Label mapping — DeBERTa-v3 CrossEncoder logit order
# Label indices:  0 = contradiction, 1 = neutral, 2 = entailment
# -----------------------------------------------------------------
_IDX_TO_LABEL = {
    0: "contradicts",
    1: "neutral",
    2: "supports",
}


def classify_stance(
    premise: str,
    hypothesis: str,
) -> Dict[str, float | str]:
    """
    Classifies whether the *premise* (evidence passage) supports,
    contradicts, or is neutral toward the *hypothesis* (subclaim).

    Args:
        premise:    The retrieved evidence passage (what we know).
        hypothesis: The atomic subclaim we are trying to verify.

    Returns:
        A dict with:
          - "label"         : str  — "supports" | "contradicts" | "neutral"
          - "confidence"    : float — softmax probability of winning label (0–1)
          - "probs"         : dict  — full softmax distribution:
                                      {"supports": f, "contradicts": f, "neutral": f}
    """
    model = _get_nli_model()

    # CrossEncoder expects list of (text1, text2) pairs
    raw_logits = model.predict(
        [(premise, hypothesis)],
        apply_softmax=False,   # we apply softmax ourselves for full prob dist
        show_progress_bar=False,
    )[0]  # shape: [3]

    # Convert logits → softmax probabilities
    logit_tensor = torch.tensor(raw_logits, dtype=torch.float32)
    probs = torch.softmax(logit_tensor, dim=0).tolist()

    prob_dict: Dict[str, float] = {
        "contradicts": round(probs[0], 4),
        "neutral":     round(probs[1], 4),
        "supports":    round(probs[2], 4),
    }

    winning_idx = int(torch.argmax(logit_tensor).item())
    label = _IDX_TO_LABEL[winning_idx]
    confidence = round(probs[winning_idx], 4)

    return {
        "label":      label,
        "confidence": confidence,
        "probs":      prob_dict,
    }


def classify_batch(
    pairs: List[Tuple[str, str]]
) -> List[Dict[str, float | str]]:
    """
    Classifies stance for a batch of (premise, hypothesis) pairs efficiently.
    Useful when we have multiple evidence snippets to check against one subclaim.

    Args:
        pairs: List of (premise, hypothesis) tuples.

    Returns:
        List of classification dicts (same format as classify_stance).
    """
    if not pairs:
        return []

    model = _get_nli_model()

    raw_logits_batch = model.predict(
        pairs,
        apply_softmax=False,
        show_progress_bar=False,
    )

    results = []
    for raw_logits in raw_logits_batch:
        logit_tensor = torch.tensor(raw_logits, dtype=torch.float32)
        probs = torch.softmax(logit_tensor, dim=0).tolist()

        prob_dict: Dict[str, float] = {
            "contradicts": round(probs[0], 4),
            "neutral":     round(probs[1], 4),
            "supports":    round(probs[2], 4),
        }

        winning_idx = int(torch.argmax(logit_tensor).item())
        label = _IDX_TO_LABEL[winning_idx]
        confidence = round(probs[winning_idx], 4)

        results.append({
            "label":      label,
            "confidence": confidence,
            "probs":      prob_dict,
        })

    return results


def compute_agreement_ratio(stances: List[str]) -> float:
    """
    Computes the Agreement Ratio used in the Dynamic Confidence formula.

    Agreement Ratio = (dominant_stance_count) / (total_stances)

    If the dominant stance is "contradicts", the ratio is negated to indicate
    that sources are predominantly against the claim.

    Args:
        stances: List of stance labels ("supports", "contradicts", "neutral")
                 from all classified evidence passages.

    Returns:
        Float in range [-1.0, 1.0]:
          +1.0 = all sources support
          -1.0 = all sources contradict
           0.0 = perfectly split or all neutral
    """
    if not stances:
        return 0.0

    counts = {"supports": 0, "contradicts": 0, "neutral": 0}
    for s in stances:
        if s in counts:
            counts[s] += 1

    total = len(stances)
    support_ratio = counts["supports"] / total
    contradict_ratio = counts["contradicts"] / total

    if support_ratio >= contradict_ratio:
        return round(support_ratio, 4)
    else:
        return round(-contradict_ratio, 4)
