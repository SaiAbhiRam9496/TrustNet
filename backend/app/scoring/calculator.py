"""
calculator.py
-------------
Phase 4 | Scoring Engine — 5-Factor Credibility Score & Dynamic Confidence.

Implements the empirical scoring formula from the Architecture Plan (Section 1.B):

    Confidence = w1 * |AgreementRatio|
               + w2 * DomainAuthorityScore
               + w3 * AvgNLISoftmax

This module is the mathematical core of TrustNet. It:
  1. Converts raw Phase 3 output (subclaim stances, evidence) into 5 factor scores.
  2. Computes the overall credibility score (0–100).
  3. Computes the dynamic empirical confidence (0.0–1.0).
  4. Maps the score + confidence into a human-readable verdict label.

All scores are derived purely from empirical signals. NO hardcoded confidence values.

Factor Weights (sum = 1.0):
  evidence_strength    → 0.30  (How much relevant evidence exists?)
  factual_consistency  → 0.30  (Do sources agree with each other?)
  source_provenance    → 0.20  (How authoritative are the sources?)
  bias_framing         → 0.10  (Are low-credibility sources over-represented?)
  incentive_conflict   → 0.10  (Is source diversity high, reducing echo chambers?)

Dynamic Confidence Weights (sum = 1.0):
  w1 (AgreementRatio)       → 0.45
  w2 (DomainAuthorityScore) → 0.30
  w3 (AvgNLISoftmax)        → 0.25
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Domain Authority Tier → Numeric Score
# -----------------------------------------------------------------
_AUTHORITY_SCORE_MAP: Dict[str, float] = {
    "High":   1.0,
    "Medium": 0.55,
    "Low":    0.20,
}

# -----------------------------------------------------------------
# Factor weights — must sum to 1.0
# -----------------------------------------------------------------
_FACTOR_WEIGHTS = {
    "evidence_strength":   0.30,
    "factual_consistency": 0.30,
    "source_provenance":   0.20,
    "bias_framing":        0.10,
    "incentive_conflict":  0.10,
}

# -----------------------------------------------------------------
# Dynamic Confidence weights — must sum to 1.0
# -----------------------------------------------------------------
_CONFIDENCE_WEIGHTS = {
    "agreement":   0.45,
    "authority":   0.30,
    "nli_softmax": 0.25,
}


# =================================================================
# --- SIGNAL COMPUTERS ---
# =================================================================

def compute_domain_authority_score(evidence_list: List[Dict[str, Any]]) -> float:
    """
    Computes mean domain authority score across all evidence items.

    Args:
        evidence_list: List of EvidenceItem-compatible dicts.

    Returns:
        Float [0.0, 1.0] — higher = more authoritative source set.
    """
    if not evidence_list:
        return 0.0

    scores = []
    for ev in evidence_list:
        tier = ev.get("domain_authority_tier", "Medium")
        scores.append(_AUTHORITY_SCORE_MAP.get(tier, 0.55))

    return round(sum(scores) / len(scores), 4)


def compute_avg_nli_softmax(subclaims_with_stances: List[Dict[str, Any]]) -> float:
    """
    Computes average NLI model confidence (softmax probability of winning label)
    across all classified subclaims.

    Args:
        subclaims_with_stances: List of dicts containing "nli_conf" key.

    Returns:
        Float [0.0, 1.0] — higher = model was more certain in its stance calls.
    """
    confs = [sc.get("nli_conf", 0.0) for sc in subclaims_with_stances if sc.get("nli_conf") is not None]
    if not confs:
        return 0.0
    return round(sum(confs) / len(confs), 4)


# =================================================================
# --- 5-FACTOR SCORERS ---
# =================================================================

def _score_evidence_strength(
    evidence_list: List[Dict[str, Any]],
    subclaims_with_stances: List[Dict[str, Any]],
) -> float:
    """
    Evidence Strength (0–100):
    Measures quantity × quality of relevant evidence.
    - Based on number of non-neutral evidence snippets.
    - Weighted by average cross-encoder relevance score.
    """
    if not evidence_list:
        return 0.0

    relevant = [ev for ev in evidence_list if ev.get("stance") in ("supports", "contradicts")]
    relevance_scores = [ev.get("relevance_score", 0.5) for ev in evidence_list]
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    # Quantity component: sigmoid-style curve capped at 10 sources = 100%
    quantity_ratio = min(len(relevant) / 10.0, 1.0)

    raw = (0.6 * quantity_ratio + 0.4 * avg_relevance) * 100
    return round(min(raw, 100.0), 2)


def _score_factual_consistency(
    subclaims_with_stances: List[Dict[str, Any]],
    agreement_ratio: float,
) -> float:
    """
    Factual Consistency (0–100):
    Measures how unanimously sources agree or disagree.
    High score = clear consensus (either strongly for OR strongly against).
    Low score = split/conflicting evidence.
    """
    if not subclaims_with_stances:
        return 0.0

    # |agreement_ratio| tells us how decisive the stance is — 1.0 = unanimous
    consistency = abs(agreement_ratio) * 100
    return round(min(consistency, 100.0), 2)


def _score_source_provenance(evidence_list: List[Dict[str, Any]]) -> float:
    """
    Source Provenance (0–100):
    Based purely on domain authority tier distribution.
    """
    return round(compute_domain_authority_score(evidence_list) * 100, 2)


def _score_bias_framing(evidence_list: List[Dict[str, Any]]) -> float:
    """
    Bias & Framing (0–100):
    Penalizes heavy presence of Low-authority sources.
    Score = 100 - (fraction_of_low_authority_sources * 100)
    """
    if not evidence_list:
        return 0.0

    low_count = sum(1 for ev in evidence_list if ev.get("domain_authority_tier") == "Low")
    low_fraction = low_count / len(evidence_list)
    score = (1.0 - low_fraction) * 100
    return round(score, 2)


def _score_incentive_conflict(evidence_list: List[Dict[str, Any]]) -> float:
    """
    Incentive Conflict (0–100):
    Measures source domain *diversity* — many independent sources = low conflict.
    Score = unique_domains / total_evidence, scaled to 0-100.
    """
    if not evidence_list:
        return 0.0

    domains = {ev.get("source_domain", "unknown") for ev in evidence_list}
    diversity_ratio = min(len(domains) / max(len(evidence_list), 1), 1.0)
    return round(diversity_ratio * 100, 2)


# =================================================================
# --- FACTOR BREAKDOWN BUILDER ---
# =================================================================

def compute_5_factor_breakdown(
    subclaims_with_stances: List[Dict[str, Any]],
    evidence_list: List[Dict[str, Any]],
    agreement_ratio: float,
    factor_justifications: Dict[str, str],
) -> Dict[str, Any]:
    """
    Builds the full 5-Factor score breakdown.

    Args:
        subclaims_with_stances: Phase 3 NLI output.
        evidence_list:          Retrieved evidence with stances and authority tiers.
        agreement_ratio:        From nli_fast_path.compute_agreement_ratio().
        factor_justifications:  LLM-generated text justifications from synthesize_evidence().

    Returns:
        Dict matching the FactorBreakdown Pydantic schema.
    """
    es  = _score_evidence_strength(evidence_list, subclaims_with_stances)
    fc  = _score_factual_consistency(subclaims_with_stances, agreement_ratio)
    sp  = _score_source_provenance(evidence_list)
    bf  = _score_bias_framing(evidence_list)
    ic  = _score_incentive_conflict(evidence_list)

    def _factor(score: float, key: str, fallback: str) -> Dict[str, Any]:
        return {
            "score":         score,
            "weight":        _FACTOR_WEIGHTS[key],
            "justification": factor_justifications.get(key, fallback),
        }

    return {
        "evidence_strength": _factor(
            es, "evidence_strength",
            f"Found {len(evidence_list)} evidence items; {sum(1 for e in evidence_list if e.get('stance') != 'neutral')} are relevant to the claim."
        ),
        "factual_consistency": _factor(
            fc, "factual_consistency",
            f"Source agreement ratio is {agreement_ratio:.2f}; {'high' if abs(agreement_ratio) > 0.6 else 'moderate' if abs(agreement_ratio) > 0.3 else 'low'} consistency."
        ),
        "source_provenance": _factor(
            sp, "source_provenance",
            f"Average domain authority score: {sp:.1f}/100."
        ),
        "bias_framing": _factor(
            bf, "bias_framing",
            f"Low-authority source fraction: {100 - bf:.1f}%."
        ),
        "incentive_conflict": _factor(
            ic, "incentive_conflict",
            f"{len({e.get('source_domain') for e in evidence_list})} unique domains across {len(evidence_list)} sources."
        ),
    }


# =================================================================
# --- OVERALL SCORE ---
# =================================================================

def compute_overall_score(factor_breakdown: Dict[str, Any]) -> float:
    """
    Weighted average of 5-factor scores.

    Returns:
        Float [0.0, 100.0]
    """
    total = 0.0
    for key, weight in _FACTOR_WEIGHTS.items():
        factor = factor_breakdown.get(key, {})
        total += factor.get("score", 0.0) * weight
    return round(min(total, 100.0), 2)


# =================================================================
# --- DYNAMIC EMPIRICAL CONFIDENCE ---
# =================================================================

def compute_dynamic_confidence(
    agreement_ratio: float,
    domain_authority_score: float,
    avg_nli_softmax: float,
) -> float:
    """
    Implements the Architecture Plan confidence formula:

        Confidence = w1 * |AgreementRatio|
                   + w2 * DomainAuthorityScore
                   + w3 * AvgNLISoftmax

    NOTE: |AgreementRatio| measures decisiveness regardless of direction.
    The actual DIRECTION (support vs. contradict) is captured in verdict_label.

    Returns:
        Float [0.0, 1.0]
    """
    w1 = _CONFIDENCE_WEIGHTS["agreement"]
    w2 = _CONFIDENCE_WEIGHTS["authority"]
    w3 = _CONFIDENCE_WEIGHTS["nli_softmax"]

    confidence = (
        w1 * abs(agreement_ratio)
        + w2 * domain_authority_score
        + w3 * avg_nli_softmax
    )
    return round(min(max(confidence, 0.0), 1.0), 4)


# =================================================================
# --- VERDICT LABEL MAPPER ---
# =================================================================

def determine_verdict_label(
    overall_score: float,
    agreement_ratio: float,
    confidence: float,
    is_uncertain: bool,
) -> str:
    """
    Maps the empirical scores to an AVeriTeC-compatible verdict label.

    Label hierarchy (highest priority first):
      1. "Insufficient Evidence"            — uncertainty gate triggered
      2. "Conflicting Evidence / Cherrypicking" — split signals, high confidence
      3. "Supported"                         — strong positive evidence
      4. "Refuted"                           — strong negative evidence
      5. "Not Enough Evidence"               — weak overall score

    Args:
        overall_score:   Float [0, 100] — weighted 5-factor score.
        agreement_ratio: Float [-1, 1]  — signed stance direction.
        confidence:      Float [0, 1]   — dynamic empirical confidence.
        is_uncertain:    Bool           — True if uncertainty gate fired.

    Returns:
        str — verdict label string.
    """
    if is_uncertain:
        return "Insufficient Evidence"

    # Conflicting: split sources with reasonable confidence
    if -0.35 < agreement_ratio < 0.35 and confidence >= 0.45 and overall_score >= 45:
        return "Conflicting Evidence / Cherrypicking"

    # Strong support
    if agreement_ratio >= 0.35 and overall_score >= 60:
        return "Supported"

    # Strong refutation
    if agreement_ratio <= -0.35 and overall_score >= 60:
        return "Refuted"

    # Weak evidence — not enough to decide
    return "Not Enough Evidence"


# =================================================================
# --- MAIN SCORING ORCHESTRATOR ---
# =================================================================

def run_scoring(
    claim: str,
    subclaims_with_stances: List[Dict[str, Any]],
    evidence_list: List[Dict[str, Any]],
    synthesis_result: Dict[str, Any],
    agreement_ratio: float,
    is_uncertain: bool = False,
    uncertainty_reason: str | None = None,
) -> Dict[str, Any]:
    """
    Top-level scoring orchestrator. Called by routes.py after Phase 2 & 3 complete.

    Args:
        claim:                   Original user claim string.
        subclaims_with_stances:  Phase 3 NLI classification output.
        evidence_list:           Reranked evidence from Phase 2 (with stance + authority).
        synthesis_result:        Phase 3 llm_analyzer.synthesize_evidence() output.
        agreement_ratio:         From nli_fast_path.compute_agreement_ratio().
        is_uncertain:            Whether uncertainty gate was triggered.
        uncertainty_reason:      Reason string if gate triggered.

    """
    if not isinstance(synthesis_result, dict):
        if isinstance(synthesis_result, list) and len(synthesis_result) > 0 and isinstance(synthesis_result[0], dict):
            synthesis_result = synthesis_result[0]
        else:
            synthesis_result = {}

    factor_justifications = synthesis_result.get("factor_justifications", {})

    # Compute all signals
    domain_authority_score = compute_domain_authority_score(evidence_list)
    avg_nli_softmax = compute_avg_nli_softmax(subclaims_with_stances)

    # Build factor breakdown
    factor_breakdown = compute_5_factor_breakdown(
        subclaims_with_stances=subclaims_with_stances,
        evidence_list=evidence_list,
        agreement_ratio=agreement_ratio,
        factor_justifications=factor_justifications,
    )

    # Compute composite scores
    overall_score = compute_overall_score(factor_breakdown)
    confidence = compute_dynamic_confidence(
        agreement_ratio=agreement_ratio,
        domain_authority_score=domain_authority_score,
        avg_nli_softmax=avg_nli_softmax,
    )

    verdict_label = determine_verdict_label(
        overall_score=overall_score,
        agreement_ratio=agreement_ratio,
        confidence=confidence,
        is_uncertain=is_uncertain,
    )

    logger.info(
        "[Scoring] claim=%r | score=%.1f | confidence=%.3f | verdict=%s",
        claim[:60], overall_score, confidence, verdict_label
    )

    # Flatten subclaim texts for AnalysisResponse.subclaims field
    subclaim_texts = [sc.get("text", "") for sc in subclaims_with_stances]

    return {
        "claim":              claim,
        "overall_score":      overall_score,
        "verdict_label":      verdict_label,
        "confidence":         confidence,
        "is_uncertain":       is_uncertain,
        "uncertainty_reason": uncertainty_reason,
        "factors":            factor_breakdown,
        "subclaims":          subclaim_texts,
        "evidence":           evidence_list,
        "summary":            synthesis_result.get("summary", ""),
        # --- pipeline inspector data (for Live UI) ---
        "pipeline_data": {
            "agreement_ratio":        round(agreement_ratio, 4),
            "domain_authority_score": round(domain_authority_score, 4),
            "avg_nli_softmax":        round(avg_nli_softmax, 4),
            "supported_subclaims":    synthesis_result.get("supported_subclaims", []),
            "refuted_subclaims":      synthesis_result.get("refuted_subclaims", []),
            "insufficient_subclaims": synthesis_result.get("insufficient_subclaims", []),
        },
    }
