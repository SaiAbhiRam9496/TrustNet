"""
uncertainty_gate.py
-------------------
Phase 4 | Abstention Logic — "Insufficient Evidence" Guard.

This module implements the uncertainty gate that prevents TrustNet from
producing overconfident verdicts when the evidence is too thin to support
any meaningful conclusion.

The gate fires (is_uncertain = True) when ANY of the following conditions
are met:

  1. CONFIDENCE TOO LOW:
     Dynamic confidence < UNCERTAINTY_CONFIDENCE_THRESHOLD (default 0.40)
     → The three-signal confidence formula couldn't build a reliable estimate.

  2. TOO FEW SOURCES:
     Total retrieved evidence items < MIN_REQUIRED_SOURCES (default 2)
     → Not enough independent data points to cross-verify.

  3. ALL STANCES NEUTRAL:
     Every single NLI classification returned "neutral"
     → No evidence snippet took a clear pro/con position on the claim.

  4. NO SUBCLAIMS RESOLVED:
     Zero subclaims received a "supports" or "contradicts" classification
     → The decomposer produced subclaims but NLI couldn't resolve any.

When the gate fires, the scoring engine is instructed to return the
verdict label "Insufficient Evidence" instead of a directional verdict.
This maps directly to the AVeriTeC "Not Enough Info" class.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def check_uncertainty(
    confidence: float,
    evidence_list: List[Dict[str, Any]],
    subclaims_with_stances: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates whether the evidence base is strong enough to support a verdict.

    Args:
        confidence:              Dynamic empirical confidence score [0.0, 1.0].
        evidence_list:           Retrieved and reranked evidence items.
        subclaims_with_stances:  Phase 3 NLI-classified subclaims.

    Returns:
        Dict with:
          - "is_uncertain"  : bool            — True if gate fires.
          - "reason"        : Optional[str]   — Human-readable reason if gate fires.
          - "gate_checks"   : Dict[str, bool] — Individual check results (for UI inspector).
    """
    gate_checks: Dict[str, bool] = {}
    triggered_reasons: List[str] = []

    # --- Check 1: Confidence threshold ---
    low_confidence = confidence < settings.UNCERTAINTY_CONFIDENCE_THRESHOLD
    gate_checks["low_confidence"] = low_confidence
    if low_confidence:
        triggered_reasons.append(
            f"Dynamic confidence ({confidence:.3f}) is below the minimum threshold "
            f"({settings.UNCERTAINTY_CONFIDENCE_THRESHOLD})."
        )

    # --- Check 2: Minimum source count ---
    source_count = len(evidence_list)
    insufficient_sources = source_count < settings.MIN_REQUIRED_SOURCES
    gate_checks["insufficient_sources"] = insufficient_sources
    if insufficient_sources:
        triggered_reasons.append(
            f"Only {source_count} evidence source(s) retrieved; "
            f"minimum required is {settings.MIN_REQUIRED_SOURCES}."
        )

    # --- Check 3: All stances neutral ---
    stances = [sc.get("stance", "neutral") for sc in subclaims_with_stances]
    all_neutral = bool(stances) and all(s == "neutral" for s in stances)
    gate_checks["all_stances_neutral"] = all_neutral
    if all_neutral:
        triggered_reasons.append(
            "All NLI classifications returned 'neutral' — "
            "no evidence snippet took a clear stance on the claim."
        )

    # --- Check 4: No resolved subclaims ---
    resolved = [sc for sc in subclaims_with_stances if sc.get("stance") in ("supports", "contradicts")]
    no_resolved = bool(subclaims_with_stances) and len(resolved) == 0
    gate_checks["no_resolved_subclaims"] = no_resolved
    if no_resolved:
        triggered_reasons.append(
            f"{len(subclaims_with_stances)} subclaim(s) generated but none received a "
            "directional NLI classification (supports / contradicts)."
        )

    is_uncertain = bool(triggered_reasons)

    if is_uncertain:
        combined_reason = " | ".join(triggered_reasons)
        logger.warning("[UncertaintyGate] FIRED — %s", combined_reason)
    else:
        combined_reason = None
        logger.debug("[UncertaintyGate] PASSED — confidence=%.3f, sources=%d", confidence, source_count)

    return {
        "is_uncertain": is_uncertain,
        "reason":       combined_reason,
        "gate_checks":  gate_checks,
    }


def should_abstain(gate_result: Dict[str, Any]) -> bool:
    """
    Convenience helper — returns True if the gate fired and the system should
    return "Insufficient Evidence" instead of a directional verdict.

    Args:
        gate_result: Return value of check_uncertainty().

    Returns:
        bool
    """
    return gate_result.get("is_uncertain", False)
