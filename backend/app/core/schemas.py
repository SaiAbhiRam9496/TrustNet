from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Incoming request body for POST /analyze."""
    claim: str = Field(..., min_length=3, description="The textual claim or article excerpt to analyze.")


# Kept as alias for backwards compatibility
ClaimAnalysisRequest = AnalysisRequest


class EvidenceItem(BaseModel):
    title: str = Field(..., description="Title of the web source.")
    url: str = Field(..., description="URL link of the web source.")
    snippet: str = Field(..., description="Extracted text excerpt from the page.")
    source_domain: str = Field(..., description="Domain name (e.g. nytimes.com, wikipedia.org).")
    domain_authority_tier: str = Field(
        default="Medium",
        description="Quality tier: 'High' (.gov, .edu, top news), 'Medium', or 'Low' (unverified blogs)."
    )
    stance: str = Field(
        default="neutral",
        description="Direction: 'supports', 'contradicts', or 'neutral'."
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cross-encoder relevance score (0.0 to 1.0)."
    )


class FactorScore(BaseModel):
    score: float = Field(..., ge=0.0, le=100.0, description="Calculated score (0 to 100).")
    weight: float = Field(..., ge=0.0, le=1.0, description="Factor weight in overall score.")
    justification: str = Field(..., description="Human-readable explanation for this score.")


class FactorBreakdown(BaseModel):
    evidence_strength: FactorScore
    factual_consistency: FactorScore
    source_provenance: FactorScore
    bias_framing: FactorScore
    incentive_conflict: FactorScore


class AnalysisResponse(BaseModel):
    claim: str
    overall_score: float = Field(..., ge=0.0, le=100.0)
    verdict_label: str = Field(
        ...,
        description="'Supported', 'Refuted', 'Conflicting Evidence / Cherrypicking', 'Not Enough Evidence', or 'Insufficient Evidence'"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Empirical dynamic confidence score (0.0 to 1.0).")
    is_uncertain: bool = Field(default=False, description="True if Uncertainty Gate triggered abstention.")
    uncertainty_reason: Optional[str] = Field(default=None, description="Reason for abstention if triggered.")
    summary: Optional[str] = Field(default=None, description="Plain-English synthesis summary from Gemini.")
    factors: FactorBreakdown
    subclaims: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    pipeline_inspector: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-stage intermediate data for the Live Pipeline Inspector UI."
    )
