"""
routes.py
---------
Phase 5 | REST API Endpoints.

Endpoints:
  GET  /health  — Server health + model readiness check.
  POST /analyze — Full 5-stage pipeline: Decompose → Search → Rank → NLI → Score.

Pipeline Stages (maps directly to Live Pipeline Inspector UI stages):
  Stage 1: Claim decomposition into atomic subclaims + search queries (Gemini).
  Stage 2: Tavily web search + Trafilatura text extraction.
  Stage 3: BM25 + Dense (RRF) Hybrid Ranking + Cross-Encoder Reranking.
  Stage 4: Local DeBERTa-v3 NLI stance classification per subclaim.
  Stage 5: Dynamic Confidence scoring + Gemini synthesis verdict.

All CPU-bound ML operations (HybridIndex, CrossEncoder, NLI) run inside
`run_in_threadpool` to avoid blocking FastAPI's async event loop.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.core.schemas import AnalysisRequest, AnalysisResponse, EvidenceItem, FactorBreakdown, FactorScore
from app.core.exceptions import LLMError, RetrievalError, ScoringError

from app.reasoning.claim_decomposer import decompose_claim, get_all_search_queries
from app.reasoning.nli_fast_path import classify_batch, compute_agreement_ratio
from app.reasoning.llm_analyzer import synthesize_evidence

from app.retrieval.search_provider import TavilySearchProvider
from app.retrieval.scraper import extract_article_content
from app.retrieval.hybrid_index import HybridIndex
from app.retrieval.reranker import CrossEncoderReranker

from app.scoring.calculator import (
    compute_domain_authority_score,
    compute_avg_nli_softmax,
    compute_dynamic_confidence,
    compute_5_factor_breakdown,
    compute_overall_score,
    determine_verdict_label,
    run_scoring,
)
from app.scoring.uncertainty_gate import check_uncertainty

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Singletons loaded once at import time ---
_search_provider = TavilySearchProvider()
_reranker = CrossEncoderReranker()


# ================================================================
# GET /health
# ================================================================

@router.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Returns server health status and basic configuration info.
    Used by CI/CD and the frontend to confirm the backend is alive.
    """
    return {
        "status":  "ok",
        "service": "TrustNet Fact-Checking API",
        "version": "1.0.0-phase5",
    }


# ================================================================
# POST /analyze
# ================================================================

@router.post("/analyze", tags=["Analysis"])
async def analyze_claim(request: AnalysisRequest) -> Dict[str, Any]:
    """
    Full 5-Stage TrustNet fact-checking pipeline.

    Accepts a claim string and returns:
      - A credibility verdict (Supported / Refuted / Conflicting / Insufficient Evidence)
      - A dynamic empirical confidence score (0.0 – 1.0)
      - 5-factor credibility breakdown
      - Full pipeline inspector data for the Live UI demo
    """
    claim = request.claim.strip()
    logger.info("[/analyze] Received claim: %r", claim[:80])

    pipeline_stages: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Claim Decomposition
    # ─────────────────────────────────────────────────────────────
    try:
        subclaims = await run_in_threadpool(decompose_claim, claim)
        search_queries = get_all_search_queries(subclaims)
        pipeline_stages["stage1_decomposition"] = {
            "subclaims":      [sc["text"] for sc in subclaims],
            "search_queries": search_queries,
        }
        logger.info("[Stage 1] %d subclaims, %d queries", len(subclaims), len(search_queries))
    except Exception as e:
        raise LLMError(f"Claim decomposition failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: Web Search + Scraping
    # ─────────────────────────────────────────────────────────────
    raw_evidence: List[Dict[str, Any]] = []

    # Use up to 3 queries to balance coverage vs. latency
    for query in search_queries[:3]:
        try:
            results = await run_in_threadpool(_search_provider.search, query, 5)
        except Exception:
            results = []

        for r in results:
            url = r.get("url", "")
            if not url:
                continue
            try:
                scraped = await run_in_threadpool(extract_article_content, url, None, r.get("snippet", ""))
            except Exception:
                scraped = {"domain": "", "authority_tier": "Low", "title": "", "text": ""}

            snippet = scraped.get("text", "").strip()[:800] or r.get("snippet", "")
            if not snippet:
                continue

            raw_evidence.append({
                "title":                r.get("title", scraped.get("title", "")),
                "url":                  url,
                "snippet":              snippet,
                "source_domain":        scraped.get("domain", ""),
                "domain_authority_tier": scraped.get("authority_tier", "Low"),
                "stance":               "neutral",
                "relevance_score":      0.5,
            })

    pipeline_stages["stage2_retrieval"] = {
        "total_raw_sources": len(raw_evidence),
        "urls": [ev["url"] for ev in raw_evidence],
    }
    logger.info("[Stage 2] %d raw evidence items scraped", len(raw_evidence))

    if not raw_evidence:
        raise RetrievalError("No evidence could be retrieved for the given claim.")

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: Hybrid BM25+Dense Ranking + Cross-Encoder Reranking
    # ─────────────────────────────────────────────────────────────
    try:
        index = HybridIndex()
        ranked = await run_in_threadpool(index.rank_passages, claim, raw_evidence)
        top_passages = ranked[:10]
        reranked = await run_in_threadpool(_reranker.rerank, claim, top_passages, 8)
    except Exception as e:
        # Fall back to raw evidence if ranking fails
        logger.warning("[Stage 3] Ranking failed: %s — using raw evidence", e)
        reranked = raw_evidence[:8]

    pipeline_stages["stage3_ranking"] = {
        "passages_after_ranking":  len(ranked) if "ranked" in dir() else len(raw_evidence),
        "passages_after_reranking": len(reranked),
        "top_passages": [
            {
                "url":             ev.get("url", ""),
                "source_domain":   ev.get("source_domain", ""),
                "relevance_score": round(ev.get("relevance_score", 0.0), 4),
                "snippet_preview": ev.get("snippet", "")[:120],
            }
            for ev in reranked
        ],
    }
    logger.info("[Stage 3] %d passages after reranking", len(reranked))

    # ─────────────────────────────────────────────────────────────
    # STAGE 4: NLI Stance Classification
    # ─────────────────────────────────────────────────────────────

    # Step A: Classify each evidence snippet against the ORIGINAL CLAIM
    # This gives us the per-evidence stance used in scoring
    evidence_nli_pairs = [
        (ev.get("snippet", ""), claim)
        for ev in reranked
        if ev.get("snippet")
    ]
    try:
        evidence_nli_results = await run_in_threadpool(classify_batch, evidence_nli_pairs)
        for i, ev in enumerate(reranked):
            if i < len(evidence_nli_results):
                ev["stance"] = evidence_nli_results[i]["label"]
    except Exception as e:
        logger.warning("[Stage 4] Evidence NLI failed: %s", e)

    # Step B: Classify each subclaim (majority vote across top evidence snippets)
    subclaims_with_stances: List[Dict[str, Any]] = []
    sc_nli_debug: List[Dict] = []

    for sc in subclaims:
        sc_pairs = [
            (ev.get("snippet", ""), sc["text"])
            for ev in reranked[:5]
            if ev.get("snippet")
        ]
        if sc_pairs:
            try:
                sc_results = await run_in_threadpool(classify_batch, sc_pairs)
                stances = [r["label"]      for r in sc_results]
                confs   = [r["confidence"] for r in sc_results]
                majority_stance = Counter(stances).most_common(1)[0][0]
                avg_conf = round(sum(confs) / len(confs), 4)
            except Exception:
                majority_stance, avg_conf = "neutral", 0.0
        else:
            majority_stance, avg_conf = "neutral", 0.0

        subclaims_with_stances.append({
            "text":     sc["text"],
            "stance":   majority_stance,
            "nli_conf": avg_conf,
        })
        sc_nli_debug.append({
            "subclaim": sc["text"],
            "stance":   majority_stance,
            "nli_conf": avg_conf,
        })

    pipeline_stages["stage4_nli"] = {
        "subclaim_stances": sc_nli_debug,
        "evidence_stances": [
            {"url": ev.get("url", ""), "stance": ev.get("stance", "neutral")}
            for ev in reranked
        ],
    }
    logger.info("[Stage 4] Subclaim stances: %s", [s["stance"] for s in subclaims_with_stances])

    # ─────────────────────────────────────────────────────────────
    # STAGE 5: Scoring + Synthesis
    # ─────────────────────────────────────────────────────────────
    try:
        all_stances      = [sc["stance"] for sc in subclaims_with_stances]
        agreement_ratio  = compute_agreement_ratio(all_stances)
        domain_auth      = compute_domain_authority_score(reranked)
        avg_nli_softmax  = compute_avg_nli_softmax(subclaims_with_stances)
        prelim_confidence = compute_dynamic_confidence(agreement_ratio, domain_auth, avg_nli_softmax)

        gate = check_uncertainty(prelim_confidence, reranked, subclaims_with_stances)

        synthesis = await run_in_threadpool(
            synthesize_evidence,
            claim,
            subclaims_with_stances,
            reranked,
        )

        result = run_scoring(
            claim=claim,
            subclaims_with_stances=subclaims_with_stances,
            evidence_list=reranked,
            synthesis_result=synthesis,
            agreement_ratio=agreement_ratio,
            is_uncertain=gate["is_uncertain"],
            uncertainty_reason=gate["reason"],
        )
    except Exception as e:
        raise ScoringError(f"Scoring engine failed: {e}")

    pipeline_stages["stage5_scoring"] = {
        "agreement_ratio":        round(agreement_ratio, 4),
        "domain_authority_score": round(domain_auth, 4),
        "avg_nli_softmax":        round(avg_nli_softmax, 4),
        "dynamic_confidence":     result["confidence"],
        "overall_score":          result["overall_score"],
        "verdict":                result["verdict_label"],
        "uncertainty_gate":       gate["gate_checks"],
    }
    logger.info(
        "[Stage 5] score=%.1f | conf=%.3f | verdict=%s",
        result["overall_score"], result["confidence"], result["verdict_label"]
    )

    # ─────────────────────────────────────────────────────────────
    # Build final response
    # ─────────────────────────────────────────────────────────────
    return {
        # Core verdict fields
        "claim":              result["claim"],
        "overall_score":      result["overall_score"],
        "verdict_label":      result["verdict_label"],
        "confidence":         result["confidence"],
        "is_uncertain":       result["is_uncertain"],
        "uncertainty_reason": result["uncertainty_reason"],
        "summary":            result["summary"],

        # Factor breakdown
        "factors": result["factors"],

        # Subclaims & evidence
        "subclaims": result["subclaims"],
        "evidence":  reranked,

        # Full pipeline inspector — every stage's intermediate data
        "pipeline_inspector": pipeline_stages,
    }


# ================================================================
# POST /analyze/stream
# ================================================================

@router.post("/analyze/stream", tags=["Analysis"])
async def analyze_claim_stream(request: AnalysisRequest):
    """
    Server-Sent Events (SSE) streaming endpoint for the 5-Stage Pipeline.

    Yields JSON data events live after each stage completes:
      - Event "stage1": Subclaims & search queries
      - Event "stage2": Raw retrieved URLs
      - Event "stage3": Reranked top passages
      - Event "stage4": DeBERTa NLI stances
      - Event "stage5": Final score, verdict, summary & full result
    """
    async def event_generator():
        claim = request.claim.strip()
        logger.info("[/analyze/stream] Received claim: %r", claim[:80])
        pipeline_stages: Dict[str, Any] = {}

        # ─────────────────────────────────────────────────────────────
        # STAGE 1: Decomposition
        # ─────────────────────────────────────────────────────────────
        try:
            subclaims = await run_in_threadpool(decompose_claim, claim)
            search_queries = get_all_search_queries(subclaims)
            pipeline_stages["stage1_decomposition"] = {
                "subclaims":      [sc["text"] for sc in subclaims],
                "search_queries": search_queries,
            }
            logger.info("[Stream Stage 1] %d subclaims, %d queries", len(subclaims), len(search_queries))
            payload = {
                "stage": "stage1",
                "status": "running",
                "message": f"Stage 1 Complete: {len(subclaims)} subclaims generated.",
                "pipeline_inspector": pipeline_stages,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            logger.error("[Stream Stage 1 Error] %s", e)
            yield f"data: {json.dumps({'status': 'error', 'error': f'Claim decomposition failed: {e}'})}\n\n"
            return

        # ─────────────────────────────────────────────────────────────
        # STAGE 2: Web Search & Scraping
        # ─────────────────────────────────────────────────────────────
        raw_evidence: List[Dict[str, Any]] = []
        try:
            for query in search_queries[:3]:
                try:
                    results = await run_in_threadpool(_search_provider.search, query, 5)
                except Exception:
                    results = []

                for r in results:
                    url = r.get("url", "")
                    if not url:
                        continue
                    try:
                        scraped = await run_in_threadpool(extract_article_content, url, None, r.get("snippet", ""))
                    except Exception:
                        scraped = {"domain": "", "authority_tier": "Low", "title": "", "text": ""}

                    snippet = scraped.get("text", "").strip()[:800] or r.get("snippet", "")
                    if not snippet:
                        continue

                    raw_evidence.append({
                        "title":                r.get("title", scraped.get("title", "")),
                        "url":                  url,
                        "snippet":              snippet,
                        "source_domain":        scraped.get("domain", ""),
                        "domain_authority_tier": scraped.get("authority_tier", "Low"),
                        "stance":               "neutral",
                        "relevance_score":      0.5,
                    })

            pipeline_stages["stage2_retrieval"] = {
                "total_raw_sources": len(raw_evidence),
                "urls": [ev["url"] for ev in raw_evidence],
            }
            logger.info("[Stream Stage 2] %d raw evidence items scraped", len(raw_evidence))
            payload = {
                "stage": "stage2",
                "status": "running",
                "message": f"Stage 2 Complete: {len(raw_evidence)} web sources scraped.",
                "pipeline_inspector": pipeline_stages,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            logger.error("[Stream Stage 2 Error] %s", e)
            yield f"data: {json.dumps({'status': 'error', 'error': f'Retrieval failed: {e}'})}\n\n"
            return

        if not raw_evidence:
            yield f"data: {json.dumps({'status': 'error', 'error': 'No evidence could be retrieved for the given claim.'})}\n\n"
            return

        # ─────────────────────────────────────────────────────────────
        # STAGE 3: Hybrid Ranking + Cross-Encoder Reranking
        # ─────────────────────────────────────────────────────────────
        try:
            index = HybridIndex()
            ranked = await run_in_threadpool(index.rank_passages, claim, raw_evidence)
            top_passages = ranked[:10]
            reranked = await run_in_threadpool(_reranker.rerank, claim, top_passages, 8)
        except Exception as e:
            logger.warning("[Stream Stage 3] Ranking warning: %s", e)
            reranked = raw_evidence[:8]

        pipeline_stages["stage3_ranking"] = {
            "passages_after_ranking":  len(ranked) if "ranked" in locals() else len(raw_evidence),
            "passages_after_reranking": len(reranked),
            "top_passages": [
                {
                    "url":             ev.get("url", ""),
                    "source_domain":   ev.get("source_domain", ""),
                    "relevance_score": round(ev.get("relevance_score", 0.0), 4),
                    "snippet_preview": ev.get("snippet", "")[:120],
                }
                for ev in reranked
            ],
        }
        logger.info("[Stream Stage 3] %d passages after reranking", len(reranked))
        payload = {
            "stage": "stage3",
            "status": "running",
            "message": f"Stage 3 Complete: Reranked top {len(reranked)} passages.",
            "pipeline_inspector": pipeline_stages,
        }
        yield f"data: {json.dumps(payload)}\n\n"

        # ─────────────────────────────────────────────────────────────
        # STAGE 4: DeBERTa NLI Stance Classification
        # ─────────────────────────────────────────────────────────────
        evidence_nli_pairs = [
            (ev.get("snippet", ""), claim)
            for ev in reranked
            if ev.get("snippet")
        ]
        try:
            evidence_nli_results = await run_in_threadpool(classify_batch, evidence_nli_pairs)
            for i, ev in enumerate(reranked):
                if i < len(evidence_nli_results):
                    ev["stance"] = evidence_nli_results[i]["label"]
        except Exception as e:
            logger.warning("[Stream Stage 4] Evidence NLI warning: %s", e)

        subclaims_with_stances: List[Dict[str, Any]] = []
        sc_nli_debug: List[Dict] = []

        for sc in subclaims:
            sc_pairs = [
                (ev.get("snippet", ""), sc["text"])
                for ev in reranked[:5]
                if ev.get("snippet")
            ]
            if sc_pairs:
                try:
                    sc_results = await run_in_threadpool(classify_batch, sc_pairs)
                    stances = [r["label"]      for r in sc_results]
                    confs   = [r["confidence"] for r in sc_results]
                    majority_stance = Counter(stances).most_common(1)[0][0]
                    avg_conf = round(sum(confs) / len(confs), 4)
                except Exception:
                    majority_stance, avg_conf = "neutral", 0.0
            else:
                majority_stance, avg_conf = "neutral", 0.0

            subclaims_with_stances.append({
                "text":     sc["text"],
                "stance":   majority_stance,
                "nli_conf": avg_conf,
            })
            sc_nli_debug.append({
                "subclaim": sc["text"],
                "stance":   majority_stance,
                "nli_conf": avg_conf,
            })

        pipeline_stages["stage4_nli"] = {
            "subclaim_stances": sc_nli_debug,
            "evidence_stances": [
                {"url": ev.get("url", ""), "stance": ev.get("stance", "neutral")}
                for ev in reranked
            ],
        }
        logger.info("[Stream Stage 4] Subclaim stances: %s", [s["stance"] for s in subclaims_with_stances])
        payload = {
            "stage": "stage4",
            "status": "running",
            "message": "Stage 4 Complete: DeBERTa stance classified.",
            "pipeline_inspector": pipeline_stages,
        }
        yield f"data: {json.dumps(payload)}\n\n"

        # ─────────────────────────────────────────────────────────────
        # STAGE 5: Dynamic Confidence Scoring & Synthesis
        # ─────────────────────────────────────────────────────────────
        try:
            all_stances      = [sc["stance"] for sc in subclaims_with_stances]
            agreement_ratio  = compute_agreement_ratio(all_stances)
            domain_auth      = compute_domain_authority_score(reranked)
            avg_nli_softmax  = compute_avg_nli_softmax(subclaims_with_stances)
            prelim_confidence = compute_dynamic_confidence(agreement_ratio, domain_auth, avg_nli_softmax)

            gate = check_uncertainty(prelim_confidence, reranked, subclaims_with_stances)

            synthesis = await run_in_threadpool(
                synthesize_evidence,
                claim,
                subclaims_with_stances,
                reranked,
            )

            result = run_scoring(
                claim=claim,
                subclaims_with_stances=subclaims_with_stances,
                evidence_list=reranked,
                synthesis_result=synthesis,
                agreement_ratio=agreement_ratio,
                is_uncertain=gate["is_uncertain"],
                uncertainty_reason=gate["reason"],
            )
        except Exception as e:
            logger.error("[Stream Stage 5 Error] %s", e)
            yield f"data: {json.dumps({'status': 'error', 'error': f'Scoring failed: {e}'})}\n\n"
            return

        pipeline_stages["stage5_scoring"] = {
            "agreement_ratio":        round(agreement_ratio, 4),
            "domain_authority_score": round(domain_auth, 4),
            "avg_nli_softmax":        round(avg_nli_softmax, 4),
            "dynamic_confidence":     result["confidence"],
            "overall_score":          result["overall_score"],
            "verdict":                result["verdict_label"],
            "uncertainty_gate":       gate["gate_checks"],
        }
        logger.info(
            "[Stream Stage 5] score=%.1f | conf=%.3f | verdict=%s",
            result["overall_score"], result["confidence"], result["verdict_label"]
        )

        final_response = {
            "claim":              result["claim"],
            "overall_score":      result["overall_score"],
            "verdict_label":      result["verdict_label"],
            "confidence":         result["confidence"],
            "is_uncertain":       result["is_uncertain"],
            "uncertainty_reason": result["uncertainty_reason"],
            "summary":            result["summary"],
            "factors":            result["factors"],
            "subclaims":          result["subclaims"],
            "evidence":           reranked,
            "pipeline_inspector": pipeline_stages,
        }

        payload = {
            "stage": "stage5",
            "status": "complete",
            "message": "Stage 5 Complete: Pipeline analysis complete.",
            "pipeline_inspector": pipeline_stages,
            "result": final_response,
        }
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

