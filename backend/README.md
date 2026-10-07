# TrustNet Backend Engine & API Server

Welcome to the **TrustNet Backend**. This repository houses the evidence-first AI credibility analysis REST API server.

---

## Setup & Local Development Instructions

### 1. Virtual Environment Activation
The Python virtual environment is located at `backend/venv`. Activate it using:

```bash
# On Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
Install all required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` in the root `TrustNet/` folder and paste your API keys:
- `GEMINI_API_KEY`: Get a free key from [Google AI Studio](https://aistudio.google.com/)
- `TAVILY_API_KEY`: Get a free key from [Tavily AI](https://tavily.com/)

---

## Phase Logs & Development Progress

## Phase 0: Virtual Environment & Skeleton Complete

### What Was Built (Simple English)
We set up the complete folder structure for the TrustNet backend, evaluation engine, and frontend interfaces. We created a isolated Python virtual environment (`venv`), pinned locked package versions in `requirements.txt`, and provided `.env` templates for API key setup.

### Key Files Created
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/requirements.txt`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/requirements.txt): Pinned backend Python dependencies.
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/.env.example`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/.env.example): Template for required API keys (`GEMINI_API_KEY`, `TAVILY_API_KEY`).
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/.env`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/.env): Active environment configuration file.
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/venv/`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/venv/): Isolated Python virtual environment.

### How to Test / Run
```bash
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend
source venv/bin/activate
pip install -r requirements.txt
```

---

## Phase 1: Core Schemas & Gemini LLM Client Complete

### What Was Built (Simple English)
We built the core Pydantic data schemas for claims, evidence items, 5-factor scores, and analysis responses. We configured the central settings loader (`config.py`) to read environment variables from `.env` and built the Gemini LLM client (`llm_analyzer.py`) using Google AI Studio's OpenAI-compatible base URL.

### Key Files Created
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/core/config.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/core/config.py): Settings & environment variables manager.
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/core/schemas.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/core/schemas.py): Pydantic data models for claims, evidence, 5-factor scores, and response payloads.
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/reasoning/llm_analyzer.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/reasoning/llm_analyzer.py): Gemini 2.0 Flash structured JSON query client.

### How to Test / Run
```bash
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend
source venv/bin/activate
python3 -c "from app.core.config import settings; print('LLM Model:', settings.LLM_MODEL)"
```

---

## Phase 2: Evidence Retrieval Engine Complete

### What Was Built (Simple English)
We built the complete multi-stage evidence retrieval pipeline. It connects to the Tavily AI search API (`search_provider.py`), scrapes clean article text while stripping web ads/clutter (`scraper.py`), combines sparse BM25 token matching with dense `all-MiniLM-L6-v2` vector embeddings via Reciprocal Rank Fusion (`hybrid_index.py`), and filters top relevant passages using a deep Cross-Encoder model (`reranker.py`).

### Key Files Created
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/retrieval/search_provider.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/retrieval/search_provider.py): Tavily API web search client.
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/retrieval/scraper.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/retrieval/scraper.py): Trafilatura web scraper & domain authority classifier.
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/retrieval/hybrid_index.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/retrieval/hybrid_index.py): BM25 + SentenceTransformer Reciprocal Rank Fusion (RRF).
- [`file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/retrieval/reranker.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/retrieval/reranker.py): Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) relevance passage filter.

### How to Test / Run
```bash
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend
source venv/bin/activate
python3 -c "from app.retrieval.scraper import extract_article_content; print(extract_article_content('https://www.reuters.com/world')['authority_tier'])"
```


---

## Phase 3: Reasoning & Stance Engine — Complete

### What Was Built (Simple English)
Phase 3 adds the "brain" of TrustNet — the system that understands *what* the claim is saying and *whether* the evidence agrees. A complex claim is first broken into small, checkable facts by Gemini. Then, for each piece of evidence retrieved in Phase 2, a local AI model (DeBERTa-v3) reads both the evidence and the subclaim and decides: does this evidence *support*, *contradict*, or is it *neutral* toward this claim? Finally, Gemini takes everything together — all subclaims, all stances — and writes a structured, human-readable verdict explanation that feeds into the scoring engine.

### Key Files Created / Updated

| File | Responsibility |
|---|---|
| [`claim_decomposer.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/reasoning/claim_decomposer.py) | Sends claim to Gemini → returns atomic subclaims + targeted search queries |
| [`nli_fast_path.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/reasoning/nli_fast_path.py) | Local DeBERTa-v3 NLI model — classifies each evidence snippet as supports / contradicts / neutral |
| [`llm_analyzer.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/reasoning/llm_analyzer.py) | Updated with `synthesize_evidence()` — Gemini Flash final verdict synthesis |

### How the Pipeline Connects (Phase 3 Data Flow)
```
User Claim
    │
    ▼ claim_decomposer.decompose_claim()
[subclaim_1, subclaim_2, ...]  +  search_queries[]
    │
    ▼ (Phase 2) search_provider + scraper + hybrid_index + reranker
Evidence passages per subclaim
    │
    ▼ nli_fast_path.classify_batch()
stance labels: ["supports", "contradicts", "neutral", ...]
agreement_ratio = compute_agreement_ratio(stances)
    │
    ▼ llm_analyzer.synthesize_evidence()
{summary, supported_subclaims, refuted_subclaims, factor_justifications}
    │
    ▼ Phase 4: Scoring Engine (next)
```

### How to Test
```bash
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend
source venv/bin/activate

# Test imports + Agreement Ratio math
python3 -c "
from app.reasoning.claim_decomposer import decompose_claim, get_all_search_queries
from app.reasoning.nli_fast_path import classify_stance, compute_agreement_ratio
from app.reasoning.llm_analyzer import synthesize_evidence
ratio = compute_agreement_ratio(['supports', 'supports', 'contradicts'])
print('Agreement Ratio:', ratio)  # Expected: 0.6667
print('All Phase 3 imports OK')
"
```

---

## Phase 4: Scoring Engine & Empirical Dynamic Confidence — Complete

### What Was Built (Simple English)
Phase 4 is the math brain of TrustNet. It takes all the signals from Phase 3 — how many sources agree vs. disagree, how authoritative those sources are, and how confident the DeBERTa NLI model was — and plugs them into a real mathematical formula to produce a credibility score and a verdict. There's also an "Uncertainty Gate" that checks 4 conditions before allowing a verdict: if confidence is too low, if there aren't enough sources, if all sources were neutral, or if no subclaim got resolved — it returns "Insufficient Evidence" instead of guessing.

### Key Files Created

| File | Responsibility |
|---|---|
| [`calculator.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/scoring/calculator.py) | 5-Factor scores + Dynamic Confidence formula + Verdict Label mapper |
| [`uncertainty_gate.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/scoring/uncertainty_gate.py) | 4-condition abstention guard — prevents overconfident wrong verdicts |

### Dynamic Confidence Formula (from Architecture Plan §1.B)
```
Confidence = 0.45 × |AgreementRatio|
           + 0.30 × DomainAuthorityScore
           + 0.25 × AvgNLISoftmax
```

### 5-Factor Weights
| Factor | Weight | Measured By |
|---|---|---|
| evidence_strength | 30% | Relevant source count × avg relevance score |
| factual_consistency | 30% | \|AgreementRatio\| × 100 |
| source_provenance | 20% | Average domain authority tier |
| bias_framing | 10% | Inverse of low-authority source fraction |
| incentive_conflict | 10% | Unique domain diversity ratio |

### Verdict Labels (AVeriTeC-compatible 4-class mapping)
| Verdict | Condition |
|---|---|
| `Supported` | agreement ≥ 0.35 AND score ≥ 60 |
| `Refuted` | agreement ≤ −0.35 AND score ≥ 60 |
| `Conflicting Evidence / Cherrypicking` | \|agreement\| < 0.35 AND confidence ≥ 0.45 |
| `Not Enough Evidence` | Low overall score |
| `Insufficient Evidence` | Uncertainty gate triggered |

### How to Test
```bash
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend
source venv/bin/activate
python3 -c "
from app.scoring.calculator import compute_dynamic_confidence, determine_verdict_label
from app.scoring.uncertainty_gate import check_uncertainty
conf = compute_dynamic_confidence(0.75, 0.6875, 0.82)
print('Confidence:', conf)                                   # ~0.75
print('Verdict:', determine_verdict_label(72, 0.6, conf, False))  # Supported
gate = check_uncertainty(0.20, [], [])
print('Gate fired:', gate['is_uncertain'])                   # True
"
```

---

## Phase 5: FastAPI Backend REST Server — Complete

### What Was Built (Simple English)
Phase 5 packages the entire backend pipeline into a production-grade FastAPI REST API server. It exposes endpoints for real-time claim verification (`/api/v1/analyze`) and health monitoring (`/api/v1/health`), handles custom exception formatting, configures CORS middleware for frontend integrations, and asynchronously pre-warms ML models on startup so requests process without latency bottlenecks.

### Key Files Created

| File | Responsibility |
|---|---|
| [`main.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/main.py) | FastAPI app entry point, CORS middleware, background model pre-warming |
| [`routes.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/routes.py) | REST API routes (`/api/v1/analyze` POST, `/api/v1/health` GET) |
| [`exceptions.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend/app/core/exceptions.py) | Custom HTTP exception handler and standard error payload formatter |

### How to Test / Run
```bash
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend
source venv/bin/activate

# 1. Start server
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2. Check health endpoint (in another terminal)
curl -s http://127.0.0.1:8000/api/v1/health

# 3. Test analyze endpoint
curl -s -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"claim": "NASA faked the moon landing in 1969"}'
```

---

## Phase 6: Benchmark Evaluation Suite — Complete

### What Was Built (Simple English)
Phase 6 builds the automated benchmarking and metrics evaluation engine for TrustNet. It loads standardized fact-checking claims and ground truth labels from the AVeriTeC benchmark dataset (`dev.json` and `train.json`), passes them through the TrustNet analysis API, maps predicted verdicts to 4 benchmark classes (`Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence / Cherrypicking`), and calculates quantitative performance metrics: Precision, Recall, Per-Class F1, and overall Macro-F1 score.

### Key Files Created

| File | Responsibility |
|---|---|
| [`loader.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/eval/loader.py) | AVeriTeC dataset loader (`dev.json` & `train.json`), label normalizer, sample batching |
| [`run_benchmark.py`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/eval/run_benchmark.py) | Benchmark evaluation suite, HTTP client runner, Precision/Recall/F1/Macro-F1 calculator |

### How to Run Evaluation
```bash
# Terminal 1: Start backend API server
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Run Benchmark Suite (e.g. 5 samples)
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend
source venv/bin/activate
python3 ../eval/run_benchmark.py --split dev --max-samples 5
```

---

## Phase 7: Clean Minimal UI (Dashboard & Extension) — Complete

### What Was Built (Simple English)
Phase 7 delivers the user-facing interfaces for TrustNet with a clean, monochrome design aesthetic (slate grays, crisp white/black contrast, no vibrant colors). It provides a React + Vite web dashboard featuring a **Live Pipeline Inspector** (which renders stage-by-stage data progress from subclaims to NLI stances to 5-factor math) and a Chrome Manifest v3 browser extension for one-click web fact-checking.

### Key Files Created

| File | Responsibility |
|---|---|
| [`App.jsx`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/frontend/dashboard/src/App.jsx) | Main React Dashboard application & API integration |
| [`PipelineInspector.jsx`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/frontend/dashboard/src/components/PipelineInspector.jsx) | Live Pipeline Inspector tabbed stage visualizer (Stage 1 to Stage 5) |
| [`SummaryCard.jsx`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/frontend/dashboard/src/components/SummaryCard.jsx) | Verdict badge, dynamic confidence indicator, factor bars & synthesis explanation |
| [`index.css`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/frontend/dashboard/src/index.css) | Minimalist monochrome design system & responsive layout styles |
| [`manifest.json`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/frontend/extension/manifest.json) | Chrome Manifest v3 Browser Extension configuration |
| [`popup.js`](file:///home/guest/Downloads/Work/Projects/Capstone/TrustNet/frontend/extension/popup.js) | Extension popup logic capturing selected web text and querying backend API |

### How to Run Dashboard
```bash
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/frontend/dashboard
npm run dev
# Open http://localhost:5173 in your browser
```




