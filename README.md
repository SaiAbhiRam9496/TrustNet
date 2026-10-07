# TrustNet — Evidence-First Fact-Checking Engine

TrustNet is a capstone AI system that takes any claim — a news headline, a social media post, anything — and automatically checks whether it's true, false, or uncertain. It does this by breaking the claim apart, searching the web for real evidence, analyzing each piece of evidence against the claim using AI, and then producing a final verdict with a confidence score.

Everything runs locally on your machine. No subscriptions, no cloud bills — just two free API keys.

---

## How It Works (Plain English)

When you type a claim into the dashboard, it goes through a 5-stage pipeline:

| Stage | What happens |
|-------|-------------|
| **1 — Decompose** | The claim is split into smaller, checkable sub-claims using Gemini AI |
| **2 — Retrieve** | Tavily searches the web and returns relevant articles for each sub-claim |
| **3 — Rank** | A local sentence-transformer model scores and re-ranks the evidence by relevance |
| **4 — NLI** | A local DeBERTa model reads each passage and classifies it as *supports*, *contradicts*, or *neutral* |
| **5 — Synthesize** | Gemini reads all the stances + evidence and writes a final verdict with a score |

The UI streams each stage live so you can watch it process in real time.

---

## Project Structure

```
TrustNet/
├── backend/          # Python FastAPI server — the brain of TrustNet
│   ├── app/
│   │   ├── core/         # Config, environment loading
│   │   ├── reasoning/    # NLI (DeBERTa) + LLM synthesis
│   │   ├── retrieval/    # Tavily web search + reranker
│   │   ├── scoring/      # Scoring engine, verdict calculator
│   │   ├── routes.py     # API endpoints (including SSE stream)
│   │   └── main.py       # App entry point, model warm-up
│   ├── requirements.txt
│   └── venv/             # Python virtual environment
│
├── frontend/
│   └── dashboard/        # React + Vite live dashboard UI
│
├── eval/                 # Benchmark evaluation scripts
│   ├── run_benchmark.py  # Run accuracy tests against AVeriTeC dataset
│   └── datasets/averitec/dev.json
│
├── .env                  # Your API keys (never commit this)
└── .env.example          # Template for .env
```

---

## First Run — Full Setup

Do these steps **once** when you first clone the project.

### Step 1 — API Keys

Copy the example env file and fill in your keys:

```bash
# From TrustNet/
cp .env.example .env
```

Open `.env` and paste:
- `GEMINI_API_KEY` → get a free key at [aistudio.google.com](https://aistudio.google.com/)
- `TAVILY_API_KEY` → get a free key at [tavily.com](https://tavily.com/)

---

### Step 2 — Backend Setup

```bash
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/backend

# Activate the virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The server will pre-load the NLI model and sentence transformer on first boot — this takes about 5–10 seconds. Once you see:

```
🚀 TrustNet API accepting connections at http://127.0.0.1:8000
🚀 TrustNet background warm-up complete.
```

The backend is ready.

---

### Step 3 — Frontend Setup

Open a **second terminal**:

```bash
cd /home/guest/Downloads/Work/Projects/Capstone/TrustNet/frontend/dashboard

# Install Node dependencies (first time only)
npm install

# Start the dev server
npm run dev
```

Open your browser at **http://localhost:5173**

---

## Quick Reference — Frequent Commands

### Backend

| Action | Command | Path |
|--------|---------|------|
| Activate virtual env | `source venv/bin/activate` | `backend/` |
| Start API server (dev) | `uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload` | `backend/` |
| Start API server (quiet) | `uvicorn app.main:app --host 127.0.0.1 --port 8000` | `backend/` |
| Install/update dependencies | `pip install -r requirements.txt` | `backend/` |
| Check API is alive | `curl http://127.0.0.1:8000/api/v1/health` | anywhere |
| Open Swagger API docs | Open `http://127.0.0.1:8000/docs` in browser | — |

### Frontend

| Action | Command | Path |
|--------|---------|------|
| Start dev server (live reload) | `npm run dev` | `frontend/dashboard/` |
| Build for production | `npm run build` | `frontend/dashboard/` |
| Preview production build | `npm run preview` | `frontend/dashboard/` |
| Install dependencies | `npm install` | `frontend/dashboard/` |

### Evaluation

| Action | Command | Path |
|--------|---------|------|
| Run benchmark (2 samples) | `python3 run_benchmark.py --split dev --max-samples 2 --delay 5.0` | `eval/` |
| Run benchmark (25 samples) | `python3 run_benchmark.py --split dev --max-samples 25 --delay 15.0` | `eval/` |
| Run full dev set | `python3 run_benchmark.py --split dev --delay 15.0` | `eval/` |

> **Note on `--delay`:** The Gemini free API has rate limits. Use at least `--delay 10.0` (10 seconds between samples) to avoid timeout errors at Stage 5.

---

## Known Limitations

- **Stage 5 timeouts** — If two claims are run back-to-back quickly, the Gemini free API gets rate-limited and Stage 5 (synthesis) will time out. Wait a minute between runs, or use `--delay 15.0` in eval scripts.
- **Sentence-transformer re-downloads** — On each new request, the app currently re-initialises the sentence-transformer model. This adds ~5s overhead per request. The server pre-warms on startup to minimise this.
- **Free tier only** — Both Gemini and Tavily are free tier. Heavy use will exhaust quotas.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | Python 3, FastAPI, Uvicorn |
| Claim decomposition | Google Gemini 1.5 Flash (via OpenAI-compatible API) |
| Web evidence retrieval | Tavily Search API |
| Semantic reranking | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| NLI stance detection | `cross-encoder/nli-deberta-v3-base` (local) |
| Final verdict synthesis | Google Gemini 1.5 Flash |
| Frontend UI | React 18, Vite, vanilla CSS |
| Evaluation | Python, AVeriTeC dev dataset |

---

## Dataset Setup (AVeriTeC)

The evaluation dataset is **not included in the repo** (it's 12 MB and gitignored). Download it manually:

### Option A — Direct download (recommended)

```bash
# From TrustNet/
mkdir -p eval/datasets/averitec

# Download dev split from the AVeriTeC HuggingFace repo
wget -O eval/datasets/averitec/dev.json \
  https://huggingface.co/datasets/chenxwh/AVeriTeC/resolve/main/data/dev.json
```

### Option B — Via HuggingFace datasets (Python)

```bash
# With venv active
pip install datasets

python3 -c "
from datasets import load_dataset
import json, os

ds = load_dataset('chenxwh/AVeriTeC', split='dev')
os.makedirs('eval/datasets/averitec', exist_ok=True)
with open('eval/datasets/averitec/dev.json', 'w') as f:
    json.dump(list(ds), f, indent=2)
print('Done — dev.json saved.')
"
```

Once the file is at `eval/datasets/averitec/dev.json` the benchmark script will work.
