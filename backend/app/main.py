"""
main.py
-------
Phase 5 | FastAPI Application Entry Point.

Starts the TrustNet backend server with:
  - CORS middleware (allow React frontend on localhost:5173)
  - Custom exception handlers
  - /api/v1 route prefix
  - Swagger UI at /docs
  - Startup model pre-warming (avoids cold-start delay on first request)
"""

import socket
_old_getaddrinfo = socket.getaddrinfo
def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _old_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _getaddrinfo_ipv4

import logging
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.routes import router

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="TrustNet Fact-Checking API",
    description=(
        "Evidence-first AI credibility analysis pipeline. "
        "Decomposes claims, retrieves evidence, classifies stance via DeBERTa-v3 NLI, "
        "and produces an empirical dynamic confidence verdict."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception Handlers ────────────────────────────────────────────
register_exception_handlers(app)

# ── Routes ────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ── Startup: Pre-warm models ──────────────────────────────────────
@app.on_event("startup")
async def pre_warm_models() -> None:
    """
    Fires model pre-warming as a background asyncio task so the server
    starts accepting connections immediately. Models load in the background;
    the first request will trigger any model that hasn't finished yet.
    """
    import asyncio

    async def _warm() -> None:
        from fastapi.concurrency import run_in_threadpool
        logger.info("TrustNet starting up — pre-warming ML models in background...")
        try:
            from app.reasoning.nli_fast_path import _get_nli_model
            await run_in_threadpool(_get_nli_model)
            logger.info("✅ DeBERTa-v3 NLI model loaded.")
        except Exception as e:
            logger.warning("⚠️  NLI model pre-warm failed (will load on first request): %s", e)

        try:
            from app.retrieval.hybrid_index import HybridIndex
            await run_in_threadpool(lambda: HybridIndex())
            logger.info("✅ Sentence-transformer dense model loaded.")
        except Exception as e:
            logger.warning("⚠️  HybridIndex pre-warm failed: %s", e)

        logger.info("🚀 TrustNet background warm-up complete.")

    # Fire and forget — server is ready immediately
    asyncio.create_task(_warm())
    logger.info("🚀 TrustNet API accepting connections at http://%s:%d", settings.HOST, settings.PORT)
    logger.info("📖 Swagger UI: http://%s:%d/docs", settings.HOST, settings.PORT)


# ── Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
