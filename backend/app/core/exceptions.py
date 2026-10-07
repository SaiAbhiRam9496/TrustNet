"""
exceptions.py
-------------
Phase 5 | Custom HTTP Exception Handling.

Registers application-level exception handlers so that any unhandled
error returns a clean, structured JSON response instead of a raw traceback.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


class TrustNetError(Exception):
    """Base exception class for TrustNet application errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class RetrievalError(TrustNetError):
    """Raised when the search/scraping pipeline fails to retrieve any evidence."""
    def __init__(self, message: str = "Failed to retrieve evidence from search providers."):
        super().__init__(message, status_code=503)


class LLMError(TrustNetError):
    """Raised when the Gemini LLM API call fails or returns unparseable output."""
    def __init__(self, message: str = "LLM synthesis call failed or returned invalid JSON."):
        super().__init__(message, status_code=502)


class ScoringError(TrustNetError):
    """Raised when the scoring engine encounters a mathematical error."""
    def __init__(self, message: str = "Scoring engine encountered an unexpected error."):
        super().__init__(message, status_code=500)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers all custom exception handlers on the FastAPI application instance.
    Call this once in main.py during app startup.
    """

    @app.exception_handler(TrustNetError)
    async def trustnet_error_handler(request: Request, exc: TrustNetError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error":   type(exc).__name__,
                "message": exc.message,
                "path":    str(request.url),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error":   "ValidationError",
                "message": "Request body failed validation.",
                "details": exc.errors(),
                "path":    str(request.url),
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error":   "InternalServerError",
                "message": str(exc),
                "path":    str(request.url),
            },
        )
