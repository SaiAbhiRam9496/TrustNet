import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    # LLM Configuration
    LLM_MODEL: str = "gemma-4-26b-a4b-it"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEFAULT_LLM_TIMEOUT: float = 45.0

    # Server Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Credibility Scoring Thresholds
    MIN_REQUIRED_SOURCES: int = 2
    UNCERTAINTY_CONFIDENCE_THRESHOLD: float = 0.40

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
