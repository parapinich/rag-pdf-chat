"""
Application configuration using pydantic-settings.
Loads from .env file or environment variables.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # --- Model Configuration ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"

    # --- Chunking Configuration ---
    default_chunk_strategy: str = "fixed"
    fixed_chunk_size: int = 500
    medium_chunk_size: int = 1000
    chunk_overlap: int = 50

    # --- API Settings ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Upload Settings ---
    upload_dir: str = "uploads"
    max_file_size_mb: int = 10

    # --- Retrieval Settings ---
    top_k: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance
settings = Settings()


def get_upload_path() -> Path:
    """Get the upload directory path, creating it if it doesn't exist."""
    path = Path(settings.upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
