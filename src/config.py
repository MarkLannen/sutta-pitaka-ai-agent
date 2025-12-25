"""Configuration settings for Sutta Pitaka AI Agent."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
CACHE_PATH = PROJECT_ROOT / "cache"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Cloud API keys (optional)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# RAG settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SIMILARITY_TOP_K_LOCAL = 5   # For local models (limited context handling)
SIMILARITY_TOP_K_CLOUD = 20  # For cloud models (better context handling)

# SuttaCentral API
SUTTACENTRAL_API_BASE = "https://suttacentral.net/api"
REQUEST_DELAY = 0.5  # Seconds between API requests to be polite

# Supported Nikayas and their sutta ranges
NIKAYA_RANGES = {
    "mn": (1, 152),   # Majjhima Nikaya
    "dn": (1, 34),    # Digha Nikaya
    "sn": None,       # Samyutta Nikaya (complex structure, use discovery)
    "an": None,       # Anguttara Nikaya (complex structure, use discovery)
}

# Main nikayas of the Sutta Pitaka
ALL_NIKAYAS = ["dn", "mn", "sn", "an"]

# Khuddaka Nikaya collections with Sujato translations
KN_COLLECTIONS = ["kp", "dhp", "ud", "iti", "snp", "thag", "thig"]

# All collections for ingestion
ALL_COLLECTIONS = ALL_NIKAYAS + KN_COLLECTIONS


# =============================================================================
# MODEL REGISTRY - Add new models here
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""

    id: str                      # Unique identifier (used in code)
    provider: str                # Provider type: "ollama", "anthropic", "google", "openai"
    model_id: str                # Provider's model identifier
    display_name: str            # Name shown in UI
    description: str             # Short description for users
    env_var: Optional[str]       # Required env var (None for local models)
    is_free: bool = False        # Whether the model is free to use

    def is_available(self) -> bool:
        """Check if this model can be used (has required credentials)."""
        if self.env_var is None:
            return True  # Local models don't need credentials
        return bool(os.getenv(self.env_var))


# -----------------------------------------------------------------------------
# CONFIGURE YOUR MODELS HERE
# To add a model: append a new ModelConfig to this list
# To remove a model: delete or comment out its entry
# -----------------------------------------------------------------------------

MODELS: list[ModelConfig] = [
    # === Local Models (Free) ===
    ModelConfig(
        id="ollama-llama3",
        provider="ollama",
        model_id="llama3:8b",
        display_name="Llama 3 8B (Local)",
        description="Fast, free, runs locally via Ollama",
        env_var=None,
        is_free=True,
    ),
    ModelConfig(
        id="ollama-mistral",
        provider="ollama",
        model_id="mistral:7b",
        display_name="Mistral 7B (Local)",
        description="Efficient local model via Ollama",
        env_var=None,
        is_free=True,
    ),

    # === Anthropic Models ===
    ModelConfig(
        id="claude-sonnet",
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        description="Excellent reasoning and nuanced responses",
        env_var="ANTHROPIC_API_KEY",
    ),
    ModelConfig(
        id="claude-haiku",
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        description="Fast and cost-effective",
        env_var="ANTHROPIC_API_KEY",
    ),

    # === Google Models ===
    ModelConfig(
        id="gemini-flash",
        provider="google",
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        description="Fast and capable",
        env_var="GOOGLE_API_KEY",
    ),
    ModelConfig(
        id="gemini-pro",
        provider="google",
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        description="Most capable Google model",
        env_var="GOOGLE_API_KEY",
    ),

    # === OpenAI Models ===
    ModelConfig(
        id="gpt-4o",
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        description="OpenAI's flagship model",
        env_var="OPENAI_API_KEY",
    ),
    ModelConfig(
        id="gpt-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        description="Fast and affordable",
        env_var="OPENAI_API_KEY",
    ),
]


def get_model(model_id: str) -> Optional[ModelConfig]:
    """Get a model config by its ID."""
    for model in MODELS:
        if model.id == model_id:
            return model
    return None


def get_available_models() -> list[ModelConfig]:
    """Get all models that have valid credentials configured."""
    return [m for m in MODELS if m.is_available()]


def get_default_model() -> ModelConfig:
    """Get the default model (first available, preferring free models)."""
    available = get_available_models()

    # Prefer free models
    free_models = [m for m in available if m.is_free]
    if free_models:
        return free_models[0]

    # Fall back to first available
    if available:
        return available[0]

    # Last resort: return first model even if not available
    return MODELS[0]
