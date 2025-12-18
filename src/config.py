"""Configuration settings for Pali Canon RAG Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
CACHE_PATH = PROJECT_ROOT / "cache"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LOCAL_LLM = os.getenv("LOCAL_LLM", "llama3:8b")

# Cloud API keys (optional)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# RAG settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SIMILARITY_TOP_K = 5

# SuttaCentral API
SUTTACENTRAL_API_BASE = "https://suttacentral.net/api"
REQUEST_DELAY = 0.5  # Seconds between API requests to be polite

# Supported Nikayas and their sutta ranges
NIKAYA_RANGES = {
    "mn": (1, 152),   # Majjhima Nikaya
    "dn": (1, 34),    # Digha Nikaya
    "sn": None,       # Samyutta Nikaya (complex structure, handle separately)
    "an": None,       # Anguttara Nikaya (complex structure, handle separately)
}

# Default model for generation
DEFAULT_MODEL = "ollama"  # Options: "ollama", "gemini", "claude"
