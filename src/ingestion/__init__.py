"""Data ingestion module for fetching and processing suttas."""

from .suttacentral import SuttaCentralClient
from .processor import DocumentProcessor

__all__ = ["SuttaCentralClient", "DocumentProcessor"]
