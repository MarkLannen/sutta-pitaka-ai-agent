"""SuttaCentral API client for fetching Bilara segmented texts."""

import json
import time
from pathlib import Path
from typing import Optional

import requests

from ..config import (
    SUTTACENTRAL_API_BASE,
    CACHE_PATH,
    REQUEST_DELAY,
    NIKAYA_RANGES,
)


class SuttaCentralClient:
    """Client for fetching sutta texts from SuttaCentral's Bilara API."""

    def __init__(self, translator: str = "sujato", use_cache: bool = True):
        """
        Initialize the SuttaCentral client.

        Args:
            translator: Translator name (default: "sujato" for Bhikkhu Sujato)
            use_cache: Whether to cache API responses locally
        """
        self.translator = translator
        self.use_cache = use_cache
        self.cache_dir = CACHE_PATH / "suttas"

        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, sutta_uid: str) -> Path:
        """Get the cache file path for a sutta."""
        return self.cache_dir / f"{sutta_uid}_{self.translator}.json"

    def _load_from_cache(self, sutta_uid: str) -> Optional[dict]:
        """Load sutta data from cache if available."""
        cache_path = self._get_cache_path(sutta_uid)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_to_cache(self, sutta_uid: str, data: dict) -> None:
        """Save sutta data to cache."""
        cache_path = self._get_cache_path(sutta_uid)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def fetch_sutta(self, sutta_uid: str) -> Optional[dict]:
        """
        Fetch a single sutta from SuttaCentral API.

        Args:
            sutta_uid: The sutta identifier (e.g., "mn1", "dn22", "sn56.11")

        Returns:
            Dictionary containing sutta data with translation segments,
            or None if fetch failed.
        """
        # Check cache first
        if self.use_cache:
            cached = self._load_from_cache(sutta_uid)
            if cached:
                return cached

        # Fetch from API
        url = f"{SUTTACENTRAL_API_BASE}/bilarasuttas/{sutta_uid}/{self.translator}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Cache the response
            if self.use_cache:
                self._save_to_cache(sutta_uid, data)

            return data

        except requests.RequestException as e:
            print(f"Error fetching {sutta_uid}: {e}")
            return None

    def fetch_nikaya(self, nikaya: str, progress_callback=None) -> list[dict]:
        """
        Fetch all suttas from a nikaya.

        Args:
            nikaya: Nikaya code (e.g., "mn" for Majjhima Nikaya)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of sutta data dictionaries
        """
        if nikaya not in NIKAYA_RANGES:
            raise ValueError(f"Unknown nikaya: {nikaya}")

        sutta_range = NIKAYA_RANGES[nikaya]
        if sutta_range is None:
            raise ValueError(f"Nikaya {nikaya} has complex structure, use fetch_sutta() directly")

        start, end = sutta_range
        suttas = []
        total = end - start + 1

        for i, num in enumerate(range(start, end + 1), 1):
            sutta_uid = f"{nikaya}{num}"

            if progress_callback:
                progress_callback(i, total)

            sutta = self.fetch_sutta(sutta_uid)
            if sutta:
                suttas.append(sutta)

            # Rate limiting (skip if loaded from cache)
            cache_path = self._get_cache_path(sutta_uid)
            if not cache_path.exists():
                time.sleep(REQUEST_DELAY)

        return suttas

    def get_sutta_metadata(self, sutta_data: dict) -> dict:
        """
        Extract metadata from sutta API response.

        Args:
            sutta_data: Raw API response for a sutta

        Returns:
            Dictionary with sutta metadata
        """
        # The API returns nested structure with root_text, translation, etc.
        translation = sutta_data.get("translation_text", {})

        # Get the first segment to extract sutta_uid
        segment_ids = list(translation.keys()) if translation else []
        sutta_uid = segment_ids[0].split(":")[0] if segment_ids else "unknown"

        # Extract title from segment 0.2 (standard Bilara format)
        title_key = f"{sutta_uid}:0.2"
        title = translation.get(title_key, "Unknown Title")

        # Determine nikaya from uid
        nikaya = ""
        for char in sutta_uid:
            if char.isdigit():
                break
            nikaya += char

        return {
            "sutta_uid": sutta_uid,
            "nikaya": nikaya,
            "title": title,
            "translator": self.translator,
            "segment_count": len(segment_ids),
        }
