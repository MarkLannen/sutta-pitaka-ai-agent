"""SuttaCentral API client for fetching Bilara segmented texts."""

import json
import time
from pathlib import Path
from typing import Optional, Generator, Callable

import requests

from ..config import (
    SUTTACENTRAL_API_BASE,
    CACHE_PATH,
    REQUEST_DELAY,
    NIKAYA_RANGES,
    KN_COLLECTIONS,
)
from .sutta_discovery import SuttaDiscovery
from .progress_tracker import ProgressTracker, IngestionProgress


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
        self.discovery = SuttaDiscovery(translator)
        self.progress_tracker = ProgressTracker()

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
        Fetch all suttas from a nikaya (legacy method for simple nikayas).

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
            raise ValueError(f"Nikaya {nikaya} has complex structure, use fetch_collection() instead")

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

    def get_sutta_uids(self, collection: str) -> list[str]:
        """
        Get all sutta UIDs for a collection using dynamic discovery.

        Args:
            collection: Nikaya code ("dn", "mn", "sn", "an") or
                       sub-collection ("sn12", "an1", "ud", "snp", etc.)

        Returns:
            List of sutta UIDs
        """
        return self.discovery.discover_nikaya(collection)

    def fetch_collection(
        self,
        collection: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        resume: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Fetch all suttas from a collection with resume support.

        Args:
            collection: Nikaya code or sub-collection (e.g., "sn", "sn12", "ud")
            progress_callback: Optional callback(current, total, uid) for progress
            resume: Whether to resume from previous progress

        Yields:
            Sutta data dictionaries
        """
        # Discover all sutta UIDs
        print(f"Discovering suttas in {collection}...")
        sutta_uids = self.get_sutta_uids(collection)

        if not sutta_uids:
            print(f"No suttas found for {collection}")
            return

        print(f"Found {len(sutta_uids)} suttas")

        # Initialize or resume progress
        progress = self.progress_tracker.start_job(
            collection,
            sutta_uids,
            force_new=not resume,
        )

        # Get remaining suttas
        if resume and progress.completed_count > 0:
            remaining = self.progress_tracker.get_remaining(progress, sutta_uids)
            print(f"Resuming: {progress.completed_count} already done, {len(remaining)} remaining")
        else:
            remaining = sutta_uids

        total = len(sutta_uids)
        completed = progress.completed_count

        for uid in remaining:
            if progress_callback:
                progress_callback(completed + 1, total, uid)

            try:
                sutta = self.fetch_sutta(uid)
                if sutta:
                    self.progress_tracker.mark_completed(progress, uid)
                    completed += 1
                    yield sutta
                else:
                    self.progress_tracker.mark_failed(progress, uid, "No data returned")
            except Exception as e:
                self.progress_tracker.mark_failed(progress, uid, str(e))
                print(f"Error fetching {uid}: {e}")

            # Rate limiting (only if not from cache)
            if not self._get_cache_path(uid).exists():
                time.sleep(REQUEST_DELAY)

    def fetch_all(
        self,
        collections: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Generator[dict, None, None]:
        """
        Fetch all suttas from multiple collections.

        Args:
            collections: List of collection codes. If None, fetches all.
            progress_callback: Optional callback for progress updates

        Yields:
            Sutta data dictionaries
        """
        from ..config import ALL_COLLECTIONS

        if collections is None:
            collections = ALL_COLLECTIONS

        for collection in collections:
            print(f"\n--- Fetching {collection.upper()} ---")
            yield from self.fetch_collection(collection, progress_callback)

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
