"""Discover available suttas from SuttaCentral API."""

import requests
from typing import Generator, Optional

from ..config import SUTTACENTRAL_API_BASE


class SuttaDiscovery:
    """Dynamically discover available suttas from SuttaCentral."""

    # Number of samyuttas in SN (1-56)
    SN_SAMYUTTA_COUNT = 56

    # Number of nipatas in AN (1-11)
    AN_NIPATA_COUNT = 11

    def __init__(self, translator: str = "sujato"):
        """
        Initialize the discovery client.

        Args:
            translator: Translator to filter by (default: "sujato")
        """
        self.translator = translator
        self.api_base = SUTTACENTRAL_API_BASE

    def discover_collection(self, collection_id: str) -> list[str]:
        """
        Discover all sutta UIDs in a collection using suttaplex API.

        Args:
            collection_id: e.g., "sn12", "an1", "ud", "snp"

        Returns:
            List of sutta UIDs that have translations by the configured translator
        """
        url = f"{self.api_base}/suttaplex/{collection_id}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # The response can be a list or a dict with children
            if isinstance(data, list):
                return self._extract_sutta_uids(data)
            elif isinstance(data, dict):
                # Single item or nested structure
                return self._extract_sutta_uids([data])
            return []

        except requests.RequestException as e:
            print(f"Error discovering collection {collection_id}: {e}")
            return []

    def _extract_sutta_uids(self, items: list) -> list[str]:
        """
        Recursively extract sutta UIDs from suttaplex response.

        Args:
            items: List of suttaplex items

        Returns:
            List of sutta UIDs with available translations
        """
        uids = []

        for item in items:
            if not isinstance(item, dict):
                continue

            uid = item.get("uid", "")
            item_type = item.get("type", "")

            # Check if this is a leaf node (actual sutta)
            if item_type == "leaf" or self._is_sutta_uid(uid, item):
                # Check if translator's translation exists
                if self._has_translation(item):
                    uids.append(uid)

            # Recurse into children if present
            children = item.get("children", [])
            if children:
                uids.extend(self._extract_sutta_uids(children))

        return uids

    def _is_sutta_uid(self, uid: str, item: dict) -> bool:
        """
        Determine if a UID represents a sutta (not a chapter/vagga).

        Args:
            uid: The sutta UID
            item: The suttaplex item dict

        Returns:
            True if this is a sutta, False if it's a grouping
        """
        # If it has no children and is not a "branch", it's likely a sutta
        if not item.get("children") and item.get("type") != "branch":
            return True

        # UIDs with dots after the nikaya code are usually suttas
        # e.g., sn12.1, an1.1, ud1.1
        if "." in uid:
            return True

        # Simple numbered UIDs like mn1, dn1 are suttas
        parts = uid.lstrip("abcdefghijklmnopqrstuvwxyz")
        if parts and parts[0].isdigit():
            # Check if there's a range indicator
            if "-" not in parts:
                return True

        return False

    def _has_translation(self, item: dict) -> bool:
        """
        Check if the item has a translation by the configured translator.

        Args:
            item: Suttaplex item dict

        Returns:
            True if translator's translation exists
        """
        translations = item.get("translations", [])
        for t in translations:
            if t.get("author_uid") == self.translator:
                return True
        return False

    def discover_samyutta(self, samyutta_num: int) -> list[str]:
        """
        Discover all suttas in a specific samyutta (SN).

        Args:
            samyutta_num: Samyutta number (1-56)

        Returns:
            List of sutta UIDs
        """
        return self.discover_collection(f"sn{samyutta_num}")

    def discover_nipata(self, nipata_num: int) -> list[str]:
        """
        Discover all suttas in a specific nipata (AN).

        Args:
            nipata_num: Nipata number (1-11)

        Returns:
            List of sutta UIDs
        """
        return self.discover_collection(f"an{nipata_num}")

    def discover_all_sn(self) -> Generator[str, None, None]:
        """
        Discover all SN suttas across all 56 samyuttas.

        Yields:
            Sutta UIDs
        """
        for i in range(1, self.SN_SAMYUTTA_COUNT + 1):
            for uid in self.discover_samyutta(i):
                yield uid

    def discover_all_an(self) -> Generator[str, None, None]:
        """
        Discover all AN suttas across all 11 nipatas.

        Yields:
            Sutta UIDs
        """
        for i in range(1, self.AN_NIPATA_COUNT + 1):
            for uid in self.discover_nipata(i):
                yield uid

    def discover_nikaya(self, nikaya: str) -> list[str]:
        """
        Discover all sutta UIDs for a nikaya.

        Args:
            nikaya: Nikaya code ("dn", "mn", "sn", "an") or sub-collection

        Returns:
            List of sutta UIDs
        """
        nikaya = nikaya.lower()

        if nikaya == "dn":
            return [f"dn{i}" for i in range(1, 35)]
        elif nikaya == "mn":
            return [f"mn{i}" for i in range(1, 153)]
        elif nikaya == "sn":
            return list(self.discover_all_sn())
        elif nikaya == "an":
            return list(self.discover_all_an())
        else:
            # Treat as sub-collection (e.g., "sn12", "an1", "ud", "snp")
            return self.discover_collection(nikaya)

    def get_nikaya_summary(self, nikaya: str) -> dict:
        """
        Get a summary of available suttas for a nikaya.

        Args:
            nikaya: Nikaya code

        Returns:
            Dict with count and sample UIDs
        """
        uids = self.discover_nikaya(nikaya)
        return {
            "nikaya": nikaya,
            "count": len(uids),
            "first_5": uids[:5] if uids else [],
            "last_5": uids[-5:] if uids else [],
        }
