"""Track ingestion progress for resume capability."""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class IngestionProgress:
    """Tracks the state of an ingestion job."""

    job_id: str
    nikaya: str
    started_at: str
    total_suttas: int
    completed_suttas: list[str] = field(default_factory=list)
    failed_suttas: dict[str, str] = field(default_factory=dict)  # UID -> error
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

    @property
    def completed_count(self) -> int:
        return len(self.completed_suttas)

    @property
    def failed_count(self) -> int:
        return len(self.failed_suttas)

    @property
    def remaining_count(self) -> int:
        return self.total_suttas - self.completed_count

    @property
    def progress_percent(self) -> float:
        if self.total_suttas == 0:
            return 0.0
        return self.completed_count / self.total_suttas * 100

    @property
    def is_complete(self) -> bool:
        return self.completed_count >= self.total_suttas


class ProgressTracker:
    """Manages ingestion progress with resume capability."""

    def __init__(self, progress_dir: Optional[Path] = None):
        """
        Initialize the progress tracker.

        Args:
            progress_dir: Directory to store progress files.
                         Defaults to cache/ingestion_progress/
        """
        from ..config import CACHE_PATH

        self.progress_dir = progress_dir or CACHE_PATH / "ingestion_progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def _get_progress_path(self, nikaya: str) -> Path:
        """Get the progress file path for a nikaya."""
        return self.progress_dir / f"{nikaya}_progress.json"

    def start_job(
        self,
        nikaya: str,
        sutta_uids: list[str],
        force_new: bool = False,
    ) -> IngestionProgress:
        """
        Start a new ingestion job or resume an existing one.

        Args:
            nikaya: Nikaya code
            sutta_uids: List of all sutta UIDs to ingest
            force_new: If True, start fresh even if progress exists

        Returns:
            IngestionProgress object
        """
        progress_path = self._get_progress_path(nikaya)

        if not force_new and progress_path.exists():
            # Resume existing job
            existing = self.load_progress(nikaya)
            if existing:
                # Update total in case the sutta list changed
                existing.total_suttas = len(sutta_uids)
                return existing

        # Start new job
        progress = IngestionProgress(
            job_id=f"{nikaya}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            nikaya=nikaya,
            started_at=datetime.now().isoformat(),
            total_suttas=len(sutta_uids),
            completed_suttas=[],
            failed_suttas={},
            last_updated=datetime.now().isoformat(),
        )
        self.save_progress(progress)
        return progress

    def mark_completed(self, progress: IngestionProgress, sutta_uid: str) -> None:
        """
        Mark a sutta as successfully ingested.

        Args:
            progress: The progress object
            sutta_uid: UID of completed sutta
        """
        if sutta_uid not in progress.completed_suttas:
            progress.completed_suttas.append(sutta_uid)

        # Remove from failed if it was there
        if sutta_uid in progress.failed_suttas:
            del progress.failed_suttas[sutta_uid]

        progress.last_updated = datetime.now().isoformat()
        self.save_progress(progress)

    def mark_failed(
        self,
        progress: IngestionProgress,
        sutta_uid: str,
        error: str,
    ) -> None:
        """
        Mark a sutta as failed.

        Args:
            progress: The progress object
            sutta_uid: UID of failed sutta
            error: Error message
        """
        progress.failed_suttas[sutta_uid] = error
        progress.last_updated = datetime.now().isoformat()
        self.save_progress(progress)

    def get_remaining(
        self,
        progress: IngestionProgress,
        all_uids: list[str],
    ) -> list[str]:
        """
        Get list of suttas not yet completed.

        Args:
            progress: The progress object
            all_uids: List of all sutta UIDs

        Returns:
            List of UIDs not in completed_suttas
        """
        completed_set = set(progress.completed_suttas)
        return [uid for uid in all_uids if uid not in completed_set]

    def save_progress(self, progress: IngestionProgress) -> None:
        """Save progress to disk."""
        path = self._get_progress_path(progress.nikaya)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(progress), f, indent=2)

    def load_progress(self, nikaya: str) -> Optional[IngestionProgress]:
        """
        Load existing progress from disk.

        Args:
            nikaya: Nikaya code

        Returns:
            IngestionProgress if exists, None otherwise
        """
        path = self._get_progress_path(nikaya)
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return IngestionProgress(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not load progress for {nikaya}: {e}")
            return None

    def clear_progress(self, nikaya: str) -> bool:
        """
        Clear progress for a nikaya (start fresh).

        Args:
            nikaya: Nikaya code

        Returns:
            True if progress was cleared, False if no progress existed
        """
        path = self._get_progress_path(nikaya)
        if path.exists():
            path.unlink()
            return True
        return False

    def get_all_progress(self) -> dict[str, IngestionProgress]:
        """
        Get progress for all nikayas.

        Returns:
            Dict mapping nikaya code to IngestionProgress
        """
        result = {}
        for path in self.progress_dir.glob("*_progress.json"):
            nikaya = path.stem.replace("_progress", "")
            progress = self.load_progress(nikaya)
            if progress:
                result[nikaya] = progress
        return result

    def get_summary(self) -> str:
        """
        Get a formatted summary of all ingestion progress.

        Returns:
            Formatted string summary
        """
        all_progress = self.get_all_progress()

        if not all_progress:
            return "No ingestion progress tracked yet."

        lines = ["Ingestion Progress:", "=" * 50]

        for nikaya, progress in sorted(all_progress.items()):
            status = "COMPLETE" if progress.is_complete else "IN PROGRESS"
            lines.append(
                f"{nikaya.upper()}: {progress.completed_count}/{progress.total_suttas} "
                f"({progress.progress_percent:.1f}%) - {status}"
            )
            if progress.failed_count > 0:
                lines.append(f"  Failed: {progress.failed_count}")

        return "\n".join(lines)
