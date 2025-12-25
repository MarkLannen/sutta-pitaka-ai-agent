#!/usr/bin/env python3
"""CLI script for ingesting suttas from SuttaCentral into the vector store."""

import argparse
import sys

from src.ingestion import SuttaCentralClient, DocumentProcessor, ProgressTracker
from src.indexing import VectorStoreManager
from src.config import NIKAYA_RANGES, ALL_NIKAYAS, KN_COLLECTIONS, ALL_COLLECTIONS


def progress_bar(current: int, total: int, message: str = "", width: int = 40) -> None:
    """Display a simple progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} {message}")
    sys.stdout.flush()
    if current == total:
        print()


def ingest_collection(
    collection: str,
    clear: bool = False,
    dry_run: bool = False,
    resume: bool = True,
) -> None:
    """
    Ingest a collection into the vector store.

    Args:
        collection: Collection code (e.g., "mn", "sn", "sn12", "ud")
        clear: Whether to clear existing data first
        dry_run: If True, only show what would be done
        resume: Whether to resume from previous progress
    """
    collection_upper = collection.upper()
    print(f"\n{'='*60}")
    print(f"Ingesting {collection_upper}")
    print(f"{'='*60}\n")

    # Initialize components
    client = SuttaCentralClient()

    # Discover suttas first
    print("Discovering available suttas...")
    sutta_uids = client.get_sutta_uids(collection)

    if not sutta_uids:
        print(f"No suttas found for {collection}")
        return

    print(f"Found {len(sutta_uids)} suttas\n")

    if dry_run:
        print("DRY RUN - would ingest the following suttas:")
        print(f"  First 5: {sutta_uids[:5]}")
        if len(sutta_uids) > 10:
            print(f"  Last 5:  {sutta_uids[-5:]}")
        print(f"\nTotal: {len(sutta_uids)} suttas")
        return

    processor = DocumentProcessor()
    vector_store = VectorStoreManager()

    # Clear if requested
    if clear:
        print("Clearing existing collection...")
        vector_store.clear_collection()
        print("Collection cleared.\n")

    # Check for resume
    if resume:
        progress = client.progress_tracker.load_progress(collection)
        if progress and progress.completed_count > 0:
            print(f"Resuming from previous progress: {progress.completed_count}/{progress.total_suttas}")

    print(f"Fetching and processing suttas...\n")

    # Fetch and process suttas
    all_documents = []
    fetched_count = 0

    def on_progress(current, total, uid):
        progress_bar(current, total, f"Fetching {uid}")

    for sutta in client.fetch_collection(collection, on_progress, resume=resume):
        fetched_count += 1
        metadata = client.get_sutta_metadata(sutta)
        documents = processor.process_sutta(sutta)
        all_documents.extend(documents)

    print(f"\n\nFetched {fetched_count} suttas")
    print(f"Created {len(all_documents)} document chunks\n")

    if all_documents:
        # Index documents
        print("Indexing documents with embeddings (this may take a while)...\n")
        vector_store.add_documents(all_documents, show_progress=True)

    final_count = vector_store.get_document_count()
    print(f"\n{'='*60}")
    print(f"Ingestion complete!")
    print(f"Total chunks in vector store: {final_count:,}")
    print(f"{'='*60}\n")


def ingest_all(clear: bool = False, dry_run: bool = False) -> None:
    """Ingest all available collections."""
    print("\n" + "=" * 60)
    print("FULL SUTTA PITAKA INGESTION")
    print("=" * 60)
    print(f"\nCollections to ingest: {', '.join(ALL_COLLECTIONS)}\n")

    if dry_run:
        print("DRY RUN - showing available suttas per collection:\n")
        client = SuttaCentralClient()
        total = 0
        for collection in ALL_COLLECTIONS:
            uids = client.get_sutta_uids(collection)
            print(f"  {collection.upper()}: {len(uids)} suttas")
            total += len(uids)
        print(f"\n  TOTAL: {total} suttas")
        return

    # Clear only on first collection if requested
    first = True
    for collection in ALL_COLLECTIONS:
        ingest_collection(
            collection,
            clear=clear and first,
            dry_run=False,
            resume=True,
        )
        first = False


def ingest_single_sutta(sutta_uid: str) -> None:
    """
    Ingest a single sutta into the vector store.

    Args:
        sutta_uid: Sutta identifier (e.g., "mn1", "sn12.1", "ud1.1")
    """
    print(f"\nIngesting {sutta_uid}...\n")

    client = SuttaCentralClient()
    processor = DocumentProcessor()
    vector_store = VectorStoreManager()

    sutta = client.fetch_sutta(sutta_uid)
    if not sutta:
        print(f"Error: Could not fetch {sutta_uid}")
        return

    metadata = client.get_sutta_metadata(sutta)
    print(f"Title: {metadata['title']}")
    print(f"Segments: {metadata['segment_count']}")

    documents = processor.process_sutta(sutta)
    print(f"Document chunks: {len(documents)}")

    print("\nIndexing...")
    vector_store.add_documents(documents, show_progress=True)

    print(f"\nDone! Total chunks in store: {vector_store.get_document_count():,}")


def show_status() -> None:
    """Show current ingestion status for all collections."""
    vector_store = VectorStoreManager()
    tracker = ProgressTracker()

    print(f"\n{'='*60}")
    print("INGESTION STATUS")
    print(f"{'='*60}\n")

    # Vector store status
    count = vector_store.get_document_count()
    print(f"Vector Store:")
    print(f"  Collection: sutta_pitaka")
    print(f"  Document chunks: {count:,}")
    print(f"  Ready: {'Yes' if count > 0 else 'No'}\n")

    # Progress by collection
    print("Collection Progress:")
    all_progress = tracker.get_all_progress()

    if not all_progress:
        print("  No ingestion progress tracked yet.")
    else:
        for nikaya in ALL_COLLECTIONS:
            progress = all_progress.get(nikaya)
            if progress:
                status = "COMPLETE" if progress.is_complete else "IN PROGRESS"
                print(
                    f"  {nikaya.upper()}: {progress.completed_count}/{progress.total_suttas} "
                    f"({progress.progress_percent:.1f}%) - {status}"
                )
                if progress.failed_count > 0:
                    print(f"       Failed: {progress.failed_count}")
            else:
                print(f"  {nikaya.upper()}: Not started")

    print()


def clear_progress(collection: str) -> None:
    """Clear progress for a collection."""
    tracker = ProgressTracker()
    if tracker.clear_progress(collection):
        print(f"Progress cleared for {collection}")
    else:
        print(f"No progress found for {collection}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Sutta Pitaka suttas into the vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --nikaya mn           # Ingest Majjhima Nikaya
  python ingest.py --nikaya sn           # Ingest full Samyutta Nikaya
  python ingest.py --nikaya sn12         # Ingest just SN12 (Nidanasamyutta)
  python ingest.py --nikaya an1          # Ingest AN Book of Ones
  python ingest.py --kn ud               # Ingest Udana
  python ingest.py --kn snp              # Ingest Suttanipata
  python ingest.py --all                 # Ingest entire Sutta Pitaka
  python ingest.py --sutta sn12.1        # Ingest single sutta
  python ingest.py --status              # Show ingestion status
  python ingest.py --clear-progress sn   # Clear progress and start fresh
  python ingest.py --dry-run --nikaya sn # Preview without ingesting
        """,
    )

    parser.add_argument(
        "--nikaya",
        type=str,
        help="Nikaya or sub-collection to ingest (dn, mn, sn, an, sn12, an1, etc.)",
    )
    parser.add_argument(
        "--kn",
        choices=KN_COLLECTIONS,
        help="Khuddaka Nikaya collection to ingest (kp, dhp, ud, iti, snp, thag, thig)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest entire Sutta Pitaka (all available translations)",
    )
    parser.add_argument(
        "--sutta",
        type=str,
        help="Single sutta UID to ingest (e.g., mn1, sn12.1, ud1.1)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before ingesting",
    )
    parser.add_argument(
        "--clear-progress",
        type=str,
        metavar="COLLECTION",
        help="Clear progress tracking for a collection (start fresh)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current ingestion status for all collections",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without actually ingesting",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from previous progress",
    )

    args = parser.parse_args()

    # Handle commands
    if args.status:
        show_status()
    elif args.clear_progress:
        clear_progress(args.clear_progress)
    elif args.all:
        ingest_all(clear=args.clear, dry_run=args.dry_run)
    elif args.nikaya:
        ingest_collection(
            args.nikaya,
            clear=args.clear,
            dry_run=args.dry_run,
            resume=not args.no_resume,
        )
    elif args.kn:
        ingest_collection(
            args.kn,
            clear=args.clear,
            dry_run=args.dry_run,
            resume=not args.no_resume,
        )
    elif args.sutta:
        ingest_single_sutta(args.sutta)
    else:
        parser.print_help()
        print("\nQuick start: python ingest.py --nikaya mn")


if __name__ == "__main__":
    main()
