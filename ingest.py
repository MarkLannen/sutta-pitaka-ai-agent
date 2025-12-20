#!/usr/bin/env python3
"""CLI script for ingesting suttas from SuttaCentral into the vector store."""

import argparse
import sys

from src.ingestion import SuttaCentralClient, DocumentProcessor
from src.indexing import VectorStoreManager
from src.config import NIKAYA_RANGES


def progress_bar(current: int, total: int, message: str = "", width: int = 40) -> None:
    """Display a simple progress bar."""
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} {message}")
    sys.stdout.flush()
    if current == total:
        print()


def ingest_nikaya(nikaya: str, clear: bool = False) -> None:
    """
    Ingest a nikaya into the vector store.

    Args:
        nikaya: Nikaya code (e.g., "mn")
        clear: Whether to clear existing data first
    """
    print(f"\n{'='*60}")
    print(f"Ingesting {nikaya.upper()} (Majjhima Nikaya)" if nikaya == "mn" else f"Ingesting {nikaya.upper()}")
    print(f"{'='*60}\n")

    # Initialize components
    client = SuttaCentralClient()
    processor = DocumentProcessor()
    vector_store = VectorStoreManager()

    # Clear if requested
    if clear:
        print("Clearing existing collection...")
        vector_store.clear_collection()
        print("Collection cleared.\n")

    # Get sutta range
    sutta_range = NIKAYA_RANGES.get(nikaya)
    if sutta_range is None:
        print(f"Error: Nikaya '{nikaya}' has complex structure. Use individual sutta UIDs.")
        return

    start, end = sutta_range
    total_suttas = end - start + 1

    print(f"Fetching {total_suttas} suttas from SuttaCentral...\n")

    # Fetch suttas
    suttas = []
    for i, num in enumerate(range(start, end + 1), 1):
        sutta_uid = f"{nikaya}{num}"
        progress_bar(i, total_suttas, f"Fetching {sutta_uid}")

        sutta = client.fetch_sutta(sutta_uid)
        if sutta:
            suttas.append(sutta)

    print(f"\nFetched {len(suttas)} suttas successfully.\n")

    # Process into documents
    print("Processing suttas into document chunks...\n")
    all_documents = []

    for i, sutta in enumerate(suttas, 1):
        metadata = client.get_sutta_metadata(sutta)
        progress_bar(i, len(suttas), f"Processing {metadata['sutta_uid']}")

        documents = processor.process_sutta(sutta)
        all_documents.extend(documents)

    print(f"\nCreated {len(all_documents)} document chunks.\n")

    # Index documents
    print("Indexing documents with embeddings (this may take a while)...\n")
    vector_store.add_documents(all_documents, show_progress=True)

    final_count = vector_store.get_document_count()
    print(f"\n{'='*60}")
    print(f"Ingestion complete!")
    print(f"Total chunks in vector store: {final_count:,}")
    print(f"{'='*60}\n")


def ingest_single_sutta(sutta_uid: str) -> None:
    """
    Ingest a single sutta into the vector store.

    Args:
        sutta_uid: Sutta identifier (e.g., "mn1", "dn22")
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


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Sutta Pitaka suttas into the vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --nikaya mn          # Ingest all Majjhima Nikaya suttas
  python ingest.py --nikaya mn --clear  # Clear store and re-ingest
  python ingest.py --sutta mn1          # Ingest single sutta
  python ingest.py --status             # Show current status
        """,
    )

    parser.add_argument(
        "--nikaya",
        choices=["mn", "dn"],
        help="Nikaya to ingest (mn=Majjhima, dn=Digha)",
    )
    parser.add_argument(
        "--sutta",
        type=str,
        help="Single sutta UID to ingest (e.g., mn1, dn22)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before ingesting",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current vector store status",
    )

    args = parser.parse_args()

    if args.status:
        vector_store = VectorStoreManager()
        count = vector_store.get_document_count()
        print(f"\nVector store status:")
        print(f"  Collection: sutta_pitaka")
        print(f"  Document chunks: {count:,}")
        print(f"  Ready: {'Yes' if count > 0 else 'No'}\n")
        return

    if args.nikaya:
        ingest_nikaya(args.nikaya, clear=args.clear)
    elif args.sutta:
        ingest_single_sutta(args.sutta)
    else:
        parser.print_help()
        print("\n💡 Quick start: python ingest.py --nikaya mn")


if __name__ == "__main__":
    main()
