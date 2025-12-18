"""Document processor for converting sutta data into LlamaIndex Documents."""

import json
from typing import Iterator

from llama_index.core.schema import Document

from ..config import CHUNK_SIZE, CHUNK_OVERLAP
from .suttacentral import SuttaCentralClient


class DocumentProcessor:
    """Process sutta data into chunked LlamaIndex Documents with citation metadata."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target size for text chunks (in characters, roughly ~tokens)
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (avg 4 chars per token for English)."""
        return len(text) // 4

    def process_sutta(self, sutta_data: dict) -> list[Document]:
        """
        Convert a single sutta into chunked Documents.

        Args:
            sutta_data: Raw API response from SuttaCentral

        Returns:
            List of LlamaIndex Document objects with metadata
        """
        translation = sutta_data.get("translation_text", {})
        if not translation:
            return []

        # Get metadata
        client = SuttaCentralClient()
        metadata = client.get_sutta_metadata(sutta_data)
        sutta_uid = metadata["sutta_uid"]

        # Sort segments by their IDs to maintain order
        segments = []
        for seg_id, text in translation.items():
            if text and text.strip():  # Skip empty segments
                segments.append((seg_id, text.strip()))

        # Sort by segment ID (handles mn1:0.1, mn1:0.2, mn1:1.1, etc.)
        segments.sort(key=lambda x: self._segment_sort_key(x[0]))

        # Chunk segments while preserving boundaries
        documents = []
        current_chunk_text = []
        current_chunk_segments = []
        current_size = 0

        for seg_id, text in segments:
            text_size = self._estimate_tokens(text)

            # If adding this segment exceeds chunk size, save current chunk
            if current_size + text_size > self.chunk_size and current_chunk_text:
                doc = self._create_document(
                    current_chunk_text,
                    current_chunk_segments,
                    metadata,
                )
                documents.append(doc)

                # Handle overlap: keep some segments for next chunk
                overlap_text = []
                overlap_segments = []
                overlap_size = 0

                for i in range(len(current_chunk_text) - 1, -1, -1):
                    seg_size = self._estimate_tokens(current_chunk_text[i])
                    if overlap_size + seg_size <= self.chunk_overlap:
                        overlap_text.insert(0, current_chunk_text[i])
                        overlap_segments.insert(0, current_chunk_segments[i])
                        overlap_size += seg_size
                    else:
                        break

                current_chunk_text = overlap_text
                current_chunk_segments = overlap_segments
                current_size = overlap_size

            current_chunk_text.append(text)
            current_chunk_segments.append(seg_id)
            current_size += text_size

        # Don't forget the last chunk
        if current_chunk_text:
            doc = self._create_document(
                current_chunk_text,
                current_chunk_segments,
                metadata,
            )
            documents.append(doc)

        return documents

    def _segment_sort_key(self, seg_id: str) -> tuple:
        """
        Create a sort key for segment IDs.

        Handles formats like: mn1:0.1, mn1:1.1, mn1:10.2, mn1:100.3
        """
        try:
            # Split "mn1:1.2" into "mn1" and "1.2"
            uid_part, num_part = seg_id.split(":", 1)

            # Split "1.2" into [1, 2]
            parts = num_part.split(".")
            nums = [int(p) for p in parts if p.isdigit()]

            return (uid_part, *nums)
        except (ValueError, AttributeError):
            return (seg_id, 0, 0)

    def _create_document(
        self,
        texts: list[str],
        segment_ids: list[str],
        sutta_metadata: dict,
    ) -> Document:
        """
        Create a LlamaIndex Document from chunk data.

        Args:
            texts: List of text segments in this chunk
            segment_ids: List of segment IDs corresponding to texts
            sutta_metadata: Metadata from the sutta

        Returns:
            LlamaIndex Document with full metadata
        """
        # Combine texts with newlines
        combined_text = "\n".join(texts)

        # Create segment range for citation
        if len(segment_ids) == 1:
            segment_range = segment_ids[0]
        else:
            # Extract just the segment part (after the colon)
            start_seg = segment_ids[0].split(":")[-1]
            end_seg = segment_ids[-1].split(":")[-1]
            sutta_uid = sutta_metadata["sutta_uid"]
            segment_range = f"{sutta_uid}:{start_seg}-{end_seg}"

        # Build document metadata
        doc_metadata = {
            "sutta_uid": sutta_metadata["sutta_uid"],
            "nikaya": sutta_metadata["nikaya"],
            "title": sutta_metadata["title"],
            "translator": sutta_metadata["translator"],
            "segment_range": segment_range,
            "segment_ids": json.dumps(segment_ids),  # JSON string for ChromaDB compatibility
            "segment_count": len(segment_ids),
        }

        return Document(
            text=combined_text,
            metadata=doc_metadata,
        )

    def process_suttas(self, suttas: list[dict]) -> Iterator[Document]:
        """
        Process multiple suttas into Documents.

        Args:
            suttas: List of sutta data dictionaries

        Yields:
            LlamaIndex Document objects
        """
        for sutta in suttas:
            documents = self.process_sutta(sutta)
            for doc in documents:
                yield doc

    def process_nikaya(
        self,
        nikaya: str,
        progress_callback=None,
    ) -> list[Document]:
        """
        Fetch and process an entire nikaya.

        Args:
            nikaya: Nikaya code (e.g., "mn")
            progress_callback: Optional callback(current, total, sutta_uid) for progress

        Returns:
            List of all Documents from the nikaya
        """
        client = SuttaCentralClient()

        def fetch_progress(current, total):
            if progress_callback:
                progress_callback(current, total, f"Fetching {nikaya}{current}")

        suttas = client.fetch_nikaya(nikaya, progress_callback=fetch_progress)

        all_documents = []
        for i, sutta in enumerate(suttas, 1):
            if progress_callback:
                metadata = client.get_sutta_metadata(sutta)
                progress_callback(i, len(suttas), f"Processing {metadata['sutta_uid']}")

            documents = self.process_sutta(sutta)
            all_documents.extend(documents)

        return all_documents
