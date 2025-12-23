"""Agent memory for persisting learned insights."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json

import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding

from ..config import CHROMA_PATH, OLLAMA_BASE_URL, EMBED_MODEL


@dataclass
class WisdomEntry:
    """A stored insight from previous research."""

    query: str
    answer: str
    citations: list[str]  # List of sutta UIDs cited
    created_at: str
    similarity_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": json.dumps(self.citations),
            "created_at": self.created_at,
        }

    @classmethod
    def from_metadata(cls, metadata: dict, text: str, score: float = 0.0) -> "WisdomEntry":
        """Create from ChromaDB metadata."""
        return cls(
            query=metadata.get("query", ""),
            answer=text,
            citations=json.loads(metadata.get("citations", "[]")),
            created_at=metadata.get("created_at", ""),
            similarity_score=score,
        )


class AgentMemory:
    """
    Persistent memory for the Sutta Pitaka AI Agent.

    Stores synthesized answers from previous research so the agent
    can recall relevant knowledge without re-searching.
    """

    COLLECTION_NAME = "agent_wisdom"
    SIMILARITY_THRESHOLD = 0.80  # Minimum score to consider a "hit"

    def __init__(self):
        """Initialize the agent memory store."""
        # Ensure persist directory exists
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model (same as main vector store)
        self.embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        # Get or create the wisdom collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Agent learned insights and synthesized answers"}
        )

    def recall(self, query: str, threshold: Optional[float] = None) -> Optional[WisdomEntry]:
        """
        Check if the agent has previously researched a similar question.

        Args:
            query: The user's question
            threshold: Minimum similarity score (default: SIMILARITY_THRESHOLD)

        Returns:
            WisdomEntry if a relevant past answer is found, None otherwise
        """
        if self.collection.count() == 0:
            return None

        threshold = threshold or self.SIMILARITY_THRESHOLD

        # Embed the query
        query_embedding = self.embed_model.get_query_embedding(query)

        # Search for similar past queries
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )

        # Check if we found a relevant match
        if not results["documents"] or not results["documents"][0]:
            return None

        # ChromaDB returns L2 distance; convert to similarity
        # Lower distance = higher similarity
        distance = results["distances"][0][0]
        # Approximate similarity from L2 distance (normalized embeddings)
        similarity = 1.0 / (1.0 + distance)

        if similarity < threshold:
            return None

        # Return the matched wisdom entry
        return WisdomEntry.from_metadata(
            metadata=results["metadatas"][0][0],
            text=results["documents"][0][0],
            score=similarity,
        )

    def save(
        self,
        query: str,
        answer: str,
        citations: list[str],
    ) -> str:
        """
        Save a synthesized answer to memory.

        Args:
            query: The original question
            answer: The synthesized answer
            citations: List of sutta UIDs that were cited

        Returns:
            The ID of the stored entry
        """
        # Generate unique ID
        entry_id = f"wisdom_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Embed the query (we search by query similarity)
        query_embedding = self.embed_model.get_query_embedding(query)

        # Store in ChromaDB
        self.collection.add(
            ids=[entry_id],
            embeddings=[query_embedding],
            documents=[answer],
            metadatas=[{
                "query": query,
                "citations": json.dumps(citations),
                "created_at": datetime.now().isoformat(),
            }]
        )

        return entry_id

    def get_entry_count(self) -> int:
        """Get the number of stored wisdom entries."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all stored wisdom (for debugging/reset)."""
        self.chroma_client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.chroma_client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Agent learned insights and synthesized answers"}
        )

    def get_all_entries(self, limit: int = 100) -> list[WisdomEntry]:
        """
        Get all stored wisdom entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of WisdomEntry objects
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )

        entries = []
        for i, doc in enumerate(results["documents"]):
            entry = WisdomEntry.from_metadata(
                metadata=results["metadatas"][i],
                text=doc,
            )
            entries.append(entry)

        return entries
