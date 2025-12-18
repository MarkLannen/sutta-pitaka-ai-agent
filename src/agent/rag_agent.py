"""ReAct-style RAG agent for the Pali Canon."""

from typing import Optional

from ..indexing import VectorStoreManager
from ..retrieval import RAGQueryEngine


class PaliRAGAgent:
    """
    High-level agent interface for querying the Pali Canon.

    Provides a simple interface for the Streamlit app and handles
    model switching and query execution.
    """

    AVAILABLE_MODELS = {
        "ollama": "Local (Ollama - Free)",
        "gemini": "Google Gemini 1.5 Flash",
        "claude": "Anthropic Claude 3.5 Sonnet",
    }

    def __init__(self, model_type: str = "ollama"):
        """
        Initialize the Pali RAG Agent.

        Args:
            model_type: Initial model to use - "ollama", "gemini", or "claude"
        """
        self.vector_store = VectorStoreManager()
        self.model_type = model_type
        self._query_engine: Optional[RAGQueryEngine] = None

    @property
    def query_engine(self) -> RAGQueryEngine:
        """Lazily initialize query engine."""
        if self._query_engine is None:
            self._query_engine = RAGQueryEngine(
                vector_store=self.vector_store,
                model_type=self.model_type,
            )
        return self._query_engine

    def is_ready(self) -> bool:
        """Check if the agent is ready (has indexed documents)."""
        return self.vector_store.collection_exists()

    def get_document_count(self) -> int:
        """Get number of indexed document chunks."""
        return self.vector_store.get_document_count()

    def set_model(self, model_type: str) -> None:
        """
        Switch to a different model.

        Args:
            model_type: Model to switch to - "ollama", "gemini", or "claude"
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_type}")

        self.model_type = model_type
        if self._query_engine is not None:
            self._query_engine.switch_model(model_type)

    def ask(self, question: str) -> dict:
        """
        Ask a question about the Pali Canon.

        Args:
            question: User's question

        Returns:
            Dictionary with 'answer', 'citations', and 'formatted' keys
        """
        if not self.is_ready():
            return {
                "answer": "No suttas have been indexed yet. Please run the ingestion script first.",
                "citations": [],
                "formatted": "No suttas have been indexed yet. Please run the ingestion script first.",
            }

        response = self.query_engine.query(question)

        return {
            "answer": response.answer,
            "citations": [
                {
                    "sutta_uid": c.sutta_uid,
                    "segment_range": c.segment_range,
                    "title": c.title,
                    "text": c.text_snippet,
                    "score": c.score,
                }
                for c in response.citations
            ],
            "formatted": response.format_with_sources(),
        }

    def search(self, query: str) -> list[dict]:
        """
        Search for relevant passages without generating an answer.

        Args:
            query: Search query

        Returns:
            List of citation dictionaries
        """
        if not self.is_ready():
            return []

        citations = self.query_engine.retrieve_only(query)

        return [
            {
                "sutta_uid": c.sutta_uid,
                "segment_range": c.segment_range,
                "title": c.title,
                "text": c.text_snippet,
                "score": c.score,
            }
            for c in citations
        ]
