"""ReAct-style RAG agent for the Sutta Pitaka."""

from typing import Optional

from ..config import (
    ModelConfig,
    get_model,
    get_default_model,
    get_available_models,
)
from ..indexing import VectorStoreManager
from ..retrieval import RAGQueryEngine


class SuttaPitakaRAGAgent:
    """
    High-level agent interface for querying the Sutta Pitaka.

    Provides a simple interface for the Streamlit app and handles
    model switching and query execution.
    """

    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize the Sutta Pitaka RAG Agent.

        Args:
            model_id: Model ID to use (uses default if None)
        """
        self.vector_store = VectorStoreManager()
        self.model_config = get_model(model_id) if model_id else get_default_model()
        self._query_engine: Optional[RAGQueryEngine] = None

    @property
    def query_engine(self) -> RAGQueryEngine:
        """Lazily initialize query engine."""
        if self._query_engine is None:
            self._query_engine = RAGQueryEngine(
                vector_store=self.vector_store,
                model_id=self.model_config.id,
            )
        return self._query_engine

    def is_ready(self) -> bool:
        """Check if the agent is ready (has indexed documents)."""
        return self.vector_store.collection_exists()

    def get_document_count(self) -> int:
        """Get number of indexed document chunks."""
        return self.vector_store.get_document_count()

    @staticmethod
    def get_available_models() -> list[ModelConfig]:
        """Get all models that can be used (have valid credentials)."""
        return get_available_models()

    def get_current_model(self) -> ModelConfig:
        """Get the currently selected model."""
        return self.model_config

    def set_model(self, model_id: str) -> None:
        """
        Switch to a different model.

        Args:
            model_id: ID of the model to switch to

        Raises:
            ValueError: If model ID is unknown or model is not available
        """
        model_config = get_model(model_id)
        if not model_config:
            raise ValueError(f"Unknown model: {model_id}")

        if not model_config.is_available():
            raise ValueError(
                f"Model '{model_config.display_name}' requires "
                f"{model_config.env_var} to be set in your environment"
            )

        self.model_config = model_config
        if self._query_engine is not None:
            self._query_engine.switch_model(model_id)

    def ask(self, question: str) -> dict:
        """
        Ask a question about the Sutta Pitaka.

        Args:
            question: User's question

        Returns:
            Dictionary with 'answer', 'citations', 'formatted', and 'model' keys
        """
        if not self.is_ready():
            return {
                "answer": "No suttas have been indexed yet. Please run the ingestion script first.",
                "citations": [],
                "formatted": "No suttas have been indexed yet. Please run the ingestion script first.",
                "model": self.model_config.display_name,
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
            "model": self.model_config.display_name,
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
