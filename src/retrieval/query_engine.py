"""RAG query engine with citation support."""

from dataclasses import dataclass
from typing import Optional

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from ..config import (
    OLLAMA_BASE_URL,
    SIMILARITY_TOP_K,
    ModelConfig,
    get_model,
    get_default_model,
)
from ..indexing import VectorStoreManager


@dataclass
class Citation:
    """A citation from a retrieved source."""

    sutta_uid: str
    segment_range: str
    title: str
    text_snippet: str
    score: float

    def format(self) -> str:
        """Format citation for display."""
        return f"({self.sutta_uid}: {self.segment_range}) - {self.title}"


@dataclass
class RAGResponse:
    """Response from the RAG query engine."""

    answer: str
    citations: list[Citation]

    def format_with_sources(self) -> str:
        """Format response with source citations."""
        output = self.answer + "\n\n**Sources:**\n"
        for i, citation in enumerate(self.citations, 1):
            output += f"{i}. {citation.format()}\n"
            # Add snippet preview (truncated)
            snippet = citation.text_snippet[:150]
            if len(citation.text_snippet) > 150:
                snippet += "..."
            output += f"   \"{snippet}\"\n\n"
        return output


def create_llm(model_config: ModelConfig) -> LLM:
    """
    Create an LLM instance from a ModelConfig.

    Args:
        model_config: Configuration for the model to create

    Returns:
        LlamaIndex LLM instance
    """
    provider = model_config.provider

    if provider == "ollama":
        from llama_index.llms.ollama import Ollama
        return Ollama(
            model=model_config.model_id,
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0,
        )

    elif provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        import os
        return Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=model_config.model_id,
        )

    elif provider == "google":
        from llama_index.llms.gemini import Gemini
        import os
        return Gemini(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model=model_config.model_id,
        )

    elif provider == "openai":
        from llama_index.llms.openai import OpenAI
        import os
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_config.model_id,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


class RAGQueryEngine:
    """Query engine for the Sutta Pitaka RAG system."""

    SYSTEM_PROMPT = """You are a knowledgeable assistant specializing in Sutta Pitaka texts.
Your role is to provide accurate, scholarly answers based on the sutta passages provided to you.

Guidelines:
- Base your answers primarily on the provided context from the suttas
- ALWAYS cite the specific sutta for every claim or teaching you reference
- Use inline citations in the format: (SuttaUID, e.g., MN1, DN22, SN56.11)
- When quoting directly, use quotation marks and cite the source
- When paraphrasing, still cite which sutta the teaching comes from
- If the context doesn't contain enough information to answer fully, say so
- Use clear, accessible language while respecting technical Buddhist terminology
- If multiple suttas support a point, cite all relevant sources

Example citation format:
"The Buddha taught that suffering arises from craving (MN1). This is elaborated further in MN38, where..."

Context from the Sutta Pitaka:
{context_str}

Question: {query_str}

Provide a well-cited answer based on the suttas above:"""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        model_id: Optional[str] = None,
        top_k: int = SIMILARITY_TOP_K,
    ):
        """
        Initialize the RAG query engine.

        Args:
            vector_store: VectorStoreManager instance (creates new one if None)
            model_id: Model ID from config (uses default if None)
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store or VectorStoreManager()
        self.top_k = top_k

        # Get model config
        if model_id:
            self.model_config = get_model(model_id)
            if not self.model_config:
                raise ValueError(f"Unknown model ID: {model_id}")
        else:
            self.model_config = get_default_model()

        # Initialize LLM
        self.llm = create_llm(self.model_config)

        # Create query engine
        self._query_engine = self._create_query_engine()

    def _create_query_engine(self) -> RetrieverQueryEngine:
        """Create the LlamaIndex query engine."""
        # Get retriever from index
        retriever = self.vector_store.index.as_retriever(
            similarity_top_k=self.top_k
        )

        # Create response synthesizer with custom prompt
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode="compact",
            text_qa_template=PromptTemplate(self.SYSTEM_PROMPT),
        )

        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

    def switch_model(self, model_id: str) -> None:
        """
        Switch to a different LLM.

        Args:
            model_id: Model ID from config
        """
        model_config = get_model(model_id)
        if not model_config:
            raise ValueError(f"Unknown model ID: {model_id}")

        if not model_config.is_available():
            raise ValueError(
                f"Model {model_config.display_name} requires {model_config.env_var} to be set"
            )

        self.model_config = model_config
        self.llm = create_llm(model_config)
        self._query_engine = self._create_query_engine()

    def get_current_model(self) -> ModelConfig:
        """Get the currently active model config."""
        return self.model_config

    def query(self, question: str) -> RAGResponse:
        """
        Query the RAG system.

        Args:
            question: User's question about the Sutta Pitaka

        Returns:
            RAGResponse with answer and citations
        """
        # Execute query
        response = self._query_engine.query(question)

        # Extract citations from source nodes
        citations = []
        for node in response.source_nodes:
            metadata = node.node.metadata
            citation = Citation(
                sutta_uid=metadata.get("sutta_uid", "unknown"),
                segment_range=metadata.get("segment_range", ""),
                title=metadata.get("title", "Unknown Sutta"),
                text_snippet=node.node.text,
                score=node.score or 0.0,
            )
            citations.append(citation)

        return RAGResponse(
            answer=str(response),
            citations=citations,
        )

    def retrieve_only(self, question: str) -> list[Citation]:
        """
        Retrieve relevant passages without generating an answer.

        Useful for exploring what sources would be used.

        Args:
            question: Search query

        Returns:
            List of relevant citations
        """
        retriever = self.vector_store.index.as_retriever(
            similarity_top_k=self.top_k
        )
        nodes = retriever.retrieve(question)

        citations = []
        for node in nodes:
            metadata = node.node.metadata
            citation = Citation(
                sutta_uid=metadata.get("sutta_uid", "unknown"),
                segment_range=metadata.get("segment_range", ""),
                title=metadata.get("title", "Unknown Sutta"),
                text_snippet=node.node.text,
                score=node.score or 0.0,
            )
            citations.append(citation)

        return citations
