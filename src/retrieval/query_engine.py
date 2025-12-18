"""RAG query engine with citation support."""

from dataclasses import dataclass
from typing import Optional

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini

from ..config import (
    OLLAMA_BASE_URL,
    LOCAL_LLM,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    SIMILARITY_TOP_K,
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


class RAGQueryEngine:
    """Query engine for the Pali Canon RAG system."""

    SYSTEM_PROMPT = """You are a knowledgeable assistant specializing in Early Buddhist texts from the Pali Canon.
Your role is to provide accurate, helpful answers based on the sutta passages provided to you.

Guidelines:
- Base your answers primarily on the provided context from the suttas
- When quoting or paraphrasing, be faithful to the source material
- If the context doesn't contain enough information to answer fully, say so
- Use clear, accessible language while respecting the technical Buddhist terminology
- When relevant, mention which sutta the information comes from

Context from the Pali Canon:
{context_str}

Question: {query_str}

Answer based on the suttas above:"""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        model_type: str = "ollama",
        top_k: int = SIMILARITY_TOP_K,
    ):
        """
        Initialize the RAG query engine.

        Args:
            vector_store: VectorStoreManager instance (creates new one if None)
            model_type: LLM to use - "ollama", "gemini", or "claude"
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store or VectorStoreManager()
        self.model_type = model_type
        self.top_k = top_k

        # Initialize LLM based on type
        self.llm = self._create_llm(model_type)

        # Create query engine
        self._query_engine = self._create_query_engine()

    def _create_llm(self, model_type: str) -> LLM:
        """Create the appropriate LLM based on model type."""
        if model_type == "ollama":
            return Ollama(
                model=LOCAL_LLM,
                base_url=OLLAMA_BASE_URL,
                request_timeout=120.0,
            )
        elif model_type == "claude":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            return Anthropic(
                api_key=ANTHROPIC_API_KEY,
                model="claude-3-5-sonnet-20241022",
            )
        elif model_type == "gemini":
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not set in environment")
            return Gemini(
                api_key=GOOGLE_API_KEY,
                model="models/gemini-1.5-flash",
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

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

    def switch_model(self, model_type: str) -> None:
        """
        Switch to a different LLM.

        Args:
            model_type: New model type - "ollama", "gemini", or "claude"
        """
        self.model_type = model_type
        self.llm = self._create_llm(model_type)
        self._query_engine = self._create_query_engine()

    def query(self, question: str) -> RAGResponse:
        """
        Query the RAG system.

        Args:
            question: User's question about the Pali Canon

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
