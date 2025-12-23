"""Iterative AI agent with memory for comprehensive Sutta Pitaka research."""

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM

from ..config import (
    ModelConfig,
    get_model,
    get_default_model,
    SIMILARITY_TOP_K_CLOUD,
)
from ..indexing import VectorStoreManager
from ..retrieval.query_engine import Citation, create_llm
from .memory import AgentMemory, WisdomEntry


class AgentPhase(Enum):
    """Current phase of agent execution."""
    RECALL = "recall"
    SEARCH = "search"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    LEARN = "learn"
    COMPLETE = "complete"


@dataclass
class AgentProgress:
    """Progress update from the agent."""
    phase: AgentPhase
    iteration: int
    max_iterations: int
    message: str
    found_in_memory: bool = False


@dataclass
class AnalysisResult:
    """Result of gap analysis."""
    is_complete: bool
    gaps_identified: list[str]
    next_query: Optional[str] = None
    reasoning: str = ""


@dataclass
class AgentResponse:
    """Response from the iterative agent."""
    answer: str
    citations: list[Citation]
    from_memory: bool = False
    iterations_used: int = 0
    total_passages_retrieved: int = 0

    def format_with_sources(self) -> str:
        """Format response with source citations."""
        source_note = " (recalled from memory)" if self.from_memory else ""
        output = f"{self.answer}{source_note}\n\n**Sources:**\n"

        seen_suttas = set()
        for citation in self.citations:
            if citation.sutta_uid in seen_suttas:
                continue
            seen_suttas.add(citation.sutta_uid)
            output += f"- {citation.sutta_uid}: {citation.title}\n"

        return output


@dataclass
class RetrievedPassage:
    """A passage retrieved from the vector store."""
    chunk_id: str
    sutta_uid: str
    title: str
    text: str
    segment_range: str
    score: float

    def to_citation(self) -> Citation:
        """Convert to Citation for output."""
        return Citation(
            sutta_uid=self.sutta_uid,
            segment_range=self.segment_range,
            title=self.title,
            text_snippet=self.text,
            score=self.score,
        )


class SuttaPitakaAgent:
    """
    AI Agent that iteratively searches the Sutta Pitaka and remembers findings.

    This agent performs multi-pass retrieval with gap analysis to ensure
    comprehensive coverage of topics, and persists learned insights for
    future queries.
    """

    # Prompt for analyzing coverage gaps
    GAP_ANALYSIS_PROMPT = PromptTemplate(
        """You are analyzing whether the retrieved passages fully answer a research question.

Question: {query}

Retrieved passages:
{passages}

Analyze the passages and determine:
1. Do these passages fully answer the question?
2. What aspects of the question are NOT covered by these passages?
3. If more information is needed, what specific search query would help find it?

Respond in this exact format:
COMPLETE: [yes/no]
GAPS: [comma-separated list of missing aspects, or "none"]
NEXT_QUERY: [a refined search query to find missing information, or "none"]
REASONING: [brief explanation of your analysis]"""
    )

    # Prompt for synthesizing the final answer
    SYNTHESIS_PROMPT = PromptTemplate(
        """You are a scholarly assistant synthesizing research findings about the Sutta Pitaka.

Original Question: {query}

Retrieved Passages (from {num_passages} passages across {num_suttas} suttas):
{passages}

INSTRUCTIONS:
- Synthesize a comprehensive answer using ONLY the passages provided above
- Cite the sutta UID for every claim (e.g., MN1, DN22)
- If passages are contradictory or show different perspectives, note this
- If the passages do not fully answer the question, acknowledge the limitations
- Organize thematically when appropriate
- Use direct quotes sparingly, with proper attribution

Synthesized Answer:"""
    )

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        model_id: Optional[str] = None,
        max_iterations: int = 5,
        initial_top_k: int = 20,
        iteration_top_k: int = 10,
        use_memory: bool = True,
    ):
        """
        Initialize the iterative agent.

        Args:
            vector_store: VectorStoreManager instance (creates new one if None)
            model_id: Model ID from config (uses default if None)
            max_iterations: Maximum search iterations
            initial_top_k: Number of passages for initial search
            iteration_top_k: Number of passages per refinement iteration
            use_memory: Whether to use the memory system
        """
        self.vector_store = vector_store or VectorStoreManager()
        self.max_iterations = max_iterations
        self.initial_top_k = initial_top_k
        self.iteration_top_k = iteration_top_k
        self.use_memory = use_memory

        # Get model config
        if model_id:
            self.model_config = get_model(model_id)
            if not self.model_config:
                raise ValueError(f"Unknown model ID: {model_id}")
        else:
            self.model_config = get_default_model()

        # Initialize LLM
        self.llm = create_llm(self.model_config)

        # Initialize memory if enabled
        self.memory = AgentMemory() if use_memory else None

        # Progress callback
        self._progress_callback: Optional[Callable[[AgentProgress], None]] = None

    def set_progress_callback(self, callback: Callable[[AgentProgress], None]) -> None:
        """Set a callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(self, progress: AgentProgress) -> None:
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(progress)

    def switch_model(self, model_id: str) -> None:
        """Switch to a different LLM."""
        model_config = get_model(model_id)
        if not model_config:
            raise ValueError(f"Unknown model ID: {model_id}")
        if not model_config.is_available():
            raise ValueError(
                f"Model {model_config.display_name} requires {model_config.env_var}"
            )
        self.model_config = model_config
        self.llm = create_llm(model_config)

    def _retrieve(self, query: str, top_k: int) -> list[RetrievedPassage]:
        """Retrieve passages from the vector store."""
        retriever = self.vector_store.index.as_retriever(
            similarity_top_k=top_k
        )
        nodes = retriever.retrieve(query)

        passages = []
        for node in nodes:
            metadata = node.node.metadata
            passage = RetrievedPassage(
                chunk_id=node.node.node_id,
                sutta_uid=metadata.get("sutta_uid", "unknown"),
                title=metadata.get("title", "Unknown Sutta"),
                text=node.node.text,
                segment_range=metadata.get("segment_range", ""),
                score=node.score or 0.0,
            )
            passages.append(passage)

        return passages

    def _deduplicate_passages(
        self,
        passages: list[RetrievedPassage]
    ) -> list[RetrievedPassage]:
        """Remove duplicate passages by chunk_id."""
        seen_ids = set()
        unique = []
        for p in passages:
            if p.chunk_id not in seen_ids:
                seen_ids.add(p.chunk_id)
                unique.append(p)
        return unique

    def _format_passages_for_prompt(self, passages: list[RetrievedPassage]) -> str:
        """Format passages for inclusion in prompts."""
        formatted = []
        for i, p in enumerate(passages, 1):
            formatted.append(
                f"[{i}] {p.sutta_uid} ({p.title}) - {p.segment_range}\n{p.text}\n"
            )
        return "\n".join(formatted)

    def _analyze_coverage(
        self,
        query: str,
        passages: list[RetrievedPassage],
    ) -> AnalysisResult:
        """Analyze whether passages fully cover the query."""
        passages_text = self._format_passages_for_prompt(passages[:30])  # Limit for context

        prompt = self.GAP_ANALYSIS_PROMPT.format(
            query=query,
            passages=passages_text,
        )

        response = self.llm.complete(prompt)
        response_text = str(response)

        # Parse the structured response
        is_complete = False
        gaps = []
        next_query = None
        reasoning = ""

        for line in response_text.strip().split("\n"):
            line = line.strip()
            if line.startswith("COMPLETE:"):
                is_complete = "yes" in line.lower()
            elif line.startswith("GAPS:"):
                gap_text = line.replace("GAPS:", "").strip()
                if gap_text.lower() != "none":
                    gaps = [g.strip() for g in gap_text.split(",") if g.strip()]
            elif line.startswith("NEXT_QUERY:"):
                nq = line.replace("NEXT_QUERY:", "").strip()
                if nq.lower() != "none":
                    next_query = nq
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return AnalysisResult(
            is_complete=is_complete,
            gaps_identified=gaps,
            next_query=next_query,
            reasoning=reasoning,
        )

    def _synthesize(
        self,
        query: str,
        passages: list[RetrievedPassage],
    ) -> str:
        """Synthesize a comprehensive answer from passages."""
        # Get unique suttas
        unique_suttas = set(p.sutta_uid for p in passages)

        passages_text = self._format_passages_for_prompt(passages[:50])  # Limit for context

        prompt = self.SYNTHESIS_PROMPT.format(
            query=query,
            num_passages=len(passages),
            num_suttas=len(unique_suttas),
            passages=passages_text,
        )

        response = self.llm.complete(prompt)
        return str(response)

    def research(
        self,
        query: str,
        skip_memory: bool = False,
    ) -> AgentResponse:
        """
        Research a question about the Sutta Pitaka.

        Performs iterative search with gap analysis and optionally
        saves findings to memory.

        Args:
            query: The research question
            skip_memory: If True, skip memory recall (but still save to memory)

        Returns:
            AgentResponse with comprehensive answer and citations
        """
        # Phase 1: Recall from memory
        if self.use_memory and self.memory and not skip_memory:
            self._report_progress(AgentProgress(
                phase=AgentPhase.RECALL,
                iteration=0,
                max_iterations=self.max_iterations,
                message="Checking memory for previous research...",
            ))

            wisdom = self.memory.recall(query)
            if wisdom:
                self._report_progress(AgentProgress(
                    phase=AgentPhase.COMPLETE,
                    iteration=0,
                    max_iterations=self.max_iterations,
                    message="Found relevant previous research!",
                    found_in_memory=True,
                ))

                # Convert stored citations to Citation objects
                citations = [
                    Citation(
                        sutta_uid=uid,
                        segment_range="",
                        title="",
                        text_snippet="",
                        score=0.0,
                    )
                    for uid in wisdom.citations
                ]

                return AgentResponse(
                    answer=wisdom.answer,
                    citations=citations,
                    from_memory=True,
                    iterations_used=0,
                    total_passages_retrieved=0,
                )

        # Phase 2: Initial search
        self._report_progress(AgentProgress(
            phase=AgentPhase.SEARCH,
            iteration=1,
            max_iterations=self.max_iterations,
            message=f"Searching suttas (retrieving top {self.initial_top_k})...",
        ))

        all_passages = self._retrieve(query, self.initial_top_k)
        iterations_used = 1

        # Phase 3: Iterative refinement
        for i in range(2, self.max_iterations + 1):
            self._report_progress(AgentProgress(
                phase=AgentPhase.ANALYZE,
                iteration=i,
                max_iterations=self.max_iterations,
                message=f"Analyzing coverage ({len(all_passages)} passages)...",
            ))

            analysis = self._analyze_coverage(query, all_passages)

            if analysis.is_complete or not analysis.next_query:
                break

            self._report_progress(AgentProgress(
                phase=AgentPhase.SEARCH,
                iteration=i,
                max_iterations=self.max_iterations,
                message=f"Searching for: {analysis.next_query[:50]}...",
            ))

            # Retrieve more passages with refined query
            new_passages = self._retrieve(analysis.next_query, self.iteration_top_k)
            all_passages.extend(new_passages)
            all_passages = self._deduplicate_passages(all_passages)
            iterations_used = i

        # Phase 4: Synthesis
        self._report_progress(AgentProgress(
            phase=AgentPhase.SYNTHESIZE,
            iteration=iterations_used,
            max_iterations=self.max_iterations,
            message=f"Synthesizing answer from {len(all_passages)} passages...",
        ))

        answer = self._synthesize(query, all_passages)

        # Convert to citations
        citations = [p.to_citation() for p in all_passages]
        cited_suttas = list(set(p.sutta_uid for p in all_passages))

        # Phase 5: Learn (save to memory)
        if self.use_memory and self.memory:
            self._report_progress(AgentProgress(
                phase=AgentPhase.LEARN,
                iteration=iterations_used,
                max_iterations=self.max_iterations,
                message="Saving insights to memory...",
            ))

            self.memory.save(
                query=query,
                answer=answer,
                citations=cited_suttas,
            )

        self._report_progress(AgentProgress(
            phase=AgentPhase.COMPLETE,
            iteration=iterations_used,
            max_iterations=self.max_iterations,
            message="Research complete!",
        ))

        return AgentResponse(
            answer=answer,
            citations=citations,
            from_memory=False,
            iterations_used=iterations_used,
            total_passages_retrieved=len(all_passages),
        )

    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        if self.memory:
            self.memory.clear()

    def get_memory_count(self) -> int:
        """Get the number of stored memory entries."""
        if self.memory:
            return self.memory.get_entry_count()
        return 0
