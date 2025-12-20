Here is the updated **PLAN.md** with the integrated "Learning" and "Memory" phases, following your existing format and structure.

---

# Sutta Pitaka AI Agent - Implementation Plan

## Current State (as of Dec 19, 2024)

### What's Working

* ChromaDB vector store with 1,043 chunks from Majjhima Nikaya (152 suttas)
* SuttaCentral API integration for fetching suttas
* Document processor for chunking suttas
* Basic query engine (retrieves top-k similar chunks)
* Streamlit UI at `app.py`
* Multi-model support (Ollama, Anthropic, Google, OpenAI)

### What Was Done Today

1. Renamed project from `pali-canon-rag-agent` to `sutta-pitaka-rag-agent`
2. Updated all code references (collection name, class names, UI text)
3. Re-ingested MN data with new collection name `sutta_pitaka`
4. Updated system prompt to require inline citations
5. Made prompt stricter to only reference retrieved context (no LLM hallucination)

### Current Limitation

The current system only retrieves top 5 chunks per query. For comprehensive scholarly queries like "describe all ways Buddha taught about karma," this is insufficient. Furthermore, the agent has no "memory"—it performs the same expensive search every time a question is repeated.

---

## Goal: Reflective Agentic Iterative Search

Build an AI agent that iteratively searches the entire Sutta Pitaka until it has gathered comprehensive information, and **persists that knowledge** in a long-term memory store to "learn" over time.

### How It Should Work

1. User asks: "Give me a synopsis of all the ways the Buddha described karma"
2. **Recall Phase:** Agent checks a "Wisdom" collection to see if it has researched this before.
3. **Search Phase:** If no memory exists, agent performs initial broad search (top 20-30 results).
4. **Iterative Phase:** Agent analyzes results, identifies gaps, and generates refined queries.
5. **Synthesis Phase:** Agent aggregates findings into a comprehensive, well-cited answer.
6. **Learning Phase:** Agent saves the final synthesis back into its memory for future use.

---

## Implementation Plan

### Phase 1: Extend Ingestion (Ingest Full Sutta Pitaka)

**Files to modify:** `src/config.py`, `ingest.py`, `src/ingestion/suttacentral.py`

1. Add support for all nikayas (DN, SN, AN, KN).
2. Implement **Recursive Character Splitting** in `processor.py` to better handle formulaic/repetitive text in SN/AN.
3. Add batch ingestion command: `python ingest.py --all`

### Phase 2: Increase Retrieval Capacity

**Files to modify:** `src/config.py`, `src/retrieval/query_engine.py`

1. Increase `SIMILARITY_TOP_K` from 5 to 20-30.
2. Add metadata filtering (by nikaya, by topic tags if available).

### Phase 3: Build Iterative AI Agent with Memory

**New file:** `src/agent/iterative_agent.py`

```python
class SuttaPitakaAgent:
    """
    AI Agent that iteratively searches and remembers findings.
    """
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.memory = AgentMemory(vector_store) # Persistence layer
        self.max_iterations = 5

    def search(self, query: str) -> AgentResponse:
        # 1. Check if we've already 'learned' this
        wisdom = self.memory.recall(query)
        if wisdom: return wisdom

        # 2. Iterative search loop
        all_passages = []
        for i in range(self.max_iterations):
            new_passages = self._retrieve(query)
            all_passages.extend(new_passages)
            analysis = self._analyze_coverage(query, all_passages)
            if analysis.is_complete: break
            query = analysis.next_query

        # 3. Synthesize and save to memory
        response = self._synthesize(query, all_passages)
        self.memory.save(query, response)
        return response

```

### Phase 4: Analysis & Synthesis Prompts

1. **Analysis Prompt (Gap Identification):**
* "Identify what is missing from these passages to fully answer [Query]. Generate a refined search term."


2. **Synthesis Prompt (Scholarly Output):**
* "Construct a thematic response using ONLY the retrieved passages. Cite every sutta."



### Phase 5: Update UI

1. Add progress indicator: "Searching... Step 2/5... Recalling from memory..."
2. Add a "Clear Agent Memory" button in the sidebar for debugging/resetting.

### Phase 6: Long-Term Memory (The "Learning" Layer)

**New file:** `src/agent/memory.py`

1. Create a dedicated ChromaDB collection `agent_wisdom`.
2. Implement `save_learned_insight()`: Stores the final LLM synthesis with metadata (original query, suttas cited).
3. Implement `recall_relevant_wisdom()`: Performs a similarity search on the `agent_wisdom` collection before starting a new search.

---

## Technical Considerations

### Context Window

* With 30-50 passages, use a **Map-Reduce** synthesis strategy to avoid hitting token limits.

### Deduplication

* Track `chunk_id` during iterations to ensure the agent doesn't process the same text twice in one session.

### Persistence

* Ensure the `agent_wisdom` collection is persisted to disk so "learning" survives application restarts.

---

## Next Session: Suggested Starting Point

1. Start by increasing `SIMILARITY_TOP_K` to 20 in `src/config.py`.
2. Implement the `AgentMemory` class in `src/agent/memory.py` to create the `agent_wisdom` collection.
3. Test a "Search -> Save -> Recall" loop with a single query.
