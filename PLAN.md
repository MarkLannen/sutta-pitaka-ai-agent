# Sutta Pitaka AI Agent - Implementation Plan

## Current State (as of Dec 22, 2024)

### What's Working
- ChromaDB vector store with 1,043 chunks from Majjhima Nikaya (152 suttas)
- SuttaCentral API integration for fetching suttas
- Document processor for chunking suttas
- **Iterative AI Agent** with multi-pass search and gap analysis
- **Agent Memory** - persists learned insights in `agent_wisdom` collection
- **Pali Dictionary** - 142K entries from SuttaCentral API
- **Pali Term Search** - search for Pali terms across cached suttas
- Streamlit UI with Chat and Pali Tools tabs
- Multi-model support (Ollama, Anthropic, Google, OpenAI)
- Dynamic retrieval: 5 chunks for local models, 20 for cloud models
- Progress indicators during research phases

### What Was Done (Dec 22)
1. **Phase 3 & 6: Iterative Agent with Memory** ✅
   - Created `src/agent/iterative_agent.py` with `SuttaPitakaAgent`
   - Created `src/agent/memory.py` with `AgentMemory` class
   - Multi-pass search with LLM-driven gap analysis
   - Persists synthesized answers in `agent_wisdom` ChromaDB collection
   - Recalls relevant prior research before new searches

2. **Phase 4: Analysis & Synthesis Prompts** ✅
   - Gap analysis prompt identifies missing information
   - Synthesis prompt constructs thematic, well-cited responses

3. **Phase 5: UI Updates** ✅
   - Progress indicator showing research phases (Recall → Search → Analyze → Synthesize → Learn)
   - "Clear Memory" button in sidebar
   - Memory status display

4. **Consolidated Agents**
   - Removed `ai_agent.py` (simple RAG agent)
   - `SuttaPitakaAgent` now handles all queries (iterative by default)

5. **Phase 9: Pali Dictionary & Term Search** ✅ (NEW)
   - Created `src/dictionary/pali_dictionary.py` - dictionary lookup
   - Created `src/dictionary/pali_search.py` - term search across suttas
   - UI: "Pali Tools" tab with Term Search and Dictionary sub-tabs
   - Example: "nirodha" → 231 occurrences across 49 suttas

### What Was Done (Dec 21)
1. Renamed "RAG Agent" to "AI Agent" throughout UI and codebase
2. Fixed Gemini model IDs (removed `models/` prefix)
3. Made top_k dynamic based on model provider
4. Added `.env` file support for API keys

### What Was Done (Dec 19)
1. Renamed project from `pali-canon-rag-agent` to `sutta-pitaka-rag-agent`
2. Updated all code references (collection name, class names, UI text)
3. Re-ingested MN data with new collection name `sutta_pitaka`
4. Updated system prompt to require inline citations
5. Made prompt stricter to only reference retrieved context

---

## Completed Phases

### Phase 2: Increase Retrieval Capacity ✅ DONE

- ✅ Made top_k dynamic based on model type (5 for local, 20 for cloud)
- ✅ Added `SIMILARITY_TOP_K_LOCAL` and `SIMILARITY_TOP_K_CLOUD` config options
- ✅ `get_top_k_for_model()` helper function selects appropriate value

### Phase 3: Build Iterative AI Agent with Memory ✅ DONE

**Files created:** `src/agent/iterative_agent.py`, `src/agent/memory.py`

- ✅ `SuttaPitakaAgent` class with multi-pass search
- ✅ Gap analysis to identify missing information
- ✅ Deduplication of retrieved chunks
- ✅ Progress callbacks for UI updates

### Phase 4: Analysis & Synthesis Prompts ✅ DONE

- ✅ Gap analysis prompt in `iterative_agent.py`
- ✅ Synthesis prompt for scholarly output with citations

### Phase 5: Update UI ✅ DONE

- ✅ Progress indicator showing research phases
- ✅ "Clear Agent Memory" button in sidebar
- ✅ Memory status display
- ✅ "Recalled from memory" indicator

### Phase 6: Long-Term Memory ✅ DONE

**File created:** `src/agent/memory.py`

- ✅ Dedicated ChromaDB collection `agent_wisdom`
- ✅ `save()` stores synthesized answers with metadata
- ✅ `recall()` searches for similar prior queries

### Phase 7: Rename Project ✅ DONE

- ✅ Updated UI titles and descriptions
- ✅ Consolidated to single `SuttaPitakaAgent` class
- ✅ Updated docstrings

### Phase 9: Pali Dictionary & Term Search ✅ DONE (NEW)

**Files created:** `src/dictionary/pali_dictionary.py`, `src/dictionary/pali_search.py`

- ✅ Pali-English dictionary with 142K entries from SuttaCentral API
- ✅ Pali term search across cached suttas
- ✅ Occurrence counts by sutta
- ✅ UI: "Pali Tools" tab with Term Search and Dictionary

---

## Remaining Phases

### Phase 1: Extend Ingestion (Ingest Full Sutta Pitaka)

**Files to modify:** `src/config.py`, `ingest.py`, `src/ingestion/suttacentral.py`

1. Add support for all nikayas (DN, SN, AN, KN)
2. Implement **Recursive Character Splitting** in `processor.py` for formulaic text
3. Add batch ingestion command: `python ingest.py --all`
4. Estimate: ~5,400 suttas total, likely 30,000-50,000 chunks

### Phase 8: Search-Only Mode (For Counting Queries)

**Note:** Partially addressed by Pali Term Search, but could be extended for English semantic search.

For questions like "list all suttas about Y":

1. Performs exhaustive vector search (high top_k, e.g., 100-500)
2. Returns list of matching suttas without LLM synthesis
3. Groups results by sutta (not by chunk)
4. Shows relevance scores and snippet previews

---

## File Structure (Current)

```
sutta-pitaka-ai-agent/
├── app.py                      # Streamlit UI (Chat + Pali Tools tabs)
├── ingest.py                   # CLI for ingestion
├── .env                        # API keys
├── PLAN.md                     # This file
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── iterative_agent.py  # Main agent with memory
│   │   └── memory.py           # Agent wisdom persistence
│   ├── dictionary/
│   │   ├── __init__.py
│   │   ├── pali_dictionary.py  # Pali-English dictionary
│   │   └── pali_search.py      # Pali term search
│   ├── config.py               # Settings
│   ├── indexing/
│   │   └── vector_store.py     # ChromaDB
│   ├── ingestion/
│   │   ├── suttacentral.py     # API client
│   │   └── processor.py        # Document chunking
│   └── retrieval/
│       └── query_engine.py     # Query engine
├── chroma_db/                  # Vector database
│   ├── sutta_pitaka/           # Main sutta embeddings
│   └── agent_wisdom/           # Learned insights
└── cache/
    ├── suttas/                 # Cached sutta JSON files
    └── pali_dictionary.json    # Cached dictionary
```

---

## Quick Start Commands

```bash
# Navigate to project
cd /Users/markl1/Documents/AI-Agents/sutta-pitaka-ai-agent

# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run app.py

# Check current ingestion status
python ingest.py --status

# Stop streamlit when done
pkill -f streamlit
```

---

## Next Steps (Priority Order)

### 1. Full Sutta Pitaka Ingestion (Phase 1)
- Extend SuttaCentral API client for DN, SN, AN, KN
- Handle complex nested structures (SN, AN)
- Batch ingest all ~5,400+ suttas
- Pali term search will automatically work on new data

### 2. Enhanced Search Mode (Phase 8)
- Add English semantic search with high top_k
- Group results by sutta
- UI toggle for "Research" vs "Search" mode

### 3. Future Enhancements
- Metadata filtering (by nikaya, topic tags)
- Export research results
- Compare passages across suttas
