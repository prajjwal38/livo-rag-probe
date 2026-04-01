# livo-rag-probe

## What this is
A RAG pipeline over 4 ML/AI video transcripts (English + Hinglish).  
Built to test **retrieval quality**, not just retrieval speed.

---

## Architecture

```
YouTube URL
    │
    ▼
[Phase 2 — Ingestion]
yt-dlp download → AssemblyAI transcription → Semantic chunking (llm_chunks/)
    │
    ▼
[Phase 3 — Indexing]
Gemma-300M embeddings → ChromaDB (vector) + BM25 (keyword)
    │
    ▼
[Phase 4 — Retrieval]
User query
    ├── Vector search (ChromaDB)  ─┐
    └── Keyword search (BM25)     ─┴→ RRF fusion → Qwen3 Reranker
                                                         │
                                                         ▼
                                              Top-N ranked chunks + confidence scores
    │
    ▼
[Confidence Gate]  ◄── RETRIEVAL_MIN_RERANK_SCORE & RETRIEVAL_MIN_RRF_SCORE
    ├── BELOW threshold → ✗ Immediate refusal (no LLM call)
    └── ABOVE threshold → ✓ Proceed
                              │
                              ▼
                    [Phase 5 — Answer Generation]
                    Qwen2.5-Instruct → Structured JSON answer
                    { answer, summary_points, source timestamps, follow_up_note }
```

---

## The confidence gate

The pipeline uses a **dual-threshold hard cutoff** to eliminate silent hallucination before the LLM is ever called.

**Example A — Irrelevant query (gate blocks it):**
```
Query:  "What is the best recipe for making biryani at home?"
        │
        ├─ top-1 rerank_score : 0.18  ← below threshold (0.20)
        └─ top-1 rrf_score    : 0.006 ← below threshold (0.008)
        │
        ▼
REFUSAL (instant, no LLM cost):
"I'm sorry, but I don't have enough relevant information in my knowledge
 base to answer your question."
```

**Example B — Relevant query (gate passes it):**
```
Query:  "How are GPT-3's 175B parameters organized?"
        │
        ├─ top-1 rerank_score : 0.9987 ✓
        └─ top-1 rrf_score    : 0.0310 ✓
        │
        ▼
ANSWER (Qwen2.5-Instruct, grounded in retrieved chunk):
"GPT-3's 175 billion weights are organized into ~28,000 matrices across
 8 functional categories: Embedding, Key, Query, Value, Output,
 Up-projection, Down-projection, and Unembedding."
Source: video_2 @ 08:53  ← timestamped YouTube link returned
```

---

## Why this matters for production

Most RAG systems retrieve the top-K chunks regardless of relevance, feeding garbage context to an LLM that will confidently hallucinate an answer — with no signal to the user that the source material was completely unrelated.  
This pipeline blocks that failure mode at the retrieval layer, ensuring every generated answer is either grounded in indexed content or explicitly refused.

---

## Eval dataset

[`Golden_QA_dataset.json`](./Golden_QA_dataset.json) — hand-authored QA pairs with:
- ✅ Pinpointed relevant questions tied to exact video timestamps
- ❌ Irrelevant (out-of-domain) questions the pipeline must refuse

Results tracked in [`evaluation_report.md`](./evaluation_report.md).

---


## Project Structure
```
RAG for LIVO.ai/
├── main.py                 # Orchestrator — run the full pipeline
├── requirements.txt        # Python dependencies
├── .env.example            # Template for your secrets
├── src/
│   ├── config.py           # All constants: URLs, model names, paths, etc.
│   ├── interfaces.py       # 3 modular functions (swap models here)
│   ├── ingestion.py        # Phase 2: yt-dlp download + AssemblyAI + chunking
│   ├── vector_store.py     # Phase 3: ChromaDB + BM25 indexing
│   ├── retrieval.py        # Phase 4: Hybrid search + RRF + Qwen3 reranker
│   ├── qa_generator.py     # Phase 5: 5 Adversarial QA pairs
│   └── evaluate.py         # Phase 6: RAG Evaluation against manual dataset
├── audio_cache/            # (auto-created) yt-dlp downloads
└── chroma_store/           # (auto-created) ChromaDB persistent storage
```

## Setup

### 1. Prerequisites
- Python 3.10+
- `ffmpeg` installed and on your PATH (required by yt-dlp for audio conversion)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure secrets
```bash
cp .env.example .env
# Edit .env and add your AssemblyAI API key
```

## Running the Pipeline

### Full run (all phases)
```bash
python main.py
```

### Skip already-done phases (saves time on re-runs)
```bash
# Skip both download+transcription and indexing (only regenerate QA)
python main.py --skip-ingestion --skip-indexing

# Skip only download (transcribe + reindex)
python main.py --skip-ingestion

# Skip retrieval check during QA generation (faster, offline mode)
python main.py --no-retrieval-check
```

### Interactive grounded QA
```bash
# Answer questions with LLM synthesis over the top retrieved chunks
python main.py --skip-ingestion --skip-indexing --interactive

# Force answers into a target language
python main.py --skip-ingestion --skip-indexing --interactive --target-language Hindi

# Also print raw retrieved chunks for debugging
python main.py --skip-ingestion --skip-indexing --interactive --show-raw-hits
```

### Phase 6: Evaluation
Run the RAG pipeline against a manually curated golden QNA dataset to measure performance on specific pinpointed queries and irrelevance handling.

```bash
# Run evaluation using a manual dataset
python main.py --skip-ingestion --skip-indexing --evaluate Golden_QA_dataset.json

# Specify a custom report path
python main.py --skip-ingestion --skip-indexing --evaluate Golden_QA_dataset.json --eval-report my_results.md
```

## Relevance Guardrail (Hallucination Prevention)

The pipeline now includes a **Dual-Threshold Relevance Gate** (Option B) that protects against hallucinations for out-of-domain queries. If the best retrieved chunk fails either a semantic confidence check or a search-agreement check, the system returns a polite refusal instead of calling the LLM.

- **Configurable Thresholds** (in `src/config.py`):
    - `RETRIEVAL_MIN_RERANK_SCORE`: Minimum confidence from the Qwen reranker (e.g., `0.20`).
    - `RETRIEVAL_MIN_RRF_SCORE`: Minimum search-agreement score from RRF (e.g., `0.008`).

### VS Code Interactive Window
Open `interactive_window.py` and run the `# %%` cells. Edit `question`,
`target_language`, and `top_k`, then execute the answer cell. The output now
includes:
- a structured grounded answer
- multilingual response control
- timestamped YouTube links for cited chunks

## The 4 Modular Interface Functions (`src/interfaces.py`)

| Function | Current Provider | How to Swap |
|---|---|---|
| `clean_transcript(source)` | AssemblyAI | Replace body with any ASR API |
| `get_embeddings(text)` | google/gemma-embedding-exp-03-07 | Swap model name in `config.py` |
| `rerank_results(query, contexts)` | Qwen3-Reranker-2B | Replace body with any cross-encoder |
| `generate_structured_answer(query, contexts, target_language)` | Qwen2.5 Instruct | Swap model name in `config.py` |

## The 5 Adversarial QA Pairs

| # | Type | Videos | Failure Mode |
|---|---|---|---|
| 1 | Cross-lingual trap | video_1 + video_3 | Type A |
| 2 | Cross-lingual trap | video_2 + video_4 | Type A |
| 3 | Timestamp Sniper | video_1 | Type B |
| 4 | Timestamp Sniper | video_2 | Type B |
| 5 | Multi-hop | video_1 | Type C |

**Failure Mode Legend:**
- **Type A** — Cross-lingual confusion (returns English chunk for a Hindi concept or vice versa)
- **Type B** — Semantic near-miss (retrieves a related but wrong-timestamp chunk)
- **Type C** — Partial retrieval (only one of the two required chunks is found)

## Output Format (`adversarial_qa.json`)
```json
{
  "question": "...",
  "answer": "...",
  "source": {"video": "...", "timestamp": "...", "language": "..."},
  "distractor_risk": "What a wrong retrieval would look like",
  "failure_mode": "Type A | B | C"
}
```
