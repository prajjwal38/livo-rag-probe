# 🕵️‍♂️ Runtime Analysis — Run #2 (2026-03-27 ~02:27 IST)

## ✅ Phase 2 (Ingestion) — FULLY COMPLETE

All four videos have been **successfully downloaded AND transcribed** this run.

**Evidence (`audio_cache/` directory):**
| File | Status |
|---|---|
| `video_1.mp3` | ✅ Already existed |
| `video_1_transcript.json` (48 KB) | ✅ **Newly cached this run** |
| `video_2.mp3` | ✅ Already existed |
| `video_2_transcript.json` (71 KB) | ✅ **Newly cached this run** |
| `video_3.mp3` | ✅ Already existed |
| `video_3_transcript.json` (200 KB) | ✅ **Newly cached this run** |
| `video_4.mp3` | ✅ **Newly downloaded this run** |
| `video_4_transcript.json` (51 KB) | ✅ **Newly cached this run** |

> **Next time you run, Phase 2 is effectively free** — all `.mp3` and `_transcript.json` files exist, so no downloads and zero AssemblyAI API calls will be made.

---

## 🛑 Phase 3 (Indexing) — CRASHED

The pipeline failed at the very start of Phase 3, when `build_index()` tried to call `get_embeddings()` for the first chunk. This triggered the **lazy-load of the Gemma embedding model** (`google/gemma-embedding-exp-03-07`) from HuggingFace.

### Root Cause: 🔐 HuggingFace Gated/Private Model
The crash log contained this message:
```
If this is a private repository, make sure to pass a token having permission
to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
```

**Explanation:**  
`google/gemma-embedding-exp-03-07` is a **gated model** on HuggingFace. You must:
1. Accept the model's license agreement on the HuggingFace website.
2. Authenticate your local machine with your HuggingFace API token.

Without this, `SentenceTransformer` cannot download or load the model weights, and Phase 3 immediately fails.

**Evidence (`chroma_store/` directory):**
- Only `chroma.sqlite3` exists (empty/fresh) — **no vectors were stored**, no `bm25_index.pkl`, no `bm25_chunks.json`.

---

## 🛠️ How to Fix It (Two Options)

### Option A — Authenticate with HuggingFace (Keep Gemma)
1. Go to [https://huggingface.co/google/gemma-embedding-exp-03-07](https://huggingface.co/google/gemma-embedding-exp-03-07) and accept the access request.
2. Get your HuggingFace token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Log in from the terminal:
   ```powershell
   huggingface-cli login
   ```
   Paste your token when prompted. This saves a credential that `SentenceTransformer` will use automatically.
4. Re-run `python main.py --skip-ingestion` (ingestion is done — no need to redo it!).

### Option B — Swap to a Freely Available Embedding Model (No Auth Needed)
Change `EMBEDDING_MODEL` in `src/config.py` to a public model, for example:
```python
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
# or
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
```
These are smaller but require zero authentication and download instantly.
Then re-run with `--skip-ingestion`.

---

## ⚡ What to Run Next

Since all transcripts are cached, you can skip Phase 2 entirely:
```powershell
python main.py --skip-ingestion
```
This jumps straight to Phase 3 (embedding + ChromaDB) and Phase 5 (QA generation).
