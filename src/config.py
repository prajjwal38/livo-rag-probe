"""
config.py - Central configuration for the Modular SOTA RAG Pipeline.
Swap models or API keys here without touching any other file.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# API Keys
ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_SPEECH_MODELS: list[str] = ["universal-3-pro", "universal-2"]

# YouTube source videos
VIDEOS: list[dict] = [
    {
        "id": "video_1",
        "url": "https://www.youtube.com/watch?v=aircAruvnKk",
        "title": "But what is a Neural Network?",
        "channel": "3Blue1Brown",
        "language": "English",
    },
    {
        "id": "video_2",
        "url": "https://www.youtube.com/watch?v=wjZofJX0v4M",
        "title": "Transformers, the tech behind LLMs",
        "channel": "3Blue1Brown",
        "language": "English",
    },
    {
        "id": "video_3",
        "url": "https://www.youtube.com/watch?v=fHF22Wxuyw4",
        "title": "What is Deep Learning?",
        "channel": "CampusX",
        "language": "Hindi",
    },
    {
        "id": "video_4",
        "url": "https://www.youtube.com/watch?v=C6YtPJxNULA",
        "title": "All About ML & Deep Learning",
        "channel": "CodeWithHarry",
        "language": "Hindi",
    },
]

# Chunking
CHUNK_SIZE: int = 300
CHUNK_OVERLAP: int = 40

# Embedding model
EMBEDDING_MODEL: str = "google/embeddinggemma-300m"
EMBEDDING_PREFIX_DOC: str = "Retrieval-document: "
EMBEDDING_PREFIX_QUERY: str = "Retrieval-query: "

# Reranker model
RERANKER_MODEL: str = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_TOP_N: int = 3
RETRIEVAL_TOP_K: int = 10

# Relevance guardrail thresholds (Option B — pipeline-level hard cutoff).
# If the top-1 retrieved chunk fails EITHER threshold, the query is considered
# out-of-domain and LLM generation is skipped entirely to prevent hallucination.
#
# RETRIEVAL_MIN_RERANK_SCORE  — Qwen reranker's yes/no probability for the
#   best candidate chunk. Valid range: 0.0–1.0. All in-domain questions in the
#   evaluation report scored ≥ 0.62; set conservatively at 0.20 to avoid
#   refusing real (but tricky) questions.
#
# RETRIEVAL_MIN_RRF_SCORE  — Reciprocal Rank Fusion score of the best chunk.
#   This captures BM25+vector agreement: irrelevant queries produce lower RRF
#   because the two search legs disagree on what to return. Typical RRF values
#   for top-1 results range ~0.005–0.033; 0.008 is a safe lower boundary.
RETRIEVAL_MIN_RERANK_SCORE: float = 0.20
RETRIEVAL_MIN_RRF_SCORE: float = 0.008

# Answer generation model
ANSWER_LLM_MODEL: str = os.getenv("ANSWER_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ANSWER_MAX_CONTEXTS: int = 3
ANSWER_MAX_NEW_TOKENS: int = 450
ANSWER_TEMPERATURE: float = 0.2
ANSWER_DO_SAMPLE: bool = False

# ChromaDB
CHROMA_PERSIST_DIR: str = "./chroma_store"
CHROMA_COLLECTION_NAME: str = "rag_livo_ai"

# Audio download
AUDIO_OUTPUT_DIR: str = "./audio_cache"
