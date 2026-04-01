"""
src/retrieval.py — Phase 4: Hybrid Search (Vector + BM25) + RRF + Reranker.

Public API:
  retrieve(query, top_k) → list[dict]  (reranked, best chunks first)
"""

from __future__ import annotations

import logging
from typing import Any

from . import config
from .interfaces import get_embeddings, rerank_results
from .source_metadata import enrich_hit_metadata
from .vector_store import get_chroma_collection, get_bm25_index

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════════════════════

def _rrf_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Each document receives a score: sum(1 / (k + rank)) across all lists.
    Higher score = more relevant.

    Args:
        ranked_lists: Each sub-list is a list of document IDs ordered by
                      descending relevance for one search method.
        k:            Constant that controls the rank penalty (~60 is standard).

    Returns:
        A single merged list of document IDs sorted by fused score (best first).
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    sorted_docs = sorted(scores, key=scores.__getitem__, reverse=True)
    return [(doc, scores[doc]) for doc in sorted_docs]


# ══════════════════════════════════════════════════════════════════════════════
# Individual search legs
# ══════════════════════════════════════════════════════════════════════════════

def _vector_search(query: str, n_results: int) -> list[tuple[str, dict]]:
    """
    Dense vector search via ChromaDB.

    Returns:
        List of (chunk_text, metadata) tuples, best-match first.
    """
    query_embedding = get_embeddings(query, is_query=True)
    collection = get_chroma_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(docs, metadatas))


def _bm25_search(query: str, n_results: int) -> list[tuple[str, dict]]:
    """
    Sparse BM25 keyword search.

    Returns:
        List of (chunk_text, metadata) tuples, best-match first.
    """
    bm25, chunks = get_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Sort chunk indices by descending BM25 score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [(chunks[i]["text"], {k: v for k, v in chunks[i].items() if k != "text"})
            for i in top_indices]


# ══════════════════════════════════════════════════════════════════════════════
# Public Retrieve Function
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[dict]:
    """
    Full retrieval pipeline:
      1. Run Vector search  → top_k candidates
      2. Run BM25 search    → top_k candidates
      3. RRF fusion         → unified ranked list
      4. Reranker           → final top RERANKER_TOP_N results

    Args:
        query:  The user's natural language question.
        top_k:  Number of candidates from each search leg (default from config).

    Returns:
        List of dicts (best match first):
        [{"text": str, "video_id": str, "title": str, "channel": str,
          "language": str, "timestamp": str}, ...]
    """
    logger.info(f"[retrieve] Query: {query!r}")

    # ── Step 1 & 2: Run both search legs ─────────────────────────────────────
    vector_hits = _vector_search(query, top_k)
    bm25_hits   = _bm25_search(query, top_k)

    # Build lookup: text → metadata (texts are used as IDs for RRF)
    text_to_meta: dict[str, dict] = {}
    for text, meta in vector_hits + bm25_hits:
        text_to_meta[text] = meta      # later entry overwrites, both are same meta

    vector_ranked = [t for t, _ in vector_hits]
    bm25_ranked   = [t for t, _ in bm25_hits]

    # ── Step 3: RRF Fusion ───────────────────────────────────────────────────
    fused_order = _rrf_fusion([vector_ranked, bm25_ranked]) # returns list of (text, rrf_score)
    fused_texts = [t for t, _ in fused_order][:top_k]
    fused_score_map = dict(fused_order)
    logger.info(f"[retrieve] RRF produced {len(fused_texts)} candidates.")

    # ── Step 4: Rerank ───────────────────────────────────────────────────────
    top_items = rerank_results(query, fused_texts) # returns list of (text, cross_score)

    # Assemble final result dicts
    final_results = []
    for text, cross_score in top_items:
        meta = enrich_hit_metadata(text, text_to_meta.get(text, {}))
        final_results.append({
            "text": text, 
            "rerank_score": cross_score,
            "rrf_score": fused_score_map.get(text, 0.0),
            **meta
        })

    logger.info(f"[retrieve] Returning {len(final_results)} reranked results.")
    return final_results
