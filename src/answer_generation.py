from __future__ import annotations

import logging
from typing import Any

from . import config
from .interfaces import generate_structured_answer
from .retrieval import retrieve

logger = logging.getLogger(__name__)


def _format_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "N/A"


def _prepare_contexts(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for index, hit in enumerate(hits[: config.ANSWER_MAX_CONTEXTS], start=1):
        contexts.append(
            {
                "source_id": f"SRC_{index}",
                "text": hit["text"],
                "title": hit.get("title"),
                "channel": hit.get("channel"),
                "language": hit.get("language"),
                "timestamp": hit.get("timestamp"),
                "timestamp_url": hit.get("timestamp_url"),
                "topic_summary": hit.get("topic_summary"),
                "video_id": hit.get("video_id"),
                "rerank_score": hit.get("rerank_score"),
                "rrf_score": hit.get("rrf_score"),
            }
        )
    return contexts


def _fallback_answer(
    query: str,
    contexts: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    error: Exception,
) -> dict[str, Any]:
    logger.error("[answer_query] Falling back after generation failure: %s", error)
    answer = (
        "Structured generation was unavailable, so this response is a retrieval-only fallback. "
        "Use the linked sources below to inspect the most relevant chunks."
    )
    if contexts:
        answer += f" Top evidence was retrieved for: {query}"

    return {
        "query": query,
        "answer_language": "fallback",
        "answer": answer,
        "summary_points": [
            context["topic_summary"]
            for context in contexts
            if context.get("topic_summary")
        ][:3],
        "follow_up_note": f"Generation error: {error}",
        "used_source_ids": [context["source_id"] for context in contexts],
        "sources": contexts,
        "raw_retrieval": hits,
    }


def answer_query(
    query: str,
    top_k: int = config.RETRIEVAL_TOP_K,
    target_language: str | None = None,
) -> dict[str, Any]:
    hits = retrieve(query, top_k=top_k)
    contexts = _prepare_contexts(hits)

    if not contexts:
        return {
            "query": query,
            "answer_language": target_language or "same-as-query",
            "answer": "No relevant chunks were retrieved for this question.",
            "summary_points": [],
            "follow_up_note": "Try a more specific question or rebuild the index.",
            "used_source_ids": [],
            "sources": [],
            "raw_retrieval": [],
        }

    # ── Relevance Guardrail (Option B — dual-threshold hard cutoff) ────────────
    # Inspect the single best chunk returned by the reranker. If it fails either
    # the rerank confidence gate OR the RRF agreement gate, the query is treated
    # as out-of-domain and we refuse immediately — no LLM call, no hallucination.
    top_hit = hits[0]
    top_rerank = top_hit.get("rerank_score", 0.0) or 0.0
    top_rrf    = top_hit.get("rrf_score", 0.0) or 0.0

    if top_rerank < config.RETRIEVAL_MIN_RERANK_SCORE or top_rrf < config.RETRIEVAL_MIN_RRF_SCORE:
        logger.info(
            "[answer_query] Relevance guardrail triggered — rerank=%.4f (min=%.2f), "
            "rrf=%.4f (min=%.4f). Returning refusal.",
            top_rerank, config.RETRIEVAL_MIN_RERANK_SCORE,
            top_rrf,    config.RETRIEVAL_MIN_RRF_SCORE,
        )
        return {
            "query": query,
            "answer_language": target_language or "same-as-query",
            "answer": (
                "I'm sorry, but I don't have enough relevant information in my "
                "knowledge base to answer your question. Please ask something "
                "related to the indexed video content."
            ),
            "summary_points": [],
            "follow_up_note": (
                f"Relevance guardrail: top-1 rerank={top_rerank:.4f} "
                f"(threshold={config.RETRIEVAL_MIN_RERANK_SCORE}), "
                f"rrf={top_rrf:.4f} "
                f"(threshold={config.RETRIEVAL_MIN_RRF_SCORE})."
            ),
            "used_source_ids": [],
            "sources": [],
            "raw_retrieval": hits,  # kept so evaluate.py can still log the scores
        }
    # ──────────────────────────────────────────────────────────────────────────

    try:
        generated = generate_structured_answer(
            query=query,
            contexts=contexts,
            target_language=target_language,
        )
    except Exception as exc:
        return _fallback_answer(query, contexts, hits, exc)

    used_ids = set(generated.get("used_source_ids", []))
    sources = [
        context for context in contexts if not used_ids or context["source_id"] in used_ids
    ]

    return {
        "query": query,
        "answer_language": generated.get("answer_language", target_language or "same-as-query"),
        "answer": generated.get("answer", ""),
        "summary_points": generated.get("summary_points", []),
        "follow_up_note": generated.get("follow_up_note", ""),
        "used_source_ids": sorted(used_ids) if used_ids else [context["source_id"] for context in sources],
        "sources": sources,
        "raw_retrieval": hits,
    }


def render_answer(result: dict[str, Any]) -> str:
    lines = [
        f"Question: {result['query']}",
        f"Answer language: {result.get('answer_language', 'unknown')}",
        "",
        "Answer:",
        result.get("answer", ""),
    ]

    summary_points = result.get("summary_points") or []
    if summary_points:
        lines.extend(["", "Key points:"])
        lines.extend(f"- {point}" for point in summary_points)

    follow_up_note = result.get("follow_up_note")
    if follow_up_note:
        lines.extend(["", "Note:", follow_up_note])

    sources = result.get("sources") or []
    if sources:
        lines.extend(["", "Sources:"])
        for source in sources:
            lines.append(
                f"- {source['source_id']} | {source.get('title', 'Unknown title')} | "
                f"{source.get('timestamp', 'Unknown timestamp')} | "
                f"Reranker Confidence: {_format_score(source.get('rerank_score'))} | "
                f"Base RRF Score: {_format_score(source.get('rrf_score'))} | "
                f"{source.get('timestamp_url', 'URL unavailable')}"
            )

    return "\n".join(lines)
