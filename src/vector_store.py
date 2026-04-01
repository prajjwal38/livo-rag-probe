"""
src/vector_store.py - Phase 3: embedding, ChromaDB, and BM25 indexing.

Public API:
  build_index(chunks) -> store all chunk embeddings in ChromaDB and a BM25 index
  get_chroma_collection() -> return the persistent ChromaDB collection
  get_bm25_index() -> return the saved BM25 index plus raw chunks
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from . import config
from .interfaces import get_embeddings

logger = logging.getLogger(__name__)

_BM25_INDEX_PATH = Path(config.CHROMA_PERSIST_DIR) / "bm25_index.pkl"
_BM25_CHUNKS_PATH = Path(config.CHROMA_PERSIST_DIR) / "bm25_chunks.json"


def _get_chroma_client() -> chromadb.PersistentClient:
    """Create or reuse a persistent ChromaDB client."""
    Path(config.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_chroma_collection() -> Any:
    """Return the persistent ChromaDB collection, creating it if needed."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


def _save_bm25(index: BM25Okapi, chunks: list[dict[str, Any]]) -> None:
    with open(_BM25_INDEX_PATH, "wb") as file_obj:
        pickle.dump(index, file_obj)
    with open(_BM25_CHUNKS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(chunks, file_obj, ensure_ascii=False, indent=2)
    logger.info("[vector_store] BM25 index saved.")


def get_bm25_index() -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Load the saved BM25 index and its corresponding chunks."""
    if not _BM25_INDEX_PATH.exists() or not _BM25_CHUNKS_PATH.exists():
        raise FileNotFoundError("BM25 index not found. Run build_index() first.")

    with open(_BM25_INDEX_PATH, "rb") as file_obj:
        index = pickle.load(file_obj)
    with open(_BM25_CHUNKS_PATH, "r", encoding="utf-8") as file_obj:
        chunks = json.load(file_obj)

    logger.info("[vector_store] BM25 index loaded (%s chunks).", len(chunks))
    return index, chunks


def _has_saved_indexes(collection: Any) -> bool:
    """
    Return True only when ALL three conditions are met:
      - ChromaDB collection has at least one vector
      - BM25 index pickle exists on disk
      - BM25 chunks JSON exists on disk
    """
    chroma_ok = collection.count() > 0
    bm25_ok = _BM25_INDEX_PATH.exists() and _BM25_CHUNKS_PATH.exists()
    if chroma_ok and not bm25_ok:
        logger.warning(
            "[build_index] ChromaDB has %s vectors but BM25 index is missing "
            "— will rebuild both indexes.",
            collection.count(),
        )
    return chroma_ok and bm25_ok


def build_index(chunks: list[dict[str, Any]]) -> None:
    """
    Embed all chunks and store them in ChromaDB. Also build and persist a
    parallel BM25 index over the same chunks.
    """
    collection = get_chroma_collection()

    if not chunks:
        if _has_saved_indexes(collection):
            logger.info(
                "[build_index] No in-memory chunks provided. "
                "Both ChromaDB and BM25 indexes already exist — reusing them."
            )
            return
        raise ValueError(
            "No chunks provided to build_index() and indexes are incomplete or missing. "
            "Re-run without --skip-ingestion so chunks are loaded from the transcript cache, "
            "or delete chroma_store/ and re-run the full pipeline."
        )

    existing_count = collection.count()
    if existing_count >= len(chunks):
        logger.info(
            "[build_index] ChromaDB already contains %s vectors (%s chunks provided). Skipping re-indexing.",
            existing_count,
            len(chunks),
        )
        if not (_BM25_INDEX_PATH.exists() and _BM25_CHUNKS_PATH.exists()):
            tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
            _save_bm25(BM25Okapi(tokenized_corpus), chunks)
        return

    logger.info("[build_index] Embedding and indexing %s chunks...", len(chunks))

    batch_size = 50
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        ids = [f"chunk_{batch_start + index}" for index in range(len(batch))]
        embeddings = [get_embeddings(chunk["text"], is_query=False) for chunk in batch]
        documents = [chunk["text"] for chunk in batch]
        metadatas = []
        for chunk in batch:
            # Base exact metadata
            meta = {
                "video_id": chunk["video_id"],
                "title": chunk["title"],
                "channel": chunk["channel"],
                "language": chunk["language"],
                "timestamp": chunk["timestamp"],
            }
            # Add semantic AI Studio metadata if it exists
            if "topic_summary" in chunk:
                meta["topic_summary"] = chunk["topic_summary"]
            if "youtube_url" in chunk:
                meta["youtube_url"] = chunk["youtube_url"]
            if "source_file_id" in chunk:
                meta["source_file_id"] = chunk["source_file_id"]
            if "start_ms" in chunk:
                meta["start_ms"] = int(chunk["start_ms"]) if str(chunk["start_ms"]).isdigit() else chunk["start_ms"]
            if "end_ms" in chunk:
                meta["end_ms"] = int(chunk["end_ms"]) if str(chunk["end_ms"]).isdigit() else chunk["end_ms"]
            if "key_entities" in chunk and isinstance(chunk["key_entities"], list):
                # ChromaDB metadata doesn't allow nested lists, so we join them 
                meta["key_entities"] = ", ".join(chunk["key_entities"])
                
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "[build_index] Upserted batch %s (%s chunks).",
            batch_start // batch_size + 1,
            len(batch),
        )

    logger.info("[build_index] ChromaDB now holds %s vectors.", collection.count())

    tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    _save_bm25(bm25, chunks)
    logger.info("[build_index] Indexing complete (ChromaDB + BM25).")
