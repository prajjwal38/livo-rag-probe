from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from . import config

logger = logging.getLogger(__name__)

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def _timestamp_to_seconds(timestamp: str | None) -> int | None:
    if not timestamp:
        return None

    parts = timestamp.strip().split(":")
    if not parts or not all(part.isdigit() for part in parts):
        return None

    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def _append_timestamp(url: str, seconds: int | None) -> str:
    if not url or seconds is None:
        return url

    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["t"] = f"{seconds}s"
    return urlunparse(parsed._replace(query=urlencode(query)))


def _extract_source_file_id(path: Path) -> str:
    suffix = "_transcript"
    stem = path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


@lru_cache(maxsize=1)
def _video_lookup() -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for video in config.VIDEOS:
        entry = {
            "video_id": video["id"],
            "youtube_url": video["url"],
            "title": video["title"],
            "channel": video["channel"],
            "language": video["language"],
        }
        keys = {
            _normalize(video["id"]),
            _normalize(video["title"]),
            _normalize(f"{video['channel']}::{video['title']}"),
        }
        for key in keys:
            if key:
                lookup.setdefault(key, entry)
    return lookup


@lru_cache(maxsize=1)
def _chunk_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    cache_dir = Path("llm_chunks")
    if not cache_dir.exists():
        return catalog

    video_lookup = _video_lookup()
    for file_path in sorted(cache_dir.glob("*.json")):
        source_file_id = _extract_source_file_id(file_path)
        fallback_video = video_lookup.get(_normalize(source_file_id), {})

        try:
            with open(file_path, "r", encoding="utf-8") as file_obj:
                chunks = json.load(file_obj)
        except Exception as exc:
            logger.warning("[source_metadata] Failed to load %s: %s", file_path, exc)
            continue

        if not isinstance(chunks, list):
            continue

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if not text:
                continue

            youtube_url = chunk.get("youtube_url") or fallback_video.get("youtube_url")
            catalog[text] = {
                "source_file_id": source_file_id,
                "youtube_url": youtube_url,
                "video_id": chunk.get("video_id"),
                "title": chunk.get("title"),
                "channel": chunk.get("channel"),
                "language": chunk.get("language"),
                "timestamp": chunk.get("timestamp"),
                "start_ms": chunk.get("start_ms"),
                "end_ms": chunk.get("end_ms"),
            }
    return catalog


def resolve_youtube_url(metadata: dict[str, Any], text: str | None = None) -> str | None:
    if metadata.get("youtube_url"):
        return str(metadata["youtube_url"])

    if text:
        catalog_match = _chunk_catalog().get(text)
        if catalog_match and catalog_match.get("youtube_url"):
            return str(catalog_match["youtube_url"])

    video_lookup = _video_lookup()
    for raw_key in (
        metadata.get("source_file_id"),
        metadata.get("video_id"),
        metadata.get("title"),
        f"{metadata.get('channel', '')}::{metadata.get('title', '')}",
    ):
        normalized = _normalize(raw_key)
        if normalized and normalized in video_lookup:
            return video_lookup[normalized]["youtube_url"]

    metadata_title = _normalize(metadata.get("title"))
    metadata_channel = _normalize(metadata.get("channel"))
    if metadata_title:
        for video in config.VIDEOS:
            video_title = _normalize(video["title"])
            video_channel = _normalize(video["channel"])
            if metadata_channel and metadata_channel != video_channel:
                continue
            if metadata_title in video_title or video_title in metadata_title:
                return video["url"]

    video_id = str(metadata.get("video_id") or "").strip()
    if _YOUTUBE_ID_RE.fullmatch(video_id):
        return f"https://www.youtube.com/watch?v={video_id}"

    return None


def enrich_hit_metadata(text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(metadata)
    catalog_match = _chunk_catalog().get(text, {})

    for key in (
        "source_file_id",
        "youtube_url",
        "title",
        "channel",
        "language",
        "timestamp",
        "video_id",
        "start_ms",
        "end_ms",
    ):
        if enriched.get(key) in (None, "") and catalog_match.get(key) not in (None, ""):
            enriched[key] = catalog_match[key]

    youtube_url = resolve_youtube_url(enriched, text=text)
    if youtube_url:
        enriched["youtube_url"] = youtube_url

    start_ms = _coerce_int(enriched.get("start_ms"))
    timestamp_seconds = start_ms // 1000 if start_ms is not None else _timestamp_to_seconds(
        enriched.get("timestamp")
    )
    if timestamp_seconds is not None:
        enriched["timestamp_seconds"] = timestamp_seconds

    if youtube_url and timestamp_seconds is not None:
        enriched["timestamp_url"] = _append_timestamp(youtube_url, timestamp_seconds)

    return enriched
