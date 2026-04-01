"""
src/interfaces.py - The modular core.

These functions are the only model/provider touchpoints for the rest of the
pipeline. To swap a model/provider, only edit the body of the relevant
function.

  clean_transcript(source) -> list[dict]
  get_embeddings(text) -> list[float]
  rerank_results(query, contexts) -> list[str]
  generate_structured_answer(query, contexts, target_language) -> dict
"""

from __future__ import annotations

import logging
import json
import re
import shutil
from typing import Any

import assemblyai as aai
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config

logger = logging.getLogger(__name__)

_embedding_model: SentenceTransformer | None = None
_reranker_tokenizer: Any | None = None
_reranker_model: Any | None = None
_answer_tokenizer: Any | None = None
_answer_model: Any | None = None
_runtime_device_logged = False


def _build_transcription_config() -> aai.TranscriptionConfig:
    """
    Build an AssemblyAI transcription config that is compatible with the
    installed SDK and the current API contract.
    """
    speech_models = list(config.ASSEMBLYAI_SPEECH_MODELS)
    if not speech_models:
        raise ValueError("ASSEMBLYAI_SPEECH_MODELS must contain at least one model.")

    return aai.TranscriptionConfig(
        speech_models=speech_models,
        language_detection=True,
        punctuate=True,
        format_text=True,
    )


def _get_runtime_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_runtime_dtype() -> torch.dtype | None:
    if not torch.cuda.is_available():
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _log_runtime_device_once() -> None:
    global _runtime_device_logged
    if _runtime_device_logged:
        return

    device = _get_runtime_device()
    if device == "cuda":
        logger.info(
            "[runtime] Using CUDA device: %s | dtype=%s",
            torch.cuda.get_device_name(0),
            _get_runtime_dtype(),
        )
    else:
        if "+cpu" in torch.__version__ and shutil.which("nvidia-smi"):
            logger.warning(
                "[runtime] NVIDIA GPU detected by the system, but PyTorch is CPU-only (%s). "
                "Install a CUDA-enabled torch build to move embeddings, reranking, and answer generation to GPU.",
                torch.__version__,
            )
        else:
            logger.info("[runtime] Using CPU runtime.")

    _runtime_device_logged = True


def clean_transcript(source: str) -> list[dict[str, int | str]]:
    """
    Transcribe and clean an audio file using AssemblyAI.

    Args:
        source: Local file path to an audio file.

    Returns:
        Sentence-level transcript entries with timestamps.
    """
    aai.settings.api_key = config.ASSEMBLYAI_API_KEY
    if not aai.settings.api_key:
        raise EnvironmentError("ASSEMBLYAI_API_KEY is not set in your .env file.")

    transcriber = aai.Transcriber()
    transcription_config = _build_transcription_config()

    logger.info("[clean_transcript] Uploading -> %s", source)
    transcript = transcriber.transcribe(source, config=transcription_config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI error: {transcript.error}")

    sentences: list[dict[str, int | str]] = []
    for sentence in transcript.get_sentences():
        sentences.append(
            {
                "text": sentence.text,
                "start_ms": sentence.start,
                "end_ms": sentence.end,
            }
        )

    logger.info("[clean_transcript] Done -> %s sentences extracted.", len(sentences))
    return sentences


def _load_embedding_model() -> SentenceTransformer:
    """Load the embedding model once."""
    global _embedding_model
    if _embedding_model is None:
        _log_runtime_device_once()
        logger.info("[get_embeddings] Loading model: %s", config.EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            trust_remote_code=True,
            device=_get_runtime_device(),
        )
        logger.info("[get_embeddings] Model ready on %s.", _embedding_model.device)
    return _embedding_model


def get_embeddings(text: str, is_query: bool = False) -> list[float]:
    """
    Generate a dense embedding for a piece of text.

    Query and document prefixes are kept separate so the embedding model uses
    the right instruction format for retrieval.
    """
    model = _load_embedding_model()
    prefix = (
        config.EMBEDDING_PREFIX_QUERY if is_query else config.EMBEDDING_PREFIX_DOC
    )
    prefixed_text = prefix + text
    embedding = model.encode(prefixed_text, normalize_embeddings=True)
    return embedding.tolist()


def _load_reranker() -> tuple[Any, Any]:
    """Load the Qwen reranker once. Guards against partial-load where tokenizer
    succeeds but model fails, which would silently return (tokenizer, None)."""
    global _reranker_tokenizer, _reranker_model
    if _reranker_tokenizer is None or _reranker_model is None:
        _log_runtime_device_once()
        logger.info("[rerank_results] Loading reranker: %s", config.RERANKER_MODEL)
        model_kwargs: dict[str, Any] = {}
        runtime_dtype = _get_runtime_dtype()
        if runtime_dtype is not None:
            model_kwargs["torch_dtype"] = runtime_dtype
            model_kwargs["low_cpu_mem_usage"] = True
        _reranker_tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL)
        _reranker_model = AutoModelForCausalLM.from_pretrained(
            config.RERANKER_MODEL,
            **model_kwargs,
        )
        if torch.cuda.is_available():
            _reranker_model = _reranker_model.to(_get_runtime_device())
        _reranker_model.eval()
        logger.info("[rerank_results] Reranker ready on %s.", _reranker_model.device)
    return _reranker_tokenizer, _reranker_model


def _score_pair(query: str, context: str, tokenizer: Any, model: Any) -> float:
    """
    Score a single query/context pair using the Qwen reranker.
    """
    prompt = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or '
        '"no".<|im_end|>\n'
        "<|im_start|>user\n"
        "<Instruct>: Given a query, determine if the following document is relevant.\n"
        f"<Query>: {query}\n"
        f"<Document>: {context}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    yes_logit = logits[yes_id].item()
    no_logit = logits[no_id].item()

    exp_yes = torch.exp(torch.tensor(yes_logit))
    exp_no = torch.exp(torch.tensor(no_logit))
    return (exp_yes / (exp_yes + exp_no)).item()


def rerank_results(query: str, contexts: list[str]) -> list[tuple[str, float]]:
    """
    Re-order candidate contexts by relevance to the query.
    Returns: list of (context_text, score)
    """
    tokenizer, model = _load_reranker()

    scored: list[tuple[float, str]] = []
    for ctx in contexts:
        score = _score_pair(query, ctx, tokenizer, model)
        scored.append((score, ctx))
        logger.debug("[rerank_results] score=%.4f ctx[:60]=%r", score, ctx[:60])

    scored.sort(key=lambda item: item[0], reverse=True)
    top_contexts = [(ctx, score) for score, ctx in scored[: config.RERANKER_TOP_N]]
    logger.info(
        "[rerank_results] Kept top %s of %s contexts.",
        len(top_contexts),
        len(contexts),
    )
    return top_contexts


def _load_answer_model() -> tuple[Any, Any]:
    global _answer_tokenizer, _answer_model
    if _answer_tokenizer is None or _answer_model is None:
        _log_runtime_device_once()
        logger.info(
            "[generate_structured_answer] Loading answer model: %s",
            config.ANSWER_LLM_MODEL,
        )
        model_kwargs: dict[str, Any] = {}
        runtime_dtype = _get_runtime_dtype()
        if runtime_dtype is not None:
            model_kwargs["torch_dtype"] = runtime_dtype
            model_kwargs["low_cpu_mem_usage"] = True
        _answer_tokenizer = AutoTokenizer.from_pretrained(config.ANSWER_LLM_MODEL)
        _answer_model = AutoModelForCausalLM.from_pretrained(
            config.ANSWER_LLM_MODEL,
            **model_kwargs,
        )
        if _answer_tokenizer.pad_token_id is None and _answer_tokenizer.eos_token_id is not None:
            _answer_tokenizer.pad_token = _answer_tokenizer.eos_token
        if torch.cuda.is_available():
            _answer_model = _answer_model.to(_get_runtime_device())
        _answer_model.eval()
        logger.info(
            "[generate_structured_answer] Answer model ready on %s.",
            _answer_model.device,
        )
    return _answer_tokenizer, _answer_model


def _extract_json_object(text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced_match.group(1) if fenced_match else text

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")

    return json.loads(candidate[start : end + 1])


def _build_answer_messages(
    query: str,
    contexts: list[dict[str, Any]],
    target_language: str | None,
) -> list[dict[str, str]]:
    context_blocks = []
    for context in contexts:
        context_blocks.append(
            "\n".join(
                [
                    f"Source ID: {context['source_id']}",
                    f"Video Title: {context.get('title', 'Unknown')}",
                    f"Channel: {context.get('channel', 'Unknown')}",
                    f"Language: {context.get('language', 'Unknown')}",
                    f"Timestamp: {context.get('timestamp', 'Unknown')}",
                    f"Timestamp URL: {context.get('timestamp_url', 'Unavailable')}",
                    f"Reranker Confidence: {context.get('rerank_score', 'N/A')}",
                    f"Base RRF Score: {context.get('rrf_score', 'N/A')}",
                    f"Topic Summary: {context.get('topic_summary', 'N/A')}",
                    f"Chunk Text: {context['text']}",
                ]
            )
        )

    language_instruction = (
        f"Respond in {target_language}."
        if target_language
        else "Respond in the same language as the user's question unless the user explicitly asks for another language."
    )

    system_message = (
        "You are a grounded multilingual RAG answering assistant. "
        "Use only the provided sources. If the sources are insufficient, say so clearly. "
        "Do not invent citations, timestamps, or facts. "
        "Return only valid JSON."
    )
    user_message = (
        f"Question: {query}\n\n"
        f"{language_instruction}\n\n"
        "You must synthesize the retrieved chunks into a clean structured answer.\n"
        "Return exactly this JSON schema:\n"
        "{\n"
        '  "answer_language": "<language used in the answer>",\n'
        '  "answer": "<2-5 sentence grounded answer>",\n'
        '  "summary_points": ["<short bullet>", "<short bullet>"],\n'
        '  "used_source_ids": ["SRC_1", "SRC_2"],\n'
        '  "follow_up_note": "<short note about uncertainty or cross-lingual nuance>"\n'
        "}\n\n"
        "Rules:\n"
        "- Use only source IDs that actually appear in the provided sources.\n"
        "- Prefer concise summary_points.\n"
        "- If the user asks a multilingual question, it is okay to answer in mixed language.\n"
        "- If evidence is weak, say that in follow_up_note.\n\n"
        "Sources:\n"
        + "\n\n".join(context_blocks)
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _generate_chat_completion(messages: list[dict[str, str]]) -> str:
    tokenizer, model = _load_answer_model()
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}" for message in messages
        )
        prompt += "\n\nASSISTANT:\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": config.ANSWER_MAX_NEW_TOKENS,
        "do_sample": config.ANSWER_DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.ANSWER_DO_SAMPLE:
        generation_kwargs["temperature"] = config.ANSWER_TEMPERATURE

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def generate_structured_answer(
    query: str,
    contexts: list[dict[str, Any]],
    target_language: str | None = None,
) -> dict[str, Any]:
    """
    Generate a grounded structured answer from retrieved contexts.
    """
    if not contexts:
        return {
            "answer_language": target_language or "same-as-query",
            "answer": "I could not find any retrieved context for this question.",
            "summary_points": [],
            "used_source_ids": [],
            "follow_up_note": "No evidence was available from retrieval.",
        }

    messages = _build_answer_messages(query, contexts, target_language)
    raw_response = _generate_chat_completion(messages)
    parsed = _extract_json_object(raw_response)

    used_source_ids = parsed.get("used_source_ids")
    if not isinstance(used_source_ids, list):
        parsed["used_source_ids"] = []
    else:
        parsed["used_source_ids"] = [
            str(source_id)
            for source_id in used_source_ids
            if isinstance(source_id, str)
        ]

    summary_points = parsed.get("summary_points")
    if not isinstance(summary_points, list):
        parsed["summary_points"] = []
    else:
        parsed["summary_points"] = [
            str(item) for item in summary_points if isinstance(item, str)
        ]

    parsed.setdefault("answer_language", target_language or "same-as-query")
    parsed.setdefault("answer", "")
    parsed.setdefault("follow_up_note", "")
    return parsed
