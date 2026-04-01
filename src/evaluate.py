"""
src/evaluate.py — Phase 6: RAG Evaluation against a Manual Golden Dataset.

Expected dataset schema (JSON array):
[
  {
    "id": "q1",
    "question": "What is backpropagation?",
    "expected_answer": "Backpropagation is...",
    "expected_video_id": "video_1",         // null for irrelevant queries
    "expected_timestamp_hint": "~12:00",    // optional, used for logging only
    "is_irrelevant": false                  // true → RAG MUST refuse / return no evidence
  },
  ...
]

Run via:
  python main.py --evaluate path/to/golden_dataset.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

from .answer_generation import answer_query

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
REFUSAL_PHRASES = [
    "no relevant chunks",
    "no evidence",
    "could not find",
    "i don't know",
    "i do not know",
    "insufficient",
    "unavailable",
    "not available",
    "no information",
    "cannot answer",
    "can't answer",
]

# ── data structures ────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    item_id: str
    question: str
    is_irrelevant: bool
    expected_video_id: str | None
    expected_timestamp_hint: str | None
    expected_answer: str | None

    # pipeline outputs
    generated_answer: str = ""
    retrieved_video_ids: list[str] = field(default_factory=list)
    top1_video_id: str | None = None
    top1_timestamp: str | None = None
    top1_rerank_score: float | None = None
    num_chunks_retrieved: int = 0
    latency_s: float = 0.0

    # scores
    irrelevance_pass: bool | None = None     # only for is_irrelevant=True items
    pinpoint_hit: bool | None = None         # only for is_irrelevant=False items
    correctness_score: int | None = None     # 0-2 rubric
    faithfulness_score: int | None = None    # 0-2 rubric
    judge_reasoning: str = ""

    @property
    def passed(self) -> bool:
        if self.is_irrelevant:
            return self.irrelevance_pass is True
        return self.pinpoint_hit is True


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_dataset(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Golden dataset not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a non-empty JSON array.")
    return data


def _is_refusal(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def _judge_answer(
    question: str,
    expected_answer: str,
    generated_answer: str,
    retrieved_chunks: list[dict],
) -> dict[str, Any]:
    """
    Lightweight rule-based judge used when no external LLM API is available.
    Returns correctness (0-2) and faithfulness (0-2) scores plus reasoning.

    Scoring rubric:
      Correctness:
        2 — generated answer shares at least 40% of key bigrams with expected
        1 — shares 15-40%
        0 — below 15%

      Faithfulness:
        2 — every sentence in the generated answer has supporting text in at
            least one retrieved chunk (substring match heuristic)
        1 — partial support found
        0 — no support found
    """

    def _bigrams(text: str) -> set[tuple[str, str]]:
        tokens = text.lower().split()
        return set(zip(tokens, tokens[1:]))

    # Correctness via bigram overlap
    exp_bi = _bigrams(expected_answer)
    gen_bi = _bigrams(generated_answer)
    if exp_bi:
        overlap = len(exp_bi & gen_bi) / len(exp_bi)
    else:
        overlap = 1.0 if not gen_bi else 0.0

    if overlap >= 0.40:
        correctness = 2
        c_reason = f"Strong bigram overlap ({overlap:.0%})"
    elif overlap >= 0.15:
        correctness = 1
        c_reason = f"Partial bigram overlap ({overlap:.0%})"
    else:
        correctness = 0
        c_reason = f"Low bigram overlap ({overlap:.0%})"

    # Faithfulness via substring check
    chunk_corpus = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
    sentences = [s.strip() for s in generated_answer.split(".") if len(s.strip()) > 20]
    if not sentences:
        faithfulness = 0
        f_reason = "No verifiable sentences in generated answer."
    else:
        supported = sum(
            1 for s in sentences
            if any(word in chunk_corpus for word in s.lower().split() if len(word) > 5)
        )
        ratio = supported / len(sentences)
        if ratio >= 0.75:
            faithfulness = 2
            f_reason = f"High grounding ({supported}/{len(sentences)} sentences supported)"
        elif ratio >= 0.40:
            faithfulness = 1
            f_reason = f"Partial grounding ({supported}/{len(sentences)} sentences supported)"
        else:
            faithfulness = 0
            f_reason = f"Poor grounding ({supported}/{len(sentences)} sentences supported)"

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "reasoning": f"Correctness: {c_reason}. Faithfulness: {f_reason}.",
    }


def _run_one(item: dict) -> EvalResult:
    question          = item["question"]
    is_irrelevant     = bool(item.get("is_irrelevant", False))
    expected_video_id = item.get("expected_video_id")
    timestamp_hint    = item.get("expected_timestamp_hint")
    expected_answer   = item.get("expected_answer", "")
    item_id           = str(item.get("id", question[:30]))

    result = EvalResult(
        item_id=item_id,
        question=question,
        is_irrelevant=is_irrelevant,
        expected_video_id=expected_video_id,
        expected_timestamp_hint=timestamp_hint,
        expected_answer=expected_answer,
    )

    t0 = time.perf_counter()
    try:
        pipeline_out = answer_query(query=question)
    except Exception as exc:
        logger.error("[evaluate] Pipeline failure for %r: %s", question[:60], exc)
        result.generated_answer = f"PIPELINE ERROR: {exc}"
        result.latency_s = time.perf_counter() - t0
        return result

    result.latency_s = time.perf_counter() - t0

    result.generated_answer   = pipeline_out.get("answer", "")
    raw_hits                  = pipeline_out.get("raw_retrieval", [])
    result.num_chunks_retrieved = len(raw_hits)
    result.retrieved_video_ids  = [h.get("video_id", "") for h in raw_hits]

    if raw_hits:
        top1 = raw_hits[0]
        result.top1_video_id    = top1.get("video_id")
        result.top1_timestamp   = top1.get("timestamp")
        result.top1_rerank_score = top1.get("rerank_score")

    # ── Irrelevance check ──
    if is_irrelevant:
        result.irrelevance_pass = _is_refusal(result.generated_answer) or result.num_chunks_retrieved == 0

    # ── Pinpoint retrieval hit ──
    else:
        if expected_video_id:
            result.pinpoint_hit = result.top1_video_id == expected_video_id
        else:
            # No video ID to compare against → check if any answer was generated
            result.pinpoint_hit = bool(result.generated_answer)

        # ── Judge answer quality ──
        if expected_answer:
            judgment = _judge_answer(
                question=question,
                expected_answer=expected_answer,
                generated_answer=result.generated_answer,
                retrieved_chunks=raw_hits,
            )
            result.correctness_score  = judgment["correctness"]
            result.faithfulness_score = judgment["faithfulness"]
            result.judge_reasoning    = judgment["reasoning"]

    return result


# ── reporting ──────────────────────────────────────────────────────────────────

def _render_report(results: list[EvalResult], dataset_path: str) -> str:
    total      = len(results)
    irrelevant = [r for r in results if r.is_irrelevant]
    relevant   = [r for r in results if not r.is_irrelevant]

    irr_pass   = sum(1 for r in irrelevant if r.irrelevance_pass)
    pin_pass   = sum(1 for r in relevant   if r.pinpoint_hit)
    avg_latency = sum(r.latency_s for r in results) / total if total else 0

    lines: list[str] = [
        "# RAG Evaluation Report",
        "",
        f"**Dataset:** `{dataset_path}`  ",
        f"**Total questions:** {total}  ",
        f"**Average latency:** {avg_latency:.2f}s per question",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Irrelevance Handling (refuse rate) | {irr_pass}/{len(irrelevant)} "
        f"({'N/A' if not irrelevant else f'{irr_pass/len(irrelevant):.0%}'}) |",
        f"| Pinpoint Retrieval Accuracy (top-1 hit) | {pin_pass}/{len(relevant)} "
        f"({'N/A' if not relevant else f'{pin_pass/len(relevant):.0%}'}) |",
    ]

    if relevant:
        judged = [r for r in relevant if r.correctness_score is not None]
        if judged:
            avg_c = sum(r.correctness_score for r in judged) / len(judged)  # type: ignore[arg-type]
            avg_f = sum(r.faithfulness_score for r in judged) / len(judged)  # type: ignore[arg-type]
            lines += [
                f"| Avg. Correctness Score (0-2) | {avg_c:.2f} |",
                f"| Avg. Faithfulness Score (0-2) | {avg_f:.2f} |",
            ]

    lines += ["", "---", ""]

    # ── Irrelevance detailed table ──
    if irrelevant:
        lines += [
            "## Irrelevance Handling",
            "",
            "| ID | Question | Refused? | Generated Answer (truncated) |",
            "| --- | --- | --- | --- |",
        ]
        for r in irrelevant:
            status  = "✅ PASS" if r.irrelevance_pass else "❌ FAIL"
            snippet = textwrap.shorten(r.generated_answer, 80)
            lines.append(f"| `{r.item_id}` | {r.question[:60]} | {status} | {snippet} |")
        lines.append("")

    # ── Relevant detailed table ──
    if relevant:
        lines += [
            "## Pinpoint Retrieval & Generation Quality",
            "",
            "| ID | Question | Expected Video | Top-1 Hit | Pinpoint? | C | F | Latency |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in relevant:
            pin_status = "✅" if r.pinpoint_hit else "❌"
            c_str = str(r.correctness_score) if r.correctness_score is not None else "—"
            f_str = str(r.faithfulness_score) if r.faithfulness_score is not None else "—"
            ev    = r.expected_video_id or "—"
            t1    = r.top1_video_id or "—"
            lines.append(
                f"| `{r.item_id}` | {r.question[:55]} | `{ev}` | `{t1}` "
                f"| {pin_status} | {c_str} | {f_str} | {r.latency_s:.1f}s |"
            )
        lines.append("")

    # ── Per-question detail ──
    lines += ["---", "", "## Per-Question Details", ""]
    for r in results:
        tag = "[IRRELEVANT]" if r.is_irrelevant else "[RELEVANT]"
        passed_str = "PASS" if r.passed else "FAIL"
        lines += [
            f"### {tag} `{r.item_id}` — {passed_str}",
            f"**Question:** {r.question}",
            "",
        ]
        if r.is_irrelevant:
            lines += [
                f"**Refused correctly:** {r.irrelevance_pass}",
                f"**Generated answer:** {textwrap.shorten(r.generated_answer, 200)}",
            ]
        else:
            lines += [
                f"**Expected video:** `{r.expected_video_id or 'any'}`  "
                f"**Timestamp hint:** `{r.expected_timestamp_hint or 'N/A'}`",
                f"**Top-1 retrieved:** video=`{r.top1_video_id}` | "
                f"timestamp=`{r.top1_timestamp}` | rerank={r.top1_rerank_score:.4f}"
                if r.top1_rerank_score is not None else
                f"**Top-1 retrieved:** video=`{r.top1_video_id}` | timestamp=`{r.top1_timestamp}`",
                f"**Pinpoint hit:** {r.pinpoint_hit}",
                "",
                f"**Expected answer:** {textwrap.shorten(r.expected_answer or '', 200)}",
                f"**Generated answer:** {textwrap.shorten(r.generated_answer, 200)}",
            ]
            if r.judge_reasoning:
                lines.append(f"**Judge:** {r.judge_reasoning}")
        lines += ["", "---", ""]

    return "\n".join(lines)


# ── public entry point ─────────────────────────────────────────────────────────

def run_evaluation(dataset_path: str, report_path: str = "evaluation_report.md") -> None:
    """
    Run the full evaluation suite against a manual golden dataset and write a
    Markdown report.

    Args:
        dataset_path: Path to the manual QNA JSON file.
        report_path:  Output path for the Markdown evaluation report.
    """
    logger.info("[evaluate] Loading dataset: %s", dataset_path)
    items = _load_dataset(dataset_path)
    logger.info("[evaluate] %d questions loaded.", len(items))

    results: list[EvalResult] = []
    for idx, item in enumerate(items, start=1):
        q_short = str(item.get("question", ""))[:60]
        logger.info("[evaluate] (%d/%d) %s", idx, len(items), q_short)
        results.append(_run_one(item))

    # ── Summary to stdout ──
    total      = len(results)
    irrelevant = [r for r in results if r.is_irrelevant]
    relevant   = [r for r in results if not r.is_irrelevant]
    irr_pass   = sum(1 for r in irrelevant if r.irrelevance_pass)
    pin_pass   = sum(1 for r in relevant   if r.pinpoint_hit)

    print(f"\n{'='*60}")
    print(f"  RAG EVALUATION COMPLETE — {total} questions")
    print(f"{'='*60}")
    print(f"  Irrelevance Handling : {irr_pass}/{len(irrelevant)} passed"
          f"  ({irr_pass/len(irrelevant):.0%})" if irrelevant else
          "  Irrelevance Handling : (no irrelevant items)")
    print(f"  Pinpoint Accuracy    : {pin_pass}/{len(relevant)} passed"
          f"  ({pin_pass/len(relevant):.0%})" if relevant else
          "  Pinpoint Accuracy    : (no relevant items)")
    print(f"{'='*60}\n")

    # ── Write report ──
    report_md = _render_report(results, dataset_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[OK] Full report saved to: {report_path}")
