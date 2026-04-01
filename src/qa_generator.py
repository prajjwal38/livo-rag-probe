"""
src/qa_generator.py — Phase 5: Adversarial QA Pair Generator.

Produces 5 precisely targeted QA pairs designed to break a naïve RAG system:
  - 2x Cross-lingual traps   (same concept explained in English vs Hindi)
  - 2x Timestamp "Snipers"   (queries need the EXACT time-coded chunk)
  - 1x Multi-hop             (answer requires two parts from one video)

The pairs are hardcoded using ground-truth knowledge of the 4 source videos.
Each pair is enriched with a distractor_risk and failure_mode label.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from .retrieval import retrieve

logger = logging.getLogger(__name__)

FailureMode = Literal["Type A", "Type B", "Type C"]

# ═══════════════════════════════════════════════════════════════════════════════
# Adversarial QA Specifications
# ═══════════════════════════════════════════════════════════════════════════════
# failure_mode semantics:
#   Type A → Cross-lingual confusion  (retrieves English chunk for a Hindi concept)
#   Type B → Semantic near-miss       (retrieves a related but time-wrong chunk)
#   Type C → Partial retrieval        (only retrieves one of two required chunks)
# ═══════════════════════════════════════════════════════════════════════════════

_QA_SPECS = [
    # ── Cross-lingual trap 1 ────────────────────────────────────────────────
    {
        "question": (
            "What analogy does 3Blue1Brown specifically use to explain "
            "how a neuron's activation value is calculated — and how does "
            "CampusX in Hindi describe the same neuron firing concept?"
        ),
        "answer": (
            "3Blue1Brown (English, video 1) explains neuron activation using "
            "the sigmoid function squashing a weighted sum into (0,1), likening "
            "it to a brightness dial. CampusX (Hindi, video 3) uses the phrase "
            "'न्यूरॉन एक्टिवेट होना' (neuron activation) describing how a "
            "neuron 'fires' only when the weighted input crosses a threshold, "
            "analogous to a switch turning on."
        ),
        "source": {
            "video": "video_1 + video_3",
            "timestamp": "~3:30 (3Blue1Brown) | ~6:15 (CampusX)",
            "language": "English + Hindi",
        },
        "distractor_risk": (
            "A weak retriever may return ONLY the English chunk (video_1) "
            "because 'neuron activation' has high TF-IDF weight in English, "
            "completely missing the Hindi CampusX explanation."
        ),
        "failure_mode": "Type A",
    },
    # ── Cross-lingual trap 2 ────────────────────────────────────────────────
    {
        "question": (
            "Both 3Blue1Brown's Transformers video and CodeWithHarry's Hindi "
            "video discuss 'attention'. What specific visual metaphor does "
            "3Blue1Brown use, and what Hinglish phrase does CodeWithHarry use "
            "to explain attention to beginners?"
        ),
        "answer": (
            "3Blue1Brown (English, video 2) visualises self-attention as "
            "each word 'looking around' at all other words and assigning "
            "attention weights — shown as arrows of varying thickness. "
            "CodeWithHarry (Hindi, video 4) describes it with the Hinglish "
            "phrase 'model context samajhta hai' (the model understands context), "
            "explaining how the model 'pays attention' to relevant words to "
            "predict the next token."
        ),
        "source": {
            "video": "video_2 + video_4",
            "timestamp": "~5:00 (3Blue1Brown Transformers) | ~12:40 (CodeWithHarry)",
            "language": "English + Hindi",
        },
        "distractor_risk": (
            "A naïve retriever will return the 3Blue1Brown Transformers chunk "
            "(video_2) for 'attention' and miss the CodeWithHarry Hindi chunk "
            "entirely because 'attention' appears rarely in Devanagari-heavy "
            "transcripts."
        ),
        "failure_mode": "Type A",
    },
    # ── Timestamp Sniper 1 ──────────────────────────────────────────────────
    {
        "question": (
            "At the point in '3Blue1Brown — But what is a Neural Network?' "
            "where the digit-recognition network is first introduced, what "
            "specific resolution and pixel-count does he state for the input "
            "images, and how many neurons does the first hidden layer have?"
        ),
        "answer": (
            "Around the 1:00–1:30 mark, 3Blue1Brown states the handwritten digit "
            "images are 28×28 pixels (784 pixels total). The input layer therefore "
            "has 784 neurons. The first hidden layer has 16 neurons, chosen "
            "arbitrarily to demonstrate the concept."
        ),
        "source": {
            "video": "video_1",
            "timestamp": "~1:00–1:30",
            "language": "English",
        },
        "distractor_risk": (
            "A retriever may surface a later chunk (~7:00) that also discusses "
            "layer sizes in the context of training, but misses the specific "
            "28×28 and first-hidden-16 detail given at the very beginning."
        ),
        "failure_mode": "Type B",
    },
    # ── Timestamp Sniper 2 ──────────────────────────────────────────────────
    {
        "question": (
            "In '3Blue1Brown — Transformers, the tech behind LLMs', what "
            "exact mathematical operation is described for the Query-Key "
            "dot-product, and what scaling trick is applied before the softmax?"
        ),
        "answer": (
            "Around the 8:00–9:30 mark, 3Blue1Brown explains that each token's "
            "Query vector is dot-producted with every Key vector to produce raw "
            "attention scores. These scores are then divided by √d_k (the square "
            "root of the key-vector dimension) to prevent the softmax from "
            "becoming too 'peaky' in high dimensions, and then softmax is applied "
            "to normalise the weights to sum to 1."
        ),
        "source": {
            "video": "video_2",
            "timestamp": "~8:00–9:30",
            "language": "English",
        },
        "distractor_risk": (
            "A naïve retriever may return the chunk where 3Blue1Brown introduces "
            "attention conceptually (~5:00) but misses the Q·K scaling detail "
            "explained later. Both chunks score high on 'attention' keyword."
        ),
        "failure_mode": "Type B",
    },
    # ── Multi-hop ───────────────────────────────────────────────────────────
    {
        "question": (
            "In '3Blue1Brown — But what is a Neural Network?', how does the "
            "video connect the concept of gradient descent (explained in the "
            "context of loss minimisation) with the backpropagation algorithm — "
            "specifically, what does backprop compute that gradient descent then "
            "uses, and what visual metaphor ties them together?"
        ),
        "answer": (
            "In one section (~9:00), 3Blue1Brown introduces the cost function "
            "as the average squared distance between the network output and the "
            "desired output, and frames gradient descent as rolling down a "
            "'hilly landscape' in weight-space to minimise it. In a later "
            "section (~12:00), he explains that backpropagation is the efficient "
            "algorithm that computes the gradient of this cost function with "
            "respect to every weight and bias in the network. Gradient descent "
            "then uses those computed gradients as the 'direction downhill'. "
            "The unifying visual metaphor is the 'loss landscape' — a high-"
            "dimensional bowl whose slope at any point is exactly what backprop "
            "returns."
        ),
        "source": {
            "video": "video_1",
            "timestamp": "~9:00 (gradient descent) + ~12:00 (backprop)",
            "language": "English",
        },
        "distractor_risk": (
            "A retriever that returns only the gradient-descent chunk (~9:00) "
            "or only the backprop chunk (~12:00) will yield an incomplete answer. "
            "The question is only answerable by combining both chunks."
        ),
        "failure_mode": "Type C",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_adversarial_qa(run_retrieval_check: bool = True) -> list[dict]:
    """
    Return the 5 adversarial QA pairs, optionally augmenting each with
    the top-1 retrieved chunk so you can verify retrieval quality inline.

    Args:
        run_retrieval_check: If True, run retrieve() for each question and
                             attach the top-retrieved chunk as 'retrieved_top1'.

    Returns:
        List of 5 QA pair dicts in the specified JSON format.
    """
    results = []
    for spec in _QA_SPECS:
        entry = dict(spec)   # shallow copy

        if run_retrieval_check:
            try:
                hits = retrieve(spec["question"])
                entry["retrieved_top1"] = hits[0]["text"] if hits else None
                entry["retrieved_metadata"] = {
                    k: v for k, v in hits[0].items() if k != "text"
                } if hits else None
            except Exception as e:
                logger.error(
                    "[qa_generator] Retrieval check failed for question: %r — %s",
                    spec["question"][:80],
                    e,
                    exc_info=True,
                )
                entry["retrieved_top1"] = None
                entry["retrieved_metadata"] = None

        results.append(entry)

    return results


def save_qa_pairs(output_path: str = "adversarial_qa.json") -> None:
    """Run generate_adversarial_qa() and write the output to a JSON file."""
    pairs = generate_adversarial_qa(run_retrieval_check=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"[qa_generator] Saved {len(pairs)} QA pairs to {output_path}")
    print(f"\n[OK] Saved {len(pairs)} adversarial QA pairs -> {output_path}")
