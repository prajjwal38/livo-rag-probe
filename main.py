"""
main.py — End-to-end orchestrator for the Modular SOTA RAG Pipeline.

Run:
    python main.py [--skip-ingestion] [--skip-indexing] [--no-retrieval-check]

Stages:
  1. Ingestion  -> download audio, transcribe, chunk
  2. Indexing   -> embed + store in ChromaDB + BM25
  3. QA Output  -> generate 5 adversarial QA pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular SOTA RAG Pipeline — Adversarial QA Generator"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip Phase 2 (assumes audio already downloaded and transcribed).",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip Phase 3 (assumes ChromaDB + BM25 already built).",
    )
    parser.add_argument(
        "--no-retrieval-check",
        action="store_true",
        help="Skip live retrieval pass when generating QA pairs.",
    )
    parser.add_argument(
        "--output",
        default="adversarial_qa.json",
        help="Path for the output JSON file (default: adversarial_qa.json).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive grounded QA loop for custom queries (skips Phase 5).",
    )
    parser.add_argument(
        "--target-language",
        default=None,
        help="Optional language override for interactive answers, e.g. Hindi or English.",
    )
    parser.add_argument(
        "--show-raw-hits",
        action="store_true",
        help="Also print the retrieved chunks underneath the structured answer in interactive mode.",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        metavar="DATASET_PATH",
        default=None,
        help="Phase 6: Run evaluation against a manual golden QNA dataset (JSON file path).",
    )
    parser.add_argument(
        "--eval-report",
        type=str,
        default="evaluation_report.md",
        help="Output path for the evaluation Markdown report (default: evaluation_report.md).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ─── Phase 2: Ingestion ──────────────────────────────────────────────────
    chunks: list[dict] = []

    if not args.skip_ingestion:
        logger.info("=" * 60)
        logger.info("PHASE 2 -- Ingestion (download -> transcribe -> chunk)")
        logger.info("=" * 60)
        from src.ingestion import run_ingestion
        chunks = run_ingestion()
        logger.info(f"Ingestion complete: {len(chunks)} chunks ready.")
    else:
        logger.info("Skipping Phase 2 (--skip-ingestion).")

    # ─── Phase 3: Indexing ───────────────────────────────────────────────────
    if not args.skip_indexing:
        logger.info("=" * 60)
        logger.info("PHASE 3 -- Indexing (embed -> ChromaDB + BM25)")
        logger.info("=" * 60)
        if not chunks:
            logger.warning(
                "No chunks in memory; build_index() will use the count-check "
                "to decide whether to re-embed. Ensure ChromaDB exists."
            )
        from src.vector_store import build_index
        build_index(chunks)
        logger.info("Indexing complete.")
    else:
        logger.info("Skipping Phase 3 (--skip-indexing).")

    # ─── Interactive Query Mode ──────────────────────────────────────────────
    if args.interactive:
        logger.info("=" * 60)
        logger.info("Interactive Grounded QA Mode (Type 'quit' or 'exit' to stop)")
        logger.info("=" * 60)
        from src.answer_generation import answer_query, render_answer

        while True:
            try:
                user_query = input("\n\033[94mEnter your question:\033[0m ")
                if user_query.strip().lower() in ("quit", "exit", "q"):
                    break
                if not user_query.strip():
                    continue

                result = answer_query(
                    query=user_query,
                    target_language=args.target_language,
                )

                print(f"\n\033[92mStructured Answer for: {user_query}\033[0m")
                print("-" * 60)
                print(render_answer(result))

                if args.show_raw_hits:
                    print("\nRetrieved chunks:")
                    for i, hit in enumerate(result.get("raw_retrieval", []), 1):
                        print(
                            f"\n[Rank {i}] {hit.get('title', 'Unknown title')} | "
                            f"{hit.get('timestamp', 'Unknown timestamp')}"
                        )
                        print(hit["text"])
                print("-" * 60)
            except KeyboardInterrupt:
                break
        
        logger.info("Exiting interactive mode.")
        return

    # ─── Phase 6: Evaluation ──────────────────────────────────────────────────────
    if args.evaluate:
        logger.info("=" * 60)
        logger.info("PHASE 6 -- Evaluation against golden dataset: %s", args.evaluate)
        logger.info("=" * 60)
        from src.evaluate import run_evaluation
        run_evaluation(dataset_path=args.evaluate, report_path=args.eval_report)
        return

    # ─── Phase 5: Adversarial QA Generation ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 5 -- Generating Adversarial QA Pairs")
    logger.info("=" * 60)

    from src.qa_generator import generate_adversarial_qa
    qa_pairs = generate_adversarial_qa(
        run_retrieval_check=not args.no_retrieval_check
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Done! {len(qa_pairs)} QA pairs -> {args.output}")
    print(f"\n{'='*60}")
    print(f"[OK] {len(qa_pairs)} Adversarial QA pairs saved to: {args.output}")
    print(f"{'='*60}")

    # Pretty-print a summary
    for i, pair in enumerate(qa_pairs, 1):
        print(f"\n-- QA {i} [{pair['failure_mode']}] ------------------")
        print(f"  Q: {pair['question'][:120]}...")
        print(f"  Source: {pair['source']['video']} | {pair['source']['timestamp']} | {pair['source']['language']}")
        if pair.get("retrieved_top1"):
            snippet = pair["retrieved_top1"][:100].replace("\n", " ")
            print(f"  Retrieved Top-1: {snippet}...")


if __name__ == "__main__":
    main()
