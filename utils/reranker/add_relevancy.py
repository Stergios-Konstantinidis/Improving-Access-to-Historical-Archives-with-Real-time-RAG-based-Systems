#!/usr/bin/env python3
"""
Metrics Module
==============
Orchestrates evaluation metrics computation.

Metrics computed:
- eRAG: Chunk Relevance Precision@k, Chunk Relevance NDCG@k

Usage:
    python -m metrics --run runs/baseline_k5
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FOLDER = PROJECT_ROOT / "data" / "baseline"
OUTPUT_FILE = PROJECT_ROOT / "data" / "reranker" / "processed"

from erag_metrics import compute_erag_metrics


def load_generations(filepath: str) -> List[Dict]:
    """Load generations from a JSONL file."""
    generations = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading generations", unit="line"):
            if line.strip():
                generations.append(json.loads(line))
    return generations


def load_gold_answers(filepath: str) -> Dict[str, str]:
    """Load gold answers from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_metrics(run_dir: str) -> Dict[str, Any]:
    """
    Run metrics on a generation run.

    Args:
        run_dir: Path to run directory containing generations.jsonl and gold_answers.json,
                 OR path to a single .jsonl file (gold answers extracted from file)
        generation_time_minutes: Time taken for generation (from previous step)

    Returns:
        Dict with metrics and timing info
    """
    run_path = Path(run_dir)

    # Check if it's a single .jsonl file or a directory
    if run_path.is_file() and run_path.suffix == ".jsonl":
        # Single file mode - extract gold answers from the file itself
        generations_path = run_path
        gold_answers_path = None
        run_name = run_path.stem
    else:
        # Directory mode - standard run directory structure
        generations_path = run_path / "generations.jsonl"
        gold_answers_path = run_path / "gold_answers.json"
        run_name = run_path.name

    # Load data
    print("=" * 70)
    print("METRICS COMPUTATION")
    print(f"   Run: {run_name}")
    print("=" * 70)

    generations = load_generations(str(generations_path))

    if gold_answers_path and gold_answers_path.exists():
        gold_answers = load_gold_answers(str(gold_answers_path))
    else:
        gold_answers = {
            sample["question_id"]: sample.get("reference", sample.get("gold_answer", ""))
            for sample in tqdm(generations, desc="Extracting gold answers", unit="sample")
        }

    print("\n" + "=" * 70)
    print("DEBUG - GOLD ANSWERS CHECK")
    print("=" * 70)

    missing = []
    empty = []

    for qid, ans in tqdm(gold_answers.items(), desc="Checking gold answers", unit="sample"):
        if ans is None:
            missing.append(qid)
        elif not str(ans).strip():
            empty.append(qid)

    print(f"Total samples: {len(gold_answers)}")
    print(f"Missing gold answers (None): {len(missing)}")
    print(f"Empty gold answers (''): {len(empty)}")

    if missing:
        print("Examples missing:", missing[:5])

    if empty:
        print("Examples empty:", empty[:5])

    if missing or empty:
        raise ValueError(
            f"Gold answers invalid: {len(missing)} missing, {len(empty)} empty"
        )

    print(f"\nLoaded {len(generations)} samples")

    # DEBUG: inspect first sample
    if generations:
        print("\n" + "=" * 70)
        print("DEBUG - FIRST GENERATION KEYS")
        print("=" * 70)
        print(list(generations[0].keys()))

        print("\n" + "=" * 70)
        print("DEBUG - FIRST GENERATION SAMPLE (truncated)")
        print("=" * 70)
        print(json.dumps(generations[0], indent=2, ensure_ascii=False)[:4000])

        print("\n" + "=" * 70)
        print("DEBUG - CANDIDATE CHUNK FIELDS")
        print("=" * 70)
        for candidate_key in [
            "contexts",
            "retrieved_contexts",
            "retrieved_chunks",
            "chunks",
            "documents",
            "docs",
            "contexts_text",
            "passages",
        ]:
            if candidate_key in generations[0]:
                value = generations[0][candidate_key]
                if isinstance(value, list):
                    print(f"{candidate_key}: list(len={len(value)})")
                    if value:
                        first_item = value[0]
                        print(f"  first item type: {type(first_item).__name__}")
                        if isinstance(first_item, dict):
                            print(f"  first item keys: {list(first_item.keys())}")
                        else:
                            print(f"  first item preview: {str(first_item)[:300]}")
                else:
                    print(f"{candidate_key}: type={type(value).__name__}, preview={str(value)[:300]}")

    # Compute eRAG metrics
    print(f"\nComputing eRAG metrics (LLM-Judged Chunk Contribution)...")
    erag_start = time.perf_counter()
    erag_results = compute_erag_metrics(generations, gold_answers)
    erag_time = time.perf_counter() - erag_start

    # DEBUG: inspect first eRAG result
    if erag_results:
        print("\n" + "=" * 70)
        print("DEBUG - FIRST eRAG RESULT")
        print("=" * 70)
        print(json.dumps(erag_results[0], indent=2, ensure_ascii=False)[:10000])

    # Merge results
    merged_results = []
    for gen, erag in tqdm(
        zip(generations, erag_results),
        total=min(len(generations), len(erag_results)),
        desc="Merging results",
        unit="sample",
    ):
        question_id = gen["question_id"]
        merged_results.append({
            "question_id": question_id,
            "question": gen.get("question", ""),
            "gold_answer": gold_answers.get(question_id, ""),
            "generated_answer": gen.get("generated_answer", ""),
            "chunk_relevance_precision_at_k": erag["chunk_relevance_precision_at_k"],
            "chunk_relevance_ndcg_at_k": erag["chunk_relevance_ndcg_at_k"],
            "chunk_details": erag.get("chunk_details", []),
        })

    # Compute averages
    n = len(merged_results)

    def safe_avg(key):
        values = [r[key] for r in merged_results if r.get(key) is not None and r.get(key) != -1]
        return sum(values) / len(values) if values else 0.0

    avg_chunk_relevance_precision = safe_avg("chunk_relevance_precision_at_k")
    avg_chunk_relevance_ndcg = safe_avg("chunk_relevance_ndcg_at_k")

    # Count "I don't have enough information" answers
    insufficient_count = 0
    for result in tqdm(merged_results, desc="Counting insufficient answers", unit="sample"):
        raw_answer = result.get("generated_answer", "")
        answer = str(raw_answer).lower().strip().strip('"\'')
        if (
            "i don't have enough information" in answer
            or "je n'ai pas assez d'informations" in answer
            or "je n'ai pas assez d'information" in answer
        ):
            insufficient_count += 1

    OUTPUT_FILE.mkdir(parents=True, exist_ok=True)

    # Save results
    metrics_path = OUTPUT_FILE / f"{run_name}_metrics.json" if run_path.is_file() else OUTPUT_FILE / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "chunk_relevance_precision_at_k": round(avg_chunk_relevance_precision, 4),
        "chunk_relevance_ndcg_at_k": round(avg_chunk_relevance_ndcg, 4),
        "num_samples": n,
        "insufficient_answers_count": insufficient_count,
        "erag_eval_time_seconds": round(erag_time, 2),
    }

    summary_path = OUTPUT_FILE / f"{run_name}_summary.json" if run_path.is_file() else OUTPUT_FILE / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print_results(summary)

    return summary


def print_results(summary: Dict[str, Any]):
    """Print results in formatted output."""
    chunk_relevance_precision_pct = summary["chunk_relevance_precision_at_k"] * 100
    chunk_relevance_ndcg_pct = summary["chunk_relevance_ndcg_at_k"] * 100

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print("\nResults:")
    num_samples = summary.get("num_samples", 0)
    print(f"   • Chunk Relevance Precision@k (LJCC):  {chunk_relevance_precision_pct:.2f}%")
    print(f"   • Chunk Relevance NDCG@k (LJCC):       {chunk_relevance_ndcg_pct:.2f}%")

    insufficient_count = summary.get("insufficient_answers_count", 0)
    if num_samples > 0:
        insufficient_pct = (insufficient_count / num_samples) * 100
        print(f"\nInsufficient Answers: {insufficient_count}/{num_samples} ({insufficient_pct:.1f}%)")

    if "erag_eval_time_seconds" in summary:
        print(f"eRAG eval time: {summary['erag_eval_time_seconds']:.2f}s")

    print("\n" + "-" * 70)
    print("Copy-paste line for Google Sheets:")
    print("-" * 70)

    sheet_line = f"{chunk_relevance_precision_pct:.2f}\t{chunk_relevance_ndcg_pct:.2f}"
    print(sheet_line)
    print("-" * 70)

    print("\nColumn headers (for reference):")
    print("Chunk Relevance Precision@k (%)\tChunk Relevance NDCG@k (%)")
    print("=" * 70)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for RAG outputs.",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Path to run directory containing generations.jsonl, or path to a single .jsonl file",
        default=INPUT_FOLDER
    )

    args = parser.parse_args()
    run_metrics(args.run)


if __name__ == "__main__":
    main()