#!/usr/bin/env python3
"""
RAGAS Metrics Runner
====================
Runs RAGAS metrics from:
- generations.jsonl
- gold_answers.json

The generations.jsonl file is NOT modified.
gold_answers are read directly from gold_answers.json.

Usage:
    python run_ragas_metrics.py
    python run_ragas_metrics.py --run /path/to/run_dir
    python run_ragas_metrics.py --run /path/to/generations.jsonl --gold /path/to/gold_answers.json
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_CONCURRENT = int(os.getenv("RAGAS_MAX_CONCURRENT_SAMPLES", "5"))

DEFAULT_RUN = PROJECT_ROOT / "data" / "baseline" / "ragas"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "processed" / "ragas_erag"

from ragas_metrics import (
    load_generations,
    load_gold_answers,
    compute_ragas_metrics,
)


def extract_scores_from_retrieved_chunks(gen: Dict[str, Any]) -> List[float]:
    chunks = gen.get("retrieved_chunks", [])
    scores = []

    for i, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {}) or {}

        score = None
        for candidate in (
            chunk.get("relevance_score"),
            metadata.get("relevance_score"),
            chunk.get("is_relevant"),
            metadata.get("is_relevant"),
        ):
            if candidate is not None:
                score = candidate
                break

        if isinstance(score, bool):
            score = 1.0 if score else 0.0

        if score is None:
            score = 0.0

        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            print(
                f"[WARN] question_id={gen.get('question_id')} "
                f"chunk[{i}] invalid relevance score: {score!r}"
            )
            scores.append(0.0)

    return scores


def precision_at_k_from_scores(scores: List[float], k: int) -> float:
    if k <= 0:
        return 0.0

    topk = scores[:k]
    if not topk:
        return 0.0

    relevant_count = sum(1 for s in topk if s > 0)
    return relevant_count / len(topk)


def dcg_at_k(scores: List[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(scores[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k_from_scores(scores: List[float], k: int) -> float:
    if k <= 0:
        return 0.0

    actual = dcg_at_k(scores, k)
    ideal = dcg_at_k(sorted(scores, reverse=True), k)

    if ideal == 0:
        return 0.0

    return actual / ideal


def safe_avg(records: List[Dict[str, Any]], key: str) -> float:
    values = [
        r[key]
        for r in records
        if r.get(key) is not None and r.get(key) != -1
    ]
    return sum(values) / len(values) if values else 0.0


def safe_sum_nested(records: List[Dict[str, Any]], path: List[str]) -> float:
    total = 0.0
    for record in records:
        value: Any = record
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)

        if value is None:
            continue

        try:
            total += float(value)
        except (TypeError, ValueError):
            continue

    return total


def safe_avg_nested(records: List[Dict[str, Any]], path: List[str]) -> float:
    values = []

    for record in records:
        value: Any = record
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if value == -1:
            continue

        values.append(value)

    return sum(values) / len(values) if values else 0.0


def build_output_subdir_name(generations_file: Path) -> str:
    """
    Convertit un nom de fichier comme :
      - generations_k5.jsonl -> generations_5k
      - generations_5k.jsonl -> generations_5k
    Sinon, garde simplement le stem.
    """
    stem = generations_file.stem

    match_k_prefix = re.match(r"^generations_k(\d+)$", stem)
    if match_k_prefix:
        return f"generations_{match_k_prefix.group(1)}k"

    match_k_suffix = re.match(r"^generations_(\d+)k$", stem)
    if match_k_suffix:
        return stem

    return stem


def run_ragas_metrics(
    run_dir: str | Path,
    gold_path: str | Path | None = None,
    max_concurrent_samples: int = DEFAULT_MAX_CONCURRENT,
) -> Dict[str, Any]:
    run_path = Path(run_dir)

    if run_path.is_file() and run_path.suffix == ".jsonl":
        generations_path = run_path
        run_name = run_path.stem
        if gold_path is not None:
            gold_answers_path = Path(gold_path)
        else:
            gold_answers_path = run_path.parent / "gold_answers.json"
    else:
        generations_path = run_path / "generations.jsonl"
        run_name = run_path.name
        if gold_path is not None:
            gold_answers_path = Path(gold_path)
        else:
            gold_answers_path = run_path / "gold_answers.json"

    if not generations_path.exists():
        raise FileNotFoundError(f"generations.jsonl not found: {generations_path}")

    if not gold_answers_path.exists():
        raise FileNotFoundError(f"gold_answers.json not found: {gold_answers_path}")

    print("=" * 70)
    print("RAGAS METRICS COMPUTATION")
    print(f"Run               : {run_name}")
    print(f"Generations file  : {generations_path}")
    print(f"Gold answers file : {gold_answers_path}")
    print(f"Max concurrency   : {max_concurrent_samples}")
    print("=" * 70)

    generations = load_generations(generations_path)
    gold_answers = load_gold_answers(gold_answers_path)

    print(f"\nLoaded generations : {len(generations)}")
    print(f"Loaded gold answers: {len(gold_answers)}")

    results = compute_ragas_metrics(
        generations=generations,
        gold_answers=gold_answers,
        max_concurrent_samples=max_concurrent_samples,
    )

    merged_results: List[Dict[str, Any]] = []
    for gen, ragas in zip(generations, results):
        qid = gen["question_id"]

        scores = extract_scores_from_retrieved_chunks(gen)
        effective_k = len(scores)

        precision_k = precision_at_k_from_scores(scores, effective_k)
        ndcg_k = ndcg_at_k_from_scores(scores, effective_k)

        merged_results.append({
            "question_id": qid,
            "question": gen.get("question", ""),
            "gold_answer": gold_answers.get(qid, ""),
            "generated_answer": gen.get("generated_answer", ""),

            # Champs RAGAS
            "answer_relevancy": ragas.get("answer_relevancy"),
            "context_relevance": ragas.get("context_relevance"),
            "answer_correctness": ragas.get("answer_correctness"),
            "ragas_meta": ragas.get("meta", {}),
            "ragas_error": ragas.get("error"),

            # Champs eRAG recalculés sur le subset courant
            "chunk_relevance_precision_at_k": precision_k,
            "chunk_relevance_ndcg_at_k": ndcg_k,
            "effective_k": effective_k,
            "computed_ndcg_actual_scores": scores,
            "computed_ndcg_ideal_scores": sorted(scores, reverse=True),

            # Champs timing recopiés depuis le generations.jsonl
            "latency_ms": gen.get("latency_ms"),
            "latency_breakdown_ms": gen.get("latency_breakdown_ms", {}),
        })

    total_pipeline_ms = safe_sum_nested(generations, ["latency_ms"])
    total_retrieval_ms = safe_sum_nested(generations, ["latency_breakdown_ms", "retrieval_ms"])
    total_rerank_ms = safe_sum_nested(generations, ["latency_breakdown_ms", "rerank_ms"])
    total_generation_ms = safe_sum_nested(generations, ["latency_breakdown_ms", "generation_ms"])

    avg_pipeline_ms = safe_avg_nested(generations, ["latency_ms"])
    avg_retrieval_ms = safe_avg_nested(generations, ["latency_breakdown_ms", "retrieval_ms"])
    avg_rerank_ms = safe_avg_nested(generations, ["latency_breakdown_ms", "rerank_ms"])
    avg_generation_ms = safe_avg_nested(generations, ["latency_breakdown_ms", "generation_ms"])

    summary = {
        "answer_relevancy": round(safe_avg(merged_results, "answer_relevancy"), 4),
        "context_relevance": round(safe_avg(merged_results, "context_relevance"), 4),
        "answer_correctness": round(safe_avg(merged_results, "answer_correctness"), 4),

        # Résumés des champs eRAG recopiés
        "chunk_relevance_precision_at_k": round(safe_avg(merged_results, "chunk_relevance_precision_at_k"), 4),
        "chunk_relevance_ndcg_at_k": round(safe_avg(merged_results, "chunk_relevance_ndcg_at_k"), 4),

        "num_samples": len(merged_results),

        # Temps du pipeline source (hors calcul RAGAS/eRAG)
        "pipeline_total_time_seconds": round(total_pipeline_ms / 1000.0, 2),
        "pipeline_avg_time_ms": round(avg_pipeline_ms, 2),

        "retrieval_total_time_seconds": round(total_retrieval_ms / 1000.0, 2),
        "retrieval_avg_time_ms": round(avg_retrieval_ms, 2),

        "rerank_total_time_seconds": round(total_rerank_ms / 1000.0, 2),
        "rerank_avg_time_ms": round(avg_rerank_ms, 2),

        "generation_total_time_seconds": round(total_generation_ms / 1000.0, 2),
        "generation_avg_time_ms": round(avg_generation_ms, 2),
    }

    generations_resolved = generations_path.resolve()
    output_subdir_name = build_output_subdir_name(generations_resolved)
    final_output_dir = OUTPUT_DIR / output_subdir_name
    final_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = final_output_dir / "ragas_metrics.json"
    summary_path = final_output_dir / "ragas_summary.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("RAGAS & eRAG EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Answer Relevancy             : {summary['answer_relevancy'] * 100:.2f}%")
    print(f"Context Relevance            : {summary['context_relevance'] * 100:.2f}%")
    print(f"Answer Correctness           : {summary['answer_correctness'] * 100:.2f}%")
    print(f"Chunk Relevance Precision@k  : {summary['chunk_relevance_precision_at_k'] * 100:.2f}%")
    print(f"Chunk Relevance NDCG@k       : {summary['chunk_relevance_ndcg_at_k'] * 100:.2f}%")

    print("\nTIMINGS (source pipeline only, hors RAGAS/eRAG)")
    print("-" * 70)
    print(f"Pipeline Total Time          : {summary['pipeline_total_time_seconds']:.2f}s")
    print(f"Pipeline Avg / Question      : {summary['pipeline_avg_time_ms']:.2f} ms")
    print(f"Retrieval Total Time         : {summary['retrieval_total_time_seconds']:.2f}s")
    print(f"Retrieval Avg / Question     : {summary['retrieval_avg_time_ms']:.2f} ms")
    print(f"Rerank Total Time            : {summary['rerank_total_time_seconds']:.2f}s")
    print(f"Rerank Avg / Question        : {summary['rerank_avg_time_ms']:.2f} ms")
    print(f"Generation Total Time        : {summary['generation_total_time_seconds']:.2f}s")
    print(f"Generation Avg / Question    : {summary['generation_avg_time_ms']:.2f} ms")

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS metrics using generations.jsonl + gold_answers.json"
    )
    parser.add_argument(
        "--run",
        type=str,
        default=str(DEFAULT_RUN),
        help="Path to run directory or directly to generations.jsonl",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Optional explicit path to gold_answers.json",
    )
    parser.add_argument(
        "--max-concurrent-samples",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum concurrent samples (default: {DEFAULT_MAX_CONCURRENT})",
    )

    args = parser.parse_args()

    run_ragas_metrics(
        run_dir=args.run,
        gold_path=args.gold,
        max_concurrent_samples=args.max_concurrent_samples,
    )


if __name__ == "__main__":
    main()