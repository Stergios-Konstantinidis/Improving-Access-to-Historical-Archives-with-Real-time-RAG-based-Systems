#!/usr/bin/env python3
"""
Metrics Module
==============
Orchestrates evaluation metrics computation.

Metrics computed:
- RAGAS: Faithfulness, Answer Relevancy, Context Relevance
- eRAG: Precision@k, NDCG@k

Usage:
    python -m metrics --run runs/baseline_k5
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .ragas_metrics import compute_ragas_metrics
from .erag_metrics import compute_erag_metrics, compute_erag_embedding_metrics


def load_generations(filepath: str) -> List[Dict]:
    """Load generations from a JSONL file."""
    generations = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line))
    return generations


def load_gold_answers(filepath: str) -> Dict[str, str]:
    """Load gold answers from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_metrics(run_dir: str, generation_time_minutes: float = 0.0) -> Dict[str, Any]:
    """
    Run all metrics on a generation run.
    
    Args:
        run_dir: Path to run directory containing generations.jsonl and gold_answers.json,
                 OR path to a single .jsonl file (gold answers extracted from file)
        generation_time_minutes: Time taken for generation (from previous step)
        
    Returns:
        Dict with all metrics and timing info
    """
    run_path = Path(run_dir)
    
    # Check if it's a single .jsonl file or a directory
    if run_path.is_file() and run_path.suffix == ".jsonl":
        # Single file mode - extract gold answers from the file itself
        generations_path = run_path
        gold_answers_path = None
        output_dir = run_path.parent
        run_name = run_path.stem
    else:
        # Directory mode - standard run directory structure
        generations_path = run_path / "generations.jsonl"
        gold_answers_path = run_path / "gold_answers.json"
        output_dir = run_path
        run_name = run_path.name
    
    # Load data
    print("=" * 70)
    print("📊 METRICS COMPUTATION")
    print(f"   Run: {run_name}")
    print("=" * 70)
    
    generations = load_generations(str(generations_path))
    
    # Load gold answers from file or extract from generations
    if gold_answers_path and gold_answers_path.exists():
        gold_answers = load_gold_answers(str(gold_answers_path))
    else:
        # Extract gold answers from generations file
        gold_answers = {
            sample["question_id"]: sample.get("reference", sample.get("gold_answer", ""))
            for sample in generations
        }
    
    print(f"\n📂 Loaded {len(generations)} samples")
    
    # Compute RAGAS metrics
    print(f"\n🔍 Computing RAGAS metrics...")
    ragas_start = time.perf_counter()
    ragas_results = compute_ragas_metrics(generations, gold_answers)
    ragas_time = time.perf_counter() - ragas_start
    
    # Compute eRAG metrics (with LJCC chunk relevance)
    print(f"\n🔍 Computing eRAG metrics (LLM-Judged Chunk Contribution)...")
    erag_start = time.perf_counter()
    erag_results = compute_erag_metrics(generations, gold_answers)
    erag_time = time.perf_counter() - erag_start
    
    # Compute eRAG embedding-based metrics (original paper approach)
    print(f"\n🔍 Computing eRAG metrics (Embedding-Based - Original Paper)...")
    #erag_embed_start = time.perf_counter()
    #erag_embed_results = compute_erag_embedding_metrics(generations, gold_answers)
    #erag_embed_time = time.perf_counter() - erag_embed_start
    
    total_eval_time = ragas_time + erag_time # + erag_embed_time
    
    # Merge results with questions and gold answers
    merged_results = []
    for gen, ragas, erag in zip(generations, ragas_results, erag_results): #, erag_embed_results):
        question_id = ragas["question_id"]
        merged_results.append({
            "question_id": question_id,
            "question": gen.get("question", ""),
            "gold_answer": gold_answers.get(question_id, ""),
            "generated_answer": gen.get("generated_answer", ""),
            "faithfulness": ragas["faithfulness"],
            "answer_relevancy": ragas["answer_relevancy"],
            "answer_correctness": ragas["answer_correctness"],
            "context_relevance": ragas["context_relevance"],
            "llm_correct_or_not": ragas.get("llm_correct_or_not", 0.0),
            "chunk_relevance_precision_at_k": erag["chunk_relevance_precision_at_k"],
            "chunk_relevance_ndcg_at_k": erag["chunk_relevance_ndcg_at_k"],
            #"erag_precision_at_k": erag_embed["erag_precision_at_k"],
            #"erag_ndcg_at_k": erag_embed["erag_ndcg_at_k"],
            "chunk_details": erag.get("chunk_details", []),
        })
    
    # Compute averages (handle None values)
    n = len(merged_results)
    
    def safe_avg(key):
        """Average values, excluding None and -1 (non-answers for faithfulness)."""
        values = [r[key] for r in merged_results if r.get(key) is not None and r.get(key) != -1]
        return sum(values) / len(values) if values else 0.0
    
    avg_faithfulness = safe_avg("faithfulness")
    avg_answer_relevancy = safe_avg("answer_relevancy")
    avg_answer_correctness = safe_avg("answer_correctness")
    avg_context_relevance = safe_avg("context_relevance")
    avg_llm_correct_or_not = safe_avg("llm_correct_or_not")
    avg_chunk_relevance_precision = safe_avg("chunk_relevance_precision_at_k")
    avg_chunk_relevance_ndcg = safe_avg("chunk_relevance_ndcg_at_k")
    #avg_erag_precision = safe_avg("erag_precision_at_k")
    #avg_erag_ndcg = safe_avg("erag_ndcg_at_k")
    
    # Count "I don't have enough information" answers
    insufficient_count = 0
    for result in merged_results:
        answer = result.get("generated_answer", "")
        if answer is None:
            continue
        answer = answer.lower().strip().strip('"\'')
        if ("i don't have enough information" in answer or
            "je n'ai pas assez d'informations" in answer or
            "je n'ai pas assez d'information" in answer):
            insufficient_count += 1
    
    # Save results
    metrics_path = output_dir / f"{run_name}_metrics.json" if run_path.is_file() else output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    # Count how many questions had actual answers for faithfulness (not -1)
    faithfulness_answer_count = len([r for r in merged_results if r.get("faithfulness") is not None and r.get("faithfulness") != -1])
    
    summary = {
        "faithfulness": round(avg_faithfulness, 4),
        "faithfulness_answer_count": faithfulness_answer_count,
        "answer_relevancy": round(avg_answer_relevancy, 4),
        "answer_correctness": round(avg_answer_correctness, 4),
        "context_relevance": round(avg_context_relevance, 4),
        "llm_correct_or_not": round(avg_llm_correct_or_not, 4),
        "chunk_relevance_precision_at_k": round(avg_chunk_relevance_precision, 4),
        "chunk_relevance_ndcg_at_k": round(avg_chunk_relevance_ndcg, 4),
        #"erag_precision_at_k": round(avg_erag_precision, 4),
        #"erag_ndcg_at_k": round(avg_erag_ndcg, 4),
        "generation_time_minutes": round(generation_time_minutes, 2),
        "evaluation_time_minutes": round(total_eval_time / 60, 2),
        "num_samples": n,
        "insufficient_answers_count": insufficient_count,
    }
    
    summary_path = output_dir / f"{run_name}_summary.json" if run_path.is_file() else output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print_results(summary)
    
    return summary


def print_results(summary: Dict[str, Any]):
    """Print results in formatted output."""
    
    # Convert to percentages
    faithfulness_pct = summary["faithfulness"] * 100
    answer_relevancy_pct = summary["answer_relevancy"] * 100
    answer_correctness_pct = summary["answer_correctness"] * 100
    context_relevance_pct = summary["context_relevance"] * 100
    llm_correct_or_not_pct = summary.get("llm_correct_or_not", 0.0) * 100
    chunk_relevance_precision_pct = summary["chunk_relevance_precision_at_k"] * 100
    chunk_relevance_ndcg_pct = summary["chunk_relevance_ndcg_at_k"] * 100
    #erag_precision_pct = summary["erag_precision_at_k"] * 100
    #       erag_ndcg_pct = summary["erag_ndcg_at_k"] * 100
    gen_time = summary["generation_time_minutes"]
    eval_time = summary["evaluation_time_minutes"]
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    
    print("\n📈 Results (bullet points):")
    faithfulness_answer_count = summary.get("faithfulness_answer_count", 0)
    num_samples = summary.get("num_samples", 0)
    print(f"   • Faithfulness (RAGAS):                {faithfulness_pct:.2f}% (based on {faithfulness_answer_count}/{num_samples} answers)")
    print(f"   • Answer Relevancy (RAGAS):            {answer_relevancy_pct:.2f}%")
    print(f"   • Answer Correctness (RAGAS):          {answer_correctness_pct:.2f}%")
    print(f"   • Context Relevance (RAGAS):           {context_relevance_pct:.2f}%")
    print(f"   • LLM Correct or Not:                  {llm_correct_or_not_pct:.2f}%")
    print(f"   • Chunk Relevance Precision@k (LJCC):  {chunk_relevance_precision_pct:.2f}%")
    print(f"   • Chunk Relevance NDCG@k (LJCC):       {chunk_relevance_ndcg_pct:.2f}%")
    #print(f"   • Precision@k (eRAG):                  {erag_precision_pct:.2f}%")
    #print(f"   • NDCG@k (eRAG):                       {erag_ndcg_pct:.2f}%")
    print(f"   • Generation Time:                     {gen_time:.2f} min")
    print(f"   • Evaluation Time:                     {eval_time:.2f} min")
    
    # Display insufficient answers count
    insufficient_count = summary.get("insufficient_answers_count", 0)
    num_samples = summary.get("num_samples", 0)
    if num_samples > 0:
        insufficient_pct = (insufficient_count / num_samples) * 100
        print(f"\n📊 Insufficient Answers: {insufficient_count}/{num_samples} ({insufficient_pct:.1f}%)")
    
    print("\n" + "-" * 70)
    print("📋 Copy-paste line for Google Sheets:")
    print("-" * 70)
    
    # Tab-separated values for easy paste
    # Order: Faithfulness | Answer Relevancy | Answer Correctness | Context Relevance | LLM Correct or Not |
    #        Chunk Relevance Precision@k | Chunk Relevance NDCG@k | eRAG Precision@k | eRAG NDCG@k | Gen Time | Eval Time
    sheet_line = f"{faithfulness_pct:.2f}\t{answer_relevancy_pct:.2f}\t{answer_correctness_pct:.2f}\t{context_relevance_pct:.2f}\t{llm_correct_or_not_pct:.2f}\t{chunk_relevance_precision_pct:.2f}\t{chunk_relevance_ndcg_pct:.2f}\t{0:.2f}\t{0:.2f}\t{gen_time:.2f}\t{eval_time:.2f}"
    print(sheet_line)
    print("-" * 70)
    
    print("\n📊 Column headers (for reference):")
    print("Faithfulness (%)\tAnswer Relevancy (%)\tAnswer Correctness (%)\tContext Relevance (%)\tLLM Correct or Not (%)\tChunk Relevance Precision@k (%)\tChunk Relevance NDCG@k (%)\tPrecision@k eRAG (%)\tNDCG@k eRAG (%)\tGen Time (min)\tEval Time (min)")
    print("=" * 70)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for RAG outputs.",
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to run directory containing generations.jsonl, or path to a single .jsonl file",
    )
    parser.add_argument(
        "--gen-time",
        type=float,
        default=0.0,
        help="Generation time in minutes (for reporting)",
    )
    
    args = parser.parse_args()
    run_metrics(args.run, args.gen_time)


if __name__ == "__main__":
    main()
