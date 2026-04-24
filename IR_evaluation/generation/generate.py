#!/usr/bin/env python3
"""
Generate Module
===============
Output: generations.jsonl (one JSON object per question)

Usage:
    python run.py generate --config configs/rag_k1.yaml
"""

import os
import sys
import csv
import time
import json
import argparse
import requests
import urllib3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Suppress SSL warnings for trusted endpoints
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import GenerationConfig, GenerationRecord, RetrievedChunk
from utils import generate_run_id, ensure_dir, get_git_commit_hash, save_jsonl
from generation.reranker import get_reranker, rerank_chunks

def load_questions(csv_path: str) -> List[Dict[str, str]]:
    """
    Load evaluation questions from CSV.
    List of question dictionaries with question_id, question, gold_answer
    """
    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Generate question_id if not present
            question_id = row.get("question_id", f"q_{idx:04d}")
            
            questions.append({
                "question_id": question_id,
                "question": row["question"],
                "gold_answer": row.get("groundtruth", ""),
                "category": row.get("category", ""),
            })
    
    return questions


def call_rag_endpoint(
    question: str,
    endpoint: str,
    top_k: int = 5,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Call the RAG endpoint and return the response.
    Returns: Dictionary with response data
    """
    payload = {
        "query": question,
        "prompt": [
            {
                "role": "system",
          
            #    "content": "You are a retrieval-augmented question answering system. You must answer using only the provided context. Do not use any prior or external knowledge. Do not make assumptions or fill in missing information. If the context does not contain sufficient information to answer the question, respond exactyl with : I don't know (for english questions) or je ne sais pas (for French questions). Your answers must be concise and factual. Answer in max 1-2 sentences",
                        "content":
               "You are a retrieval-augmented question answering system. "
    "You must answer using only the provided context and no prior or external knowledge. "
    "You may combine information from multiple documents in the context to derive the answer. "
    "Do not introduce new facts, assumptions, or interpretations that are not supported by the context. "
    "If the answer cannot be clearly and directly derived from the provided context, "
    "respond exactly with: I don't know (for English questions) or je ne sais pas (for French questions). "
    "Your answers must be concise, factual, and limited to 1–2 sentences." 
            }, 
            {
                "role": "user",
    "content": "\nQuestion: user_question \nDocuments: retrieved_docs"
            }   ],
        "k": top_k,
    "chroma_index": "apple_ocr_gemini_05m" # can be any of ["apple_ocr_gemini_05m", "apple_ocr_gemini_05m_corrected"]. default: apple_ocr_gemini_05m

    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout,
            verify=False,  # For self-signed certs
        )
        response.raise_for_status()
        result = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        # Try multiple possible keys for chunks
        chunks = (
            result.get("chunks") or 
            result.get("contexts") or 
            result.get("retrieved_context") or
            result.get("sources") or
            result.get("documents") or
            result.get("retrieved_documents") or
            result.get("context") or  # Sometimes it's singular
            []
        )
        
        # If chunks is still empty, check if there's a nested structure
        if not chunks and isinstance(result, dict):
            # Log available keys for debugging
            print(f"      [DEBUG] Endpoint response keys: {list(result.keys())}")
        
        return {
            "success": True,
            "answer": result.get("answer", result.get("response", "")),
            "chunks": chunks,
            "latency_ms": latency_ms,
            "raw_response": result,
        }
    
    except requests.exceptions.RequestException as e:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "success": False,
            "answer": "",
            "chunks": [],
            "latency_ms": latency_ms,
            "error": str(e),
        }


def parse_chunks(chunks: Any, top_k: int) -> List[Dict[str, Any]]:
    """
    Parse and normalize retrieved chunks to consistent format.
    Returns: List of normalized chunk dictionaries
    """
    if not chunks:
        return []
    
    # Handle string format (JSON string containing array)
    if isinstance(chunks, str):
        try:
            chunks = json.loads(chunks)
        except json.JSONDecodeError:
            return [{"text": chunks, "rank": 0, "score": None, "doc_id": None}]
    
    # Handle list format
    if isinstance(chunks, list):
        # Case: list with single element that is a JSON array string
        if len(chunks) == 1 and isinstance(chunks[0], str):
            try:
                inner = json.loads(chunks[0])
                if isinstance(inner, list):
                    chunks = inner
            except json.JSONDecodeError:
                pass
        
        # Case: list with single element that is a list
        if len(chunks) == 1 and isinstance(chunks[0], list):
            chunks = chunks[0]
        
        parsed = []
        for i, chunk in enumerate(chunks[:top_k]):
            if isinstance(chunk, dict):
                parsed.append({
                    "text": chunk.get("text", chunk.get("content", str(chunk))),
                    "rank": chunk.get("rank", i),
                    "score": chunk.get("score", chunk.get("similarity", None)),
                    "doc_id": chunk.get("doc_id", chunk.get("id", None)),
                })
            elif isinstance(chunk, str):
                parsed.append({
                    "text": chunk,
                    "rank": i,
                    "score": None,
                    "doc_id": None,
                })
            else:
                parsed.append({
                    "text": str(chunk),
                    "rank": i,
                    "score": None,
                    "doc_id": None,
                })
        return parsed
    
    return []


def run_generation(config: GenerationConfig, verbose: bool = True) -> str:
    """
    Run generation pipeline.
    Returns: Path to the run directory
    """
    # Track generation time
    generation_start = time.perf_counter()
    
    # Setup - use run_name from config if provided, otherwise generate run_id
    if config.run_name:
        run_id = config.run_name
    elif config.run_id:
        run_id = config.run_id
    else:
        run_id = generate_run_id(f"run_{config.reranker_name}_k{config.top_k}")
    
    base_dir = Path(__file__).parent.parent
    run_dir = ensure_dir(base_dir / config.output_dir / run_id)
    
    # Initialize reranker if specified (but we use the google colab notebook to make it faster (CF readme))
    reranker = None
    if config.reranker_name not in ("none", "default"):
        if verbose:
            print(f"📦 Initializing reranker: {config.reranker_name}")
        reranker = get_reranker(config.reranker_name)
    
    if verbose:
        print("=" * 70)
        print(f"🚀 GENERATION PIPELINE")
        print(f"   Run ID: {run_id}")
        print(f"   Endpoint: {config.rag_endpoint}")
        print(f"   Top-K: {config.top_k}")
        print(f"   Reranker: {config.reranker_name}")
        print("=" * 70)
    
    # Load questions
    questions_path = base_dir / config.questions_csv_path
    questions = load_questions(str(questions_path))
    
    if verbose:
        print(f"\n📂 Loaded {len(questions)} questions from {config.questions_csv_path}")
    
    # Generate answers
    records = []
    total_latency = 0
    success_count = 0
    
    if config.fetch_k is not None:
        fetch_k = config.fetch_k
    elif reranker:
        fetch_k = config.top_k * 3 
    else:
        fetch_k = config.top_k
    
    for i, q in enumerate(questions):
        if verbose:
            print(f"\n[{i+1}/{len(questions)}] Processing: {q['question'][:60]}...")
        
        result = call_rag_endpoint(
            question=q["question"],
            endpoint=config.rag_endpoint,
            top_k=fetch_k,
        )
        
        if result["success"]:
            success_count += 1
            if verbose:
                print(f"   ✓ Got answer ({result['latency_ms']:.0f}ms)")
        else:
            if verbose:
                print(f"   ✗ Error: {result.get('error', 'Unknown')}")
        
        total_latency += result["latency_ms"]
        
        # Parse chunks
        chunks = parse_chunks(result["chunks"], fetch_k)
        
        # Apply reranking if configured
        if reranker and chunks:
            if verbose:
                print(f"   🔄 Reranking {len(chunks)} chunks...")
            chunks = rerank_chunks(reranker, q["question"], chunks, top_k=config.top_k)
            if verbose:
                print(f"   ✓ Kept top {len(chunks)} after reranking")
        else:
            # Just take top_k
            chunks = chunks[:config.top_k]
        
        # Create record
        record = GenerationRecord(
            question_id=q["question_id"],
            question=q["question"],
            generated_answer=result["answer"],
            retrieved_chunks=chunks,
            method_metadata={
                "reranker_name": config.reranker_name,
                "top_k": config.top_k,
                "fetch_k": fetch_k if reranker else config.top_k,
                "model_name": config.model_name,
                "endpoint": config.rag_endpoint,
            },
            latency_ms=result["latency_ms"],
        )
        
        records.append(record.to_dict())
    
    # Save generations.jsonl
    generations_path = run_dir / "generations.jsonl"
    save_jsonl(records, str(generations_path))
    
    # Calculate generation time
    generation_time_minutes = (time.perf_counter() - generation_start) / 60
    
    if verbose:
        print(f"\n" + "=" * 70)
        print(f"✅ GENERATION COMPLETE")
        print(f"   Total questions: {len(questions)}")
        print(f"   Successful: {success_count}")
        print(f"   Total latency: {total_latency/1000:.1f}s")
        print(f"   Avg latency: {total_latency/len(questions):.0f}ms")
        print(f"   Output: {generations_path}")
        print("=" * 70)
        print(f"\n⏱️  Generation Time: {generation_time_minutes:.2f} minutes")
    
    # Save gold answers for metrics computation
    gold_path = run_dir / "gold_answers.json"
    gold_data = {q["question_id"]: q["gold_answer"] for q in questions}
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, indent=2, ensure_ascii=False)
    
    return str(run_dir)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate RAG answers and capture retrieval data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Load config from YAML
    config = GenerationConfig.from_yaml(args.config)
    
    # Run generation
    run_dir = run_generation(config, verbose=not args.quiet)
    
    print(f"\n📁 Run saved to: {run_dir}")
    print(f"   Next step: python run.py metrics --run {run_dir}")


if __name__ == "__main__":
    main()
