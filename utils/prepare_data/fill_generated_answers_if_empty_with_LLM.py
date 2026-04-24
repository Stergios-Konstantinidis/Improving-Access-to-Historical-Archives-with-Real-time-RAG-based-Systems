#!/usr/bin/env python3
"""
Fill generated_answer in a generations.jsonl file using ONLY the stored retrieved_chunks.

Usage:
    python fill_generated_answers.py
    python fill_generated_answers.py --k 5
    python fill_generated_answers.py --force
    python fill_generated_answers.py --workers 5
    python fill_generated_answers.py --input path/to/generations_k5.jsonl
"""

import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

# =============================
# Paths
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".." / ".env"

load_dotenv(dotenv_path=ENV_PATH)

INPUT_FILE = PROJECT_ROOT / "data" / "reranker" / "processed" / "subset" / "generations_k5.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "reranker" / "processed" / "subset" / "enriched"

DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# SYSTEM_PROMPT = """
# You are a retrieval-augmented question answering system.
#
# You must answer using only the provided context and no prior or external knowledge.
# You may combine information from multiple documents in the context to derive the answer.
#
# Your goal is to produce a short, explicit, highly relevant answer that directly addresses the question.
#
# Rules:
# - Answer the question directly and explicitly.
# - Prefer one short complete sentence.
# - Make the answer understandable on its own.
# - Include the main subject of the question in the answer when helpful.
# - Use wording that stays close to the wording of the question.
# - Keep only the information needed to answer the question.
# - Avoid raw fragments, bullet points, abbreviations, and telegraphic style.
# - If the context provides an answer, state it clearly as a direct response.
# - Only respond with:
# I don't know
# if the context provides no answer at all.
#
# Output only the answer.
# """.strip()


SYSTEM_PROMPT = """
You are a retrieval-augmented question answering system.

Answer using only the provided context.
Do not use prior knowledge.
Do not infer beyond what is explicitly stated.

Goal:
Produce the shortest answer that is fully supported by the context and directly matches the question.

Rules:
- Output only the minimal answer needed.
- Do not add background, explanation, or extra facts.
- If the answer is a name, title, date, number, place, or short phrase, output only that.
- If the question asks for a full sentence, output one short sentence only.
- Stay as close as possible to the wording found in the context.
- Normalize obvious OCR noise only when the intended answer is unambiguous.
- If the context does not contain the answer, reply exactly:
je ne sais pas
for French questions, or:
I don't know
for English questions.

Output only the answer.
""".strip()

_thread_local = threading.local()
def ensure_parent_dir(filepath: Path) -> None:
    """Create parent directory if it does not exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)


def make_llm_client() -> OpenAI:
    """Create OpenAI-compatible client from environment."""
    kwargs: Dict[str, Any] = {}

    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY

    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    return OpenAI(**kwargs)


def get_thread_client() -> OpenAI:
    """Create one client per thread."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = make_llm_client()
    return _thread_local.client


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    records: List[Dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {filepath}: {e}"
                ) from e

    return records


def save_jsonl(records: List[Dict[str, Any]], filepath: Path) -> None:
    """Save JSONL file."""
    ensure_parent_dir(filepath)

    with filepath.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save JSON file."""
    ensure_parent_dir(filepath)

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_summary_file_path(output_file: Path) -> Path:
    """
    Example:
      generations_k5_answered.jsonl
      -> generations_k5_answered_summary.json
    """
    return output_file.with_name(f"{output_file.stem}_summary.json")


def is_empty_generated_answer(record: Dict[str, Any]) -> bool:
    """Return True if generated_answer is missing, empty, or NaN."""
    answer = record.get("generated_answer")

    if answer is None:
        return True

    try:
        if isinstance(answer, (float, int)) and math.isnan(answer):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(answer, str):
        normalized = answer.strip().lower()
        if normalized in {"", "nan", "none", "null"}:
            return True

    return False


def looks_french(text: str) -> bool:
    """Very simple heuristic to choose fallback language."""
    if not text:
        return False

    lowered = f" {text.lower()} "
    french_markers = [
        " le ", " la ", " les ", " des ", " du ", " de ",
        " quel ", " quelle ", " quels ", " quelles ",
        " quand ", " où ", " pourquoi ", " comment ",
        " selon ", " dans ", " sur ", " à ", " au ", " aux ",
    ]
    score = sum(marker in lowered for marker in french_markers)
    has_french_chars = any(ch in lowered for ch in "éèêàâùûôîïçœ")

    return score >= 2 or has_french_chars


def default_unknown_answer(question: str) -> str:
    """Return the fallback unknown answer in the detected question language."""
    return "je ne sais pas" if looks_french(question) else "I don't know"


def format_chunks_for_prompt(record: Dict[str, Any], k: Optional[int] = None) -> List[str]:
    chunks = record.get("retrieved_chunks", [])
    if not isinstance(chunks, list):
        return []

    if k is not None and k >= 0:
        chunks = chunks[:k]

    formatted_chunks: List[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            text = str(chunk).strip()
            if text:
                formatted_chunks.append(f"[Chunk {idx}]\n{text}")
            continue

        text = chunk.get("text", chunk.get("content", ""))
        if not isinstance(text, str) or not text.strip():
            continue

        metadata = chunk.get("metadata", {}) or {}

        header_parts = [f"Chunk {idx}"]

        for key in ["rank", "doc_id", "id", "relevance_score"]:
            value = chunk.get(key)
            if value not in (None, ""):
                header_parts.append(f"{key}={value}")

        for key in ["date", "year", "topic", "organization", "retrieval_source"]:
            value = metadata.get(key)
            if value not in (None, ""):
                header_parts.append(f"{key}={value}")

        header = "[" + " | ".join(header_parts) + "]"
        formatted_chunks.append(f"{header}\n{text.strip()}")

    return formatted_chunks


def generate_answer_from_chunks(
    client: OpenAI,
    question: str,
    formatted_chunks: List[str],
    model: str,
) -> str:
    """Generate an answer using ONLY the provided chunks."""
    if not question.strip():
        return ""

    if not formatted_chunks:
        return default_unknown_answer(question)

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Document chunks:\n\n" + "\n\n".join(formatted_chunks)
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content or ""
    answer = " ".join(answer.split()).strip()

    if not answer:
        return default_unknown_answer(question)

    return answer


def process_one_record(
    idx: int,
    record: Dict[str, Any],
    model: str,
    k: Optional[int],
    force: bool,
) -> Tuple[int, Dict[str, Any], str]:
    """
    Process one record.
    Returns: (index, updated_record, status)
    status in {"updated", "skipped", "error"}
    """
    new_record = dict(record)

    should_generate = force or is_empty_generated_answer(new_record)
    if not should_generate:
        new_record["_generation_time_ms"] = 0.0
        return idx, new_record, "skipped"

    question = str(new_record.get("question", "")).strip()
    formatted_chunks = format_chunks_for_prompt(new_record, k=k)

    start = time.perf_counter()
    try:
        client = get_thread_client()
        answer = generate_answer_from_chunks(
            client=client,
            question=question,
            formatted_chunks=formatted_chunks,
            model=model,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        new_record["generated_answer"] = answer
        new_record["_generation_time_ms"] = round(elapsed_ms, 2)
        return idx, new_record, "updated"
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        new_record["_generation_error"] = str(e)
        new_record["_generation_time_ms"] = round(elapsed_ms, 2)
        return idx, new_record, "error"


def process_records(
    records: List[Dict[str, Any]],
    model: str,
    k: Optional[int],
    force: bool,
    workers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fill generated_answer for each record in parallel and return summary."""
    total = len(records)
    results: List[Optional[Dict[str, Any]]] = [None] * total

    updated_count = 0
    skipped_count = 0
    error_count = 0

    global_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_one_record, idx, record, model, k, force)
            for idx, record in enumerate(records)
        ]

        with tqdm(total=total, desc="Generating answers", unit="record") as pbar:
            for future in as_completed(futures):
                idx, updated_record, status = future.result()
                results[idx] = updated_record

                if status == "updated":
                    updated_count += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1

                pbar.set_postfix(
                    updated=updated_count,
                    skipped=skipped_count,
                    errors=error_count,
                )
                pbar.update(1)

    total_elapsed_s = time.perf_counter() - global_start

    final_results: List[Dict[str, Any]] = []
    generation_times_ms: List[float] = []

    for i, record in enumerate(results):
        if record is None:
            raise RuntimeError(f"Missing result for record index {i}")
        final_results.append(record)

        t = record.get("_generation_time_ms")
        if isinstance(t, (int, float)) and t > 0:
            generation_times_ms.append(float(t))

    avg_ms = sum(generation_times_ms) / len(generation_times_ms) if generation_times_ms else 0.0
    total_generation_time_ms = sum(generation_times_ms)

    summary = {
        "records": {
            "total": total,
            "updated": updated_count,
            "skipped": skipped_count,
            "errors": error_count,
        },
        "timing": {
            "total_wall_time_s": round(total_elapsed_s, 2),
            "total_generation_time_ms": round(total_generation_time_ms, 2),
            "avg_generation_time_ms": round(avg_ms, 2),
        },
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total records            : {total}")
    print(f"Updated                  : {updated_count}")
    print(f"Skipped                  : {skipped_count}")
    print(f"Errors                   : {error_count}")
    print(f"Total wall time (s)      : {total_elapsed_s:.2f}")
    print(f"Total generation time (ms): {total_generation_time_ms:.2f}")
    print(f"Avg generation time (ms) : {avg_ms:.2f}")

    return final_results, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill generated_answer using ONLY stored retrieved_chunks."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_FILE),
        help="Path to input generations.jsonl file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file. If omitted, writes <input_stem>_answered.jsonl in OUTPUT_DIR",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional number of first retrieved_chunks to use for generation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate generated_answer even if it is already non-empty",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of worker threads (default: 5)",
    )

    args = parser.parse_args()

    input_file = Path(args.input).resolve()

    if args.output is not None:
        output_file = Path(args.output).resolve()
    else:
        output_file = OUTPUT_DIR / f"{input_file.stem}_answered.jsonl"

    summary_file = build_summary_file_path(output_file)

    if args.k is not None and args.k < 0:
        raise ValueError("--k must be >= 0")

    if args.workers <= 0:
        raise ValueError("--workers must be >= 1")

    print("=" * 70)
    print("FILL GENERATED ANSWERS")
    print(f"Input   : {input_file}")
    print(f"Output  : {output_file}")
    print(f"Summary : {summary_file}")
    print(f"Model   : {args.model}")
    print(f"k       : {args.k if args.k is not None else 'all chunks'}")
    print(f"Force   : {args.force}")
    print(f"Workers : {args.workers}")
    print("=" * 70)

    main_start = time.perf_counter()

    records = load_jsonl(input_file)
    updated_records, process_summary = process_records(
        records=records,
        model=args.model,
        k=args.k,
        force=args.force,
        workers=args.workers,
    )
    save_jsonl(updated_records, output_file)

    main_elapsed_s = time.perf_counter() - main_start

    run_summary = {
        "files": {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "summary_file": str(summary_file),
        },
        "config": {
            "model": args.model,
            "k": args.k,
            "force": args.force,
            "workers": args.workers,
            "system_prompt": SYSTEM_PROMPT,
        },
        "results": process_summary["records"],
        "timing": {
            **process_summary["timing"],
            "end_to_end_total_time_s": round(main_elapsed_s, 2),
        },
    }

    save_json(run_summary, summary_file)

    print(f"\nSaved file: {output_file}")
    print(f"Saved summary: {summary_file}")
    print(f"End-to-end total time (s): {main_elapsed_s:.2f}")


if __name__ == "__main__":
    main()