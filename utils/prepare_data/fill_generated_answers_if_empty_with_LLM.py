#!/usr/bin/env python3
"""
Fill generated_answer in a generations.jsonl file using ONLY the stored retrieved_chunks.

Usage:
    python fill_generated_answers.py
    python fill_generated_answers.py --k 5
    python fill_generated_answers.py --force
    python fill_generated_answers.py --workers 5
    python fill_generated_answers.py --input path/to/generations_k5.jsonl
    python fill_generated_answers.py --stream_on
"""

import argparse
import json
import math
import os
import random
import re
import threading
import time
import traceback
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

DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL_TO_DEL_ONLY_THIS_PART", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# INITIAL
SYSTEM_PROMPT = """
You are a retrieval-augmented question answering system.

You must answer using only the provided context and no prior or external knowledge.
You may combine information from multiple documents in the context to derive the answer.

Your goal is to produce a short, explicit, highly relevant answer that directly addresses the question.

Rules:
- Answer the question directly and explicitly.
- Prefer one short complete sentence.
- Make the answer understandable on its own.
- Include the main subject of the question in the answer when helpful.
- Use wording that stays close to the wording of the question.
- Keep only the information needed to answer the question.
- Avoid raw fragments, bullet points, abbreviations, and telegraphic style.
- If the context provides an answer, state it clearly as a direct response.
- Only respond with:
I don't know
if the context provides no answer at all.

Output only the answer.
""".strip()

# Current best for average parameters
# SYSTEM_PROMPT = """
# Tu es un système de question-réponse avec récupération de documents historiques en français.
#
# Tu dois répondre uniquement à partir des extraits documentaires fournis dans le contexte.
# Tu ne dois utiliser aucune connaissance externe.
# Tu peux combiner plusieurs extraits lorsque chacun apporte une partie de la réponse.
#
# Objectif :
# produire une réponse finale courte, naturelle et factuellement exacte, comme le ferait un système RAG pour un utilisateur final.
#
# Règles de réponse :
# - Réponds toujours en français.
# - Réponds directement à la question, sans introduction inutile.
# - Ne donne pas seulement une valeur brute si elle serait ambiguë pour l’utilisateur.
#   Par exemple, réponds « Le prix était de 375 francs » plutôt que seulement « 375 ».
# - La réponse doit être une phrase courte et cohérente, sauf si la question demande explicitement une liste.
# - Inclus tous les détails nécessaires pour que la réponse soit correcte : noms, dates, lieux, montants, unités, titres, organisations, fonctions et relations.
# - Si la question demande un nom, une date, un prix, un âge, un lieu, un objet ou un nombre, conserve exactement cette information et ajoute seulement les mots nécessaires pour former une réponse claire.
# - Si la question demande une liste d’éléments, donne tous les éléments pertinents trouvés dans le contexte, sans en omettre.
# - Privilégie les formulations et les termes présents dans le contexte lorsque ceux-ci contiennent la réponse.
# - Utilise les métadonnées factuelles disponibles, comme la date, l’année, la source, le titre, la page, le lieu ou l’organisation, uniquement si elles aident à répondre à la question.
# - Ne considère jamais les signaux de classement, comme relevance_score, retrieval_rank, rrf_score ou fusion_score, comme des faits documentaires.
# - Si plusieurs extraits donnent des informations complémentaires, synthétise-les en une seule réponse.
# - Si plusieurs extraits semblent contradictoires, mentionne l’incertitude au lieu de choisir arbitrairement.
# - Si le contexte contient une réponse probable malgré du bruit OCR, donne la réponse la plus probable, mais seulement si elle est appuyée par le contexte.
# - Ne complète pas avec des suppositions.
# - Ne donne pas d’explication sur ta méthode.
# - Ne cite pas les chunks sauf si la question le demande.
# - Réponds exactement « je ne sais pas » uniquement si aucun extrait ne contient d’indice exploitable pour répondre.
#
# Format attendu :
# - Une seule phrase courte dans la majorité des cas.
# - Une liste courte uniquement lorsque la question demande plusieurs éléments.
# - Aucune préface comme « D’après le contexte » ou « Selon les documents », sauf si cela rend la réponse plus claire.
# - Ne donne que la réponse finale.
# """.strip()

# BEST FOR ACCURACY
# SYSTEM_PROMPT = """
# You are a retrieval-augmented question answering system.
#
# Answer using only the provided context.
# Do not use prior knowledge.
# Do not infer beyond what is explicitly stated.
#
# Goal:
# Produce the shortest answer that is fully supported by the context and directly matches the question.
#
# Rules:
# - Output only the minimal answer needed.
# - Do not add background, explanation, or extra facts.
# - If the answer is a name, title, date, number, place, or short phrase, output only that.
# - If the question asks for a full sentence, output one short sentence only.
# - Stay as close as possible to the wording found in the context.
# - Normalize obvious OCR noise only when the intended answer is unambiguous.
# - If the context does not contain the answer, reply exactly:
# je ne sais pas
# for French questions, or:
# I don't know
# for English questions.
#
# Output only the answer.
# """.strip()

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


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
}


def get_error_status_code(error: Exception) -> Optional[int]:
    """Extract an HTTP status code from OpenAI-compatible exceptions when available."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    return None


def is_retryable_error(error: Exception) -> bool:
    """Return True for transient API errors that are worth retrying."""
    status_code = get_error_status_code(error)
    if status_code in RETRYABLE_STATUS_CODES:
        return True

    error_name = type(error).__name__
    if error_name in RETRYABLE_ERROR_NAMES:
        return True

    message = str(error).lower()
    retryable_markers = [
        "rate limit",
        "rate_limit",
        "429",
        "timeout",
        "temporarily unavailable",
        "server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
    ]
    return any(marker in message for marker in retryable_markers)


def parse_retry_after_seconds(error: Exception) -> Optional[float]:
    """Parse retry delay from HTTP headers or OpenAI error messages."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)

    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                pass

    message = str(error)
    match = re.search(r"try again in\s+([0-9]*\.?[0-9]+)\s*(ms|s)", message, re.IGNORECASE)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "ms":
        return value / 1000.0
    return value


def compute_retry_sleep_seconds(
    error: Exception,
    attempt_number: int,
    retry_initial_wait_s: float,
    retry_max_wait_s: float,
) -> float:
    """Compute exponential backoff, respecting server-provided retry delays when present."""
    retry_after_s = parse_retry_after_seconds(error)
    exponential_wait_s = retry_initial_wait_s * (2 ** max(0, attempt_number - 1))

    if retry_after_s is not None:
        wait_s = max(retry_after_s, exponential_wait_s)
    else:
        wait_s = exponential_wait_s

    wait_s = min(retry_max_wait_s, wait_s)

    # Small jitter prevents several worker threads from retrying at exactly the same time.
    jitter_s = random.uniform(0.0, min(1.0, wait_s * 0.10))
    return wait_s + jitter_s


def generate_answer_from_chunks_with_retries(
    client: OpenAI,
    question: str,
    formatted_chunks: List[str],
    model: str,
    max_retries: int,
    retry_initial_wait_s: float,
    retry_max_wait_s: float,
    stream: bool = False,
) -> Tuple[str, Optional[float], int]:
    """Generate an answer and retry transient OpenAI-compatible API errors.

    Returns (answer, stream_first_chunk_ms, retry_count).
    stream_first_chunk_ms is None when streaming is disabled.
    """
    attempt_number = 0

    while True:
        try:
            answer, stream_first_chunk_ms = generate_answer_from_chunks(
                client=client,
                question=question,
                formatted_chunks=formatted_chunks,
                model=model,
                stream=stream,
            )
            return answer, stream_first_chunk_ms, attempt_number

        except Exception as e:
            if attempt_number >= max_retries or not is_retryable_error(e):
                raise

            attempt_number += 1
            sleep_s = compute_retry_sleep_seconds(
                error=e,
                attempt_number=attempt_number,
                retry_initial_wait_s=retry_initial_wait_s,
                retry_max_wait_s=retry_max_wait_s,
            )

            print(
                "\n"
                f"[WARN] Retryable API error on attempt {attempt_number}/{max_retries} "
                f"for model {model}: {type(e).__name__}. "
                f"Retrying in {sleep_s:.2f}s..."
            )
            time.sleep(sleep_s)


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

        best_span_text = metadata.get("best_span_text")

        if isinstance(best_span_text, str) and best_span_text.strip():
            formatted_chunks.append(
                f"{header}\n"
                f"[Full chunk]\n"
                f"{text.strip()}\n\n"
                f"[Focused evidence window]\n"
                f"{best_span_text.strip()}"
            )
        else:
            formatted_chunks.append(f"{header}\n{text.strip()}")
    return formatted_chunks


def generate_answer_from_chunks(
    client: OpenAI,
    question: str,
    formatted_chunks: List[str],
    model: str,
    stream: bool = False,
) -> Tuple[str, Optional[float]]:
    """Generate an answer using ONLY the provided chunks.

    Returns (answer, stream_first_chunk_ms).
    - When stream=False : behaves exactly as before, stream_first_chunk_ms is None.
    - When stream=True  : consumes the streamed response and measures the time
      (in ms) from the request start to the FIRST received content chunk.
    """
    if not question.strip():
        return "", None

    if not formatted_chunks:
        return default_unknown_answer(question), None

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Document chunks:\n\n" + "\n\n".join(formatted_chunks)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if not stream:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
        )

        answer = response.choices[0].message.content or ""
        answer = " ".join(answer.split()).strip()

        if not answer:
            return default_unknown_answer(question), None

        return answer, None

    # ---- Streaming mode : measure time to first content chunk ----
    request_start = time.perf_counter()
    stream_first_chunk_ms: Optional[float] = None
    pieces: List[str] = []

    response_stream = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        stream=True,
    )

    for chunk in response_stream:
        try:
            delta = getattr(chunk.choices[0].delta, "content", None)
        except Exception:
            continue
        if delta:
            if stream_first_chunk_ms is None:
                stream_first_chunk_ms = (time.perf_counter() - request_start) * 1000.0
            pieces.append(delta)

    answer = "".join(pieces)
    answer = " ".join(answer.split()).strip()

    if not answer:
        return default_unknown_answer(question), stream_first_chunk_ms

    return answer, stream_first_chunk_ms


def process_one_record(
    idx: int,
    record: Dict[str, Any],
    model: str,
    k: Optional[int],
    force: bool,
    max_retries: int,
    retry_initial_wait_s: float,
    retry_max_wait_s: float,
    stream: bool = False,
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
        answer, stream_first_chunk_ms, retry_count = generate_answer_from_chunks_with_retries(
            client=client,
            question=question,
            formatted_chunks=formatted_chunks,
            model=model,
            max_retries=max_retries,
            retry_initial_wait_s=retry_initial_wait_s,
            retry_max_wait_s=retry_max_wait_s,
            stream=stream,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        new_record["generated_answer"] = answer
        new_record["_generation_retries"] = retry_count
        new_record["_generation_time_ms"] = round(elapsed_ms, 2)

        # Streaming-only field : time to first streamed content chunk.
        if stream and stream_first_chunk_ms is not None:
            new_record["stream_first_chunk_ms"] = round(stream_first_chunk_ms, 2)

        return idx, new_record, "updated"

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        error_type = type(e).__name__
        error_message = str(e)

        print("\n" + "=" * 80)
        print("[ERROR] Failed to generate answer")
        print("=" * 80)
        print(f"Record index : {idx}")
        print(f"Question     : {question}")
        print(f"Model        : {model}")
        print(f"Error type   : {error_type}")
        print(f"Error message: {error_message}")
        print("-" * 80)
        print("Traceback:")
        print(traceback.format_exc())
        print("=" * 80)

        new_record["_generation_error"] = error_message
        new_record["_generation_error_type"] = error_type
        new_record["_generation_max_retries"] = max_retries
        new_record["_generation_time_ms"] = round(elapsed_ms, 2)
        return idx, new_record, "error"


def process_records(
    records: List[Dict[str, Any]],
    model: str,
    k: Optional[int],
    force: bool,
    workers: int,
    max_retries: int,
    retry_initial_wait_s: float,
    retry_max_wait_s: float,
    stream: bool = False,
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
            executor.submit(
                process_one_record,
                idx,
                record,
                model,
                k,
                force,
                max_retries,
                retry_initial_wait_s,
                retry_max_wait_s,
                stream,
            )
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
    stream_first_chunk_times_ms: List[float] = []

    for i, record in enumerate(results):
        if record is None:
            raise RuntimeError(f"Missing result for record index {i}")
        final_results.append(record)

        t = record.get("_generation_time_ms")
        if isinstance(t, (int, float)) and t > 0:
            generation_times_ms.append(float(t))

        s = record.get("stream_first_chunk_ms")
        if isinstance(s, (int, float)) and s > 0:
            stream_first_chunk_times_ms.append(float(s))

    avg_ms = sum(generation_times_ms) / len(generation_times_ms) if generation_times_ms else 0.0
    total_generation_time_ms = sum(generation_times_ms)

    avg_stream_first_chunk_ms = (
        sum(stream_first_chunk_times_ms) / len(stream_first_chunk_times_ms)
        if stream_first_chunk_times_ms else 0.0
    )

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
            "stream_enabled": bool(stream),
            "stream_first_chunk_count": len(stream_first_chunk_times_ms),
            "avg_stream_first_chunk_ms": round(avg_stream_first_chunk_ms, 2),
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
    if stream:
        print(f"Avg stream first chunk (ms): {avg_stream_first_chunk_ms:.2f} "
              f"(n={len(stream_first_chunk_times_ms)})")

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
        default=1,
        help="Number of worker threads (default: 1)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum retries per record for transient API errors such as 429 rate limits (default: 6)",
    )
    parser.add_argument(
        "--retry-initial-wait",
        type=float,
        default=1.0,
        help="Initial retry wait in seconds before exponential backoff (default: 1.0)",
    )
    parser.add_argument(
        "--retry-max-wait",
        type=float,
        default=30.0,
        help="Maximum retry wait in seconds between attempts (default: 30.0)",
    )
    parser.add_argument(
        "--stream_on",
        "--stream-on",
        dest="stream_on",
        action="store_true",
        help="Active la generation en streaming et enregistre stream_first_chunk_ms "
             "(temps jusqu'au premier chunk de contenu).",
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

    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")

    if args.retry_initial_wait < 0:
        raise ValueError("--retry-initial-wait must be >= 0")

    if args.retry_max_wait <= 0:
        raise ValueError("--retry-max-wait must be > 0")

    print("=" * 70)
    print("FILL GENERATED ANSWERS")
    print(f"Input   : {input_file}")
    print(f"Output  : {output_file}")
    print(f"Summary : {summary_file}")
    print(f"Model   : {args.model}")
    print(f"k       : {args.k if args.k is not None else 'all chunks'}")
    print(f"Force   : {args.force}")
    print(f"Workers : {args.workers}")
    print(f"Stream  : {args.stream_on}")
    print(f"Retries : {args.max_retries}")
    print(f"Retry initial wait (s): {args.retry_initial_wait}")
    print(f"Retry max wait (s)    : {args.retry_max_wait}")
    print("=" * 70)

    main_start = time.perf_counter()

    records = load_jsonl(input_file)
    updated_records, process_summary = process_records(
        records=records,
        model=args.model,
        k=args.k,
        force=args.force,
        workers=args.workers,
        max_retries=args.max_retries,
        retry_initial_wait_s=args.retry_initial_wait,
        retry_max_wait_s=args.retry_max_wait,
        stream=args.stream_on,
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
            "stream_on": args.stream_on,
            "max_retries": args.max_retries,
            "retry_initial_wait_s": args.retry_initial_wait,
            "retry_max_wait_s": args.retry_max_wait,
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