#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    GoogleGenerativeAiEmbeddingFunction,
)

load_dotenv()

# =============================
# Paths
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "evaluation_questions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "embeddings" / "evaluation_questions_embeddings.jsonl"

# =============================
# Env helpers
# =============================
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large").strip()

GOOGLE_EMBEDDING_API_KEY = os.getenv("GOOGLE_EMBEDDING_API_KEY", "").strip()
GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"
).strip()

EMBEDDING_REQUESTS_PER_MINUTE = int(os.getenv("EMBEDDING_REQUESTS_PER_MINUTE", "80"))
if EMBEDDING_REQUESTS_PER_MINUTE <= 0:
    raise ValueError("EMBEDDING_REQUESTS_PER_MINUTE doit être > 0")


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute doit être > 0")
        self.min_interval = 60.0 / requests_per_minute
        self.last_call_time: Optional[float] = None

    def wait(self) -> float:
        now = time.monotonic()

        if self.last_call_time is None:
            self.last_call_time = now
            return 0.0

        elapsed = now - self.last_call_time
        remaining = self.min_interval - elapsed

        waited = 0.0
        if remaining > 0:
            time.sleep(remaining)
            waited = remaining

        self.last_call_time = time.monotonic()
        return waited


def resolve_embedding_provider() -> str:
    if EMBEDDING_PROVIDER not in {"openai", "google"}:
        raise ValueError(
            f"EMBEDDING_PROVIDER invalide: '{EMBEDDING_PROVIDER}'. "
            f"Valeurs attendues: 'openai' ou 'google'."
        )
    return EMBEDDING_PROVIDER


def resolve_embedding_model(provider: str) -> str:
    if provider == "openai":
        return OPENAI_EMBEDDING_MODEL
    if provider == "google":
        return GOOGLE_EMBEDDING_MODEL
    raise ValueError(f"Provider non supporté: {provider}")


def build_embedding_function():
    provider = resolve_embedding_provider()

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY manquant dans le .env")
        return OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL,
        )

    if provider == "google":
        if not GOOGLE_EMBEDDING_API_KEY:
            raise ValueError("GOOGLE_EMBEDDING_API_KEY manquant dans le .env")
        return GoogleGenerativeAiEmbeddingFunction(
            api_key=GOOGLE_EMBEDDING_API_KEY,
            model_name=GOOGLE_EMBEDDING_MODEL,
        )

    raise ValueError("Provider d'embedding non supporté.")


def extract_query_embedding(
    embedding_fn,
    text: str,
    rate_limiter: RateLimiter,
) -> Tuple[List[float], Dict[str, float]]:
    wait_seconds = rate_limiter.wait()
    rate_limit_wait_ms = wait_seconds * 1000.0

    embedding_api_start = time.perf_counter()
    embedding = embedding_fn([text])
    embedding_api_ms = (time.perf_counter() - embedding_api_start) * 1000.0

    if embedding is None:
        raise ValueError("Embedding vide renvoyé par la fonction d'embedding.")

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if isinstance(embedding, list) and len(embedding) == 0:
        raise ValueError("Embedding vide renvoyé par la fonction d'embedding.")

    first = embedding[0] if isinstance(embedding, list) and len(embedding) > 0 else None

    if hasattr(first, "tolist"):
        first = first.tolist()

    if isinstance(first, list):
        final_embedding = [float(x) for x in first]
    elif isinstance(embedding, list) and isinstance(first, (int, float)):
        final_embedding = [float(x) for x in embedding]
    else:
        raise ValueError(
            f"Format d'embedding inattendu. Type={type(embedding)}, valeur={repr(embedding)[:500]}"
        )

    timing = {
        "rate_limit_wait_ms": round(rate_limit_wait_ms, 2),
        "embedding_api_ms": round(embedding_api_ms, 2),
    }
    return final_embedding, timing


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def build_embedding_key(text: str, provider: str, model: str) -> str:
    raw = f"{provider}::{model}::{normalize_text(text)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def unique_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère un fichier JSONL d'embeddings à partir de data/evaluation_questions.csv"
    )
    parser.add_argument(
        "--input-csv",
        default=str(CSV_PATH),
        help="CSV d'entrée (défaut: data/evaluation_questions.csv)",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(OUTPUT_FILE),
        help="JSONL de sortie (défaut: data/embeddings/evaluation_questions_embeddings.jsonl)",
    )
    parser.add_argument(
        "--question-column",
        default="question",
        help="Nom de la colonne contenant les questions",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Déduplique les questions avant génération",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_jsonl = Path(args.output_jsonl)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV introuvable: {input_csv}")

    provider = resolve_embedding_provider()
    model = resolve_embedding_model(provider)

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"INPUT_CSV = {input_csv}")
    print(f"OUTPUT_JSONL = {output_jsonl}")
    print(f"QUESTION_COLUMN = {args.question_column}")
    print(f"EMBEDDING_PROVIDER = {provider}")
    print(f"EMBEDDING_MODEL = {model}")
    print(f"EMBEDDING_REQUESTS_PER_MINUTE = {EMBEDDING_REQUESTS_PER_MINUTE}")

    df = pd.read_csv(input_csv)

    if args.question_column not in df.columns:
        raise ValueError(
            f"Colonne '{args.question_column}' introuvable dans le CSV. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    questions = [
        normalize_text(str(value))
        for value in df[args.question_column].tolist()
        if normalize_text(str(value))
    ]

    if not questions:
        raise RuntimeError("Aucune question valide trouvée dans le CSV.")

    if args.deduplicate:
        questions = unique_keep_order(questions)

    embedding_fn = build_embedding_function()
    rate_limiter = RateLimiter(EMBEDDING_REQUESTS_PER_MINUTE)

    rows: List[Dict] = []
    total_rate_limit_wait_ms = 0.0
    total_embedding_api_ms = 0.0

    for idx, question in enumerate(tqdm(questions, desc="Génération embeddings", unit="q")):
        embedding, timing = extract_query_embedding(
            embedding_fn=embedding_fn,
            text=question,
            rate_limiter=rate_limiter,
        )

        total_rate_limit_wait_ms += timing["rate_limit_wait_ms"]
        total_embedding_api_ms += timing["embedding_api_ms"]

        rows.append(
            {
                "question_id": f"q_{idx:04d}",
                "key": build_embedding_key(question, provider, model),
                "question": question,
                "provider": provider,
                "model": model,
                "embedding_dim": len(embedding),
                "embedding": embedding,
            }
        )

    write_jsonl(output_jsonl, rows)

    print(f"\nFichier généré : {output_jsonl}")
    print("\n===== RÉSUMÉ =====")
    print(f"Questions traitées           : {len(rows)}")
    print(f"Somme rate limit wait        : {total_rate_limit_wait_ms:.2f} ms ({total_rate_limit_wait_ms / 1000:.2f} s)")
    print(f"Somme embedding API          : {total_embedding_api_ms:.2f} ms ({total_embedding_api_ms / 1000:.2f} s)")

    if len(rows) > 0:
        print("\n===== MOYENNES PAR QUESTION =====")
        print(f"Moyenne embedding API        : {total_embedding_api_ms / len(rows):.2f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())