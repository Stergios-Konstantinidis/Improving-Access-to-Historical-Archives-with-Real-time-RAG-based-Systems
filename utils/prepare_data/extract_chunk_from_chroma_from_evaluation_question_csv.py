# extract_chunk_from_chroma_from_evaluation_question_csv.py
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import chromadb
from dotenv import load_dotenv
from tqdm import tqdm

from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    GoogleGenerativeAiEmbeddingFunction,
)

try:
    from google.api_core import exceptions as google_exceptions
except Exception:  # pragma: no cover - utile si le provider Google n'est pas installé
    google_exceptions = None

load_dotenv()

# =============================
# Paths
# =============================
# =============================
# Paths + CLI
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "evaluation_questions.csv"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "baseline" / "generations.jsonl"
DEFAULT_TOP_K = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Chroma chunks from evaluation questions CSV."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Input CSV containing evaluation questions.",
    )

    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of Chroma chunks to retrieve per question.",
    )

    return parser.parse_args()


args = parse_args()

CSV_PATH = args.input_csv
OUTPUT_FILE = args.output_jsonl
TOP_K = args.top_k

CHROMA_BASE_PATH = os.getenv("CHROMA_BASE_PATH", "data/chroma")
COLLECTION_NAME = os.getenv("DEFAULT_CHROMA_INDEX", "chroma_index")
CHROMA_DIR = (PROJECT_ROOT / ".." / CHROMA_BASE_PATH / COLLECTION_NAME).resolve()

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

# Limiteur utile pour les tests uniquement
EMBEDDING_REQUESTS_PER_MINUTE = int(os.getenv("EMBEDDING_REQUESTS_PER_MINUTE", "80"))

if EMBEDDING_REQUESTS_PER_MINUTE <= 0:
    raise ValueError("EMBEDDING_REQUESTS_PER_MINUTE doit être > 0")

MIN_SECONDS_BETWEEN_REQUESTS = 60.0 / EMBEDDING_REQUESTS_PER_MINUTE

# Retry embedding API: stratégie Google "truncated exponential backoff" + jitter.
# Utile notamment pour les erreurs 429 / RESOURCE_EXHAUSTED.
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "8"))
EMBEDDING_INITIAL_BACKOFF_SECONDS = float(os.getenv("EMBEDDING_INITIAL_BACKOFF_SECONDS", "1.0"))
EMBEDDING_MAX_BACKOFF_SECONDS = float(os.getenv("EMBEDDING_MAX_BACKOFF_SECONDS", "64.0"))
EMBEDDING_RETRY_DEADLINE_SECONDS = float(os.getenv("EMBEDDING_RETRY_DEADLINE_SECONDS", "300.0"))

if EMBEDDING_MAX_RETRIES < 0:
    raise ValueError("EMBEDDING_MAX_RETRIES doit être >= 0")
if EMBEDDING_INITIAL_BACKOFF_SECONDS <= 0:
    raise ValueError("EMBEDDING_INITIAL_BACKOFF_SECONDS doit être > 0")
if EMBEDDING_MAX_BACKOFF_SECONDS <= 0:
    raise ValueError("EMBEDDING_MAX_BACKOFF_SECONDS doit être > 0")
if EMBEDDING_RETRY_DEADLINE_SECONDS <= 0:
    raise ValueError("EMBEDDING_RETRY_DEADLINE_SECONDS doit être > 0")


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute doit être > 0")
        self.min_interval = 60.0 / requests_per_minute
        self.last_call_time = None

    def wait(self) -> float:
        """
        Attend si nécessaire et retourne le temps d'attente réel en secondes.
        """
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


def build_embedding_function():
    """
    Construit automatiquement la bonne embedding function à partir du .env.
    Si EMBEDDING_PROVIDER est absent, on essaie d'inférer depuis le nom de collection.
    """
    provider = EMBEDDING_PROVIDER

    if not provider:
        lowered = COLLECTION_NAME.lower()
        if "gemini" in lowered or "google" in lowered:
            provider = "google"
        elif "openai" in lowered:
            provider = "openai"

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

    raise ValueError(
        f"EMBEDDING_PROVIDER invalide: '{EMBEDDING_PROVIDER}'. "
        f"Valeurs attendues: 'openai' ou 'google'."
    )


def is_retryable_embedding_error(exc: Exception) -> bool:
    """
    Erreurs transitoires à retenter.
    Couvre Google 429 RESOURCE_EXHAUSTED et quelques erreurs réseau/serveur.
    """
    if google_exceptions is not None:
        retryable_google_errors = tuple(
            cls
            for cls in (
                getattr(google_exceptions, "ResourceExhausted", None),
                getattr(google_exceptions, "ServiceUnavailable", None),
                getattr(google_exceptions, "DeadlineExceeded", None),
                getattr(google_exceptions, "InternalServerError", None),
                getattr(google_exceptions, "TooManyRequests", None),
            )
            if cls is not None
        )
        if retryable_google_errors and isinstance(exc, retryable_google_errors):
            return True

    # Fallback générique, utile si l'exception vient d'une dépendance différente.
    class_name = exc.__class__.__name__.lower()
    message = str(exc).lower()

    retryable_names = (
        "ratelimit",
        "toomanyrequests",
        "resourceexhausted",
        "serviceunavailable",
        "deadlineexceeded",
        "timeout",
        "temporarilyunavailable",
    )

    retryable_messages = (
        "429",
        "resource exhausted",
        "too many requests",
        "rate limit",
        "quota",
        "try again later",
        "temporarily unavailable",
        "deadline exceeded",
        "service unavailable",
    )

    return any(x in class_name for x in retryable_names) or any(
        x in message for x in retryable_messages
    )


def compute_retry_sleep_seconds(attempt_index: int) -> float:
    """
    Backoff exponentiel tronqué avec jitter.
    attempt_index commence à 0 pour le premier retry après le premier échec.
    Exemple: 1+jitter, 2+jitter, 4+jitter, ... plafonné à EMBEDDING_MAX_BACKOFF_SECONDS.
    """
    exponential_wait = EMBEDDING_INITIAL_BACKOFF_SECONDS * (2 ** attempt_index)
    jitter = random.random()  # fraction aléatoire entre 0 et 1
    return min(exponential_wait + jitter, EMBEDDING_MAX_BACKOFF_SECONDS)


def call_embedding_with_retry(
    embedding_fn,
    text: str,
    rate_limiter: RateLimiter,
) -> Tuple[Any, Dict[str, float]]:
    """
    Appelle embedding_fn([text]) avec retry sur erreurs transitoires.
    Retourne l'embedding brut + timings détaillés.
    """
    attempts = 0
    total_rate_limit_wait_ms = 0.0
    total_embedding_api_ms = 0.0
    total_retry_sleep_ms = 0.0
    retry_started_at = time.monotonic()

    while True:
        attempts += 1

        wait_seconds = rate_limiter.wait()
        total_rate_limit_wait_ms += wait_seconds * 1000

        embedding_api_start = time.perf_counter()
        try:
            embedding = embedding_fn([text])
            total_embedding_api_ms += (time.perf_counter() - embedding_api_start) * 1000

            timing = {
                "rate_limit_wait_ms": round(total_rate_limit_wait_ms, 2),
                "embedding_api_ms": round(total_embedding_api_ms, 2),
                "embedding_retry_sleep_ms": round(total_retry_sleep_ms, 2),
                "embedding_attempts": attempts,
            }
            return embedding, timing

        except Exception as exc:
            total_embedding_api_ms += (time.perf_counter() - embedding_api_start) * 1000

            if not is_retryable_embedding_error(exc):
                raise

            retries_done = attempts - 1
            elapsed = time.monotonic() - retry_started_at

            if retries_done >= EMBEDDING_MAX_RETRIES:
                raise RuntimeError(
                    f"Embedding API échoué après {attempts} tentative(s) "
                    f"({EMBEDDING_MAX_RETRIES} retry max). Dernière erreur: {exc}"
                ) from exc

            sleep_seconds = compute_retry_sleep_seconds(retries_done)

            if elapsed + sleep_seconds > EMBEDDING_RETRY_DEADLINE_SECONDS:
                raise RuntimeError(
                    f"Embedding API échoué: deadline retry dépassée "
                    f"({EMBEDDING_RETRY_DEADLINE_SECONDS:.1f}s). "
                    f"Tentatives effectuées: {attempts}. Dernière erreur: {exc}"
                ) from exc

            print(
                "\n[RETRY embedding] "
                f"tentative {attempts} échouée: {exc.__class__.__name__}: {exc}. "
                f"Nouvelle tentative dans {sleep_seconds:.2f}s..."
            )
            time.sleep(sleep_seconds)
            total_retry_sleep_ms += sleep_seconds * 1000


def extract_query_embedding(
    embedding_fn,
    text: str,
    rate_limiter: RateLimiter,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Retourne:
    - l'embedding
    - un dictionnaire de timings en millisecondes
    """
    embedding, timing = call_embedding_with_retry(
        embedding_fn=embedding_fn,
        text=text,
        rate_limiter=rate_limiter,
    )

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

    return final_embedding, timing


def build_retrieved_chunks(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    documents = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    chunks: List[Dict[str, Any]] = []

    for rank, doc in enumerate(documents):
        meta = metadatas[rank] if rank < len(metadatas) and isinstance(metadatas[rank], dict) else {}

        chunks.append(
            {
                "id": ids[rank] if rank < len(ids) else None,
                "text": doc,
                "metadata": meta,
                "distance": distances[rank] if rank < len(distances) else None,
                "rank": rank,
            }
        )

    return chunks


print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"CSV_PATH = {CSV_PATH}")
print(f"OUTPUT_FILE = {OUTPUT_FILE}")
print(f"CHROMA_BASE_PATH = {CHROMA_BASE_PATH}")
print(f"COLLECTION_NAME = {COLLECTION_NAME}")
print(f"CHROMA_DIR = {CHROMA_DIR}")
print(f"CHROMA_DIR exists = {CHROMA_DIR.exists()}")
print(f"EMBEDDING_PROVIDER = {EMBEDDING_PROVIDER}")
print(f"EMBEDDING_REQUESTS_PER_MINUTE = {EMBEDDING_REQUESTS_PER_MINUTE}")
print(f"MIN_SECONDS_BETWEEN_REQUESTS = {MIN_SECONDS_BETWEEN_REQUESTS:.3f}")
print(f"EMBEDDING_MAX_RETRIES = {EMBEDDING_MAX_RETRIES}")
print(f"EMBEDDING_INITIAL_BACKOFF_SECONDS = {EMBEDDING_INITIAL_BACKOFF_SECONDS:.3f}")
print(f"EMBEDDING_MAX_BACKOFF_SECONDS = {EMBEDDING_MAX_BACKOFF_SECONDS:.3f}")
print(f"EMBEDDING_RETRY_DEADLINE_SECONDS = {EMBEDDING_RETRY_DEADLINE_SECONDS:.3f}")

if not CHROMA_DIR.exists():
    raise ValueError(f"Le dossier Chroma n'existe pas: {CHROMA_DIR}")

embedding_fn = build_embedding_function()
rate_limiter = RateLimiter(EMBEDDING_REQUESTS_PER_MINUTE)

print("\n[EMBEDDING CHECK BASELINE]")
print(f"collection_name: {COLLECTION_NAME}")
print(f"embedding_fn_type: {type(embedding_fn).__name__}")
print(f"embedding_model_name: {getattr(embedding_fn, 'model_name', None)}")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collections = client.list_collections()

print("\nCollections disponibles :")
for c in collections:
    print(f"- {c.name}")

collection_names = [c.name for c in collections]
if COLLECTION_NAME not in collection_names:
    raise ValueError(
        f"La collection '{COLLECTION_NAME}' n'existe pas dans '{CHROMA_DIR}'. "
        f"Collections disponibles: {collection_names}"
    )

collection = client.get_collection(name=COLLECTION_NAME)
print(f"\nCollection chargée : {collection.name}")

df = pd.read_csv(CSV_PATH)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

missing_questions = 0
processed_questions = 0

# Cumuls globaux
total_rate_limit_wait_ms = 0.0
total_embedding_api_ms = 0.0
total_chroma_query_ms = 0.0
total_real_pipeline_ms = 0.0  # embedding_api_ms + chroma_query_ms

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Génération JSONL"):
        question = str(row["question"]).strip()
        groundtruth = row["groundtruth"]

        if not question:
            missing_questions += 1
            continue

        query_embedding, embedding_timing = extract_query_embedding(
            embedding_fn,
            question,
            rate_limiter,
        )

        if idx == 0:
            print(f"first_question: {question}")

        chroma_start = time.perf_counter()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "distances", "metadatas"],
        )
        chroma_query_ms = (time.perf_counter() - chroma_start) * 1000

        retrieved_chunks = build_retrieved_chunks(results)

        real_pipeline_ms = embedding_timing["embedding_api_ms"] + round(chroma_query_ms, 2)

        # Cumuls
        total_rate_limit_wait_ms += embedding_timing["rate_limit_wait_ms"]
        total_embedding_api_ms += embedding_timing["embedding_api_ms"]
        total_chroma_query_ms += round(chroma_query_ms, 2)
        total_real_pipeline_ms += real_pipeline_ms
        processed_questions += 1

        output_entry = {
            "question_id": f"q_{idx:04d}",
            "question": question,
            "generated_answer": groundtruth,
            "retrieved_chunks": retrieved_chunks,
            "method_metadata": {
                "reranker_name": "default",
                "top_k": TOP_K,
                "fetch_k": TOP_K,
                "model_name": "rag_v1",
                "embedding_provider": EMBEDDING_PROVIDER or "inferred",
                "embedding_model": (
                    GOOGLE_EMBEDDING_MODEL
                    if (EMBEDDING_PROVIDER == "google" or "gemini" in COLLECTION_NAME.lower())
                    else OPENAI_EMBEDDING_MODEL
                ),
                "embedding_requests_per_minute": EMBEDDING_REQUESTS_PER_MINUTE,
                "endpoint": None,
            },
            "latency_ms": {
                "rate_limit_wait_ms": embedding_timing["rate_limit_wait_ms"],
                "embedding_api_ms": embedding_timing["embedding_api_ms"],
                "embedding_retry_sleep_ms": embedding_timing.get("embedding_retry_sleep_ms", 0.0),
                "embedding_attempts": embedding_timing.get("embedding_attempts", 1),
                "chroma_query_ms": round(chroma_query_ms, 2),
                "real_pipeline_ms": round(real_pipeline_ms, 2),
            },
        }

        f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

print(f"\nFichier généré : {OUTPUT_FILE}")
if missing_questions:
    print(f"Questions vides ignorées : {missing_questions}")

print("\n===== RÉSUMÉ GLOBAL =====")
print(f"Questions traitées : {processed_questions}")
print(f"Somme rate limit wait (test seulement) : {total_rate_limit_wait_ms:.2f} ms ({total_rate_limit_wait_ms / 1000:.2f} s)")
print(f"Somme embedding API réelle        : {total_embedding_api_ms:.2f} ms ({total_embedding_api_ms / 1000:.2f} s)")
print(f"Somme requêtes Chroma réelle      : {total_chroma_query_ms:.2f} ms ({total_chroma_query_ms / 1000:.2f} s)")
print(f"Somme pipeline réel total         : {total_real_pipeline_ms:.2f} ms ({total_real_pipeline_ms / 1000:.2f} s)")

if processed_questions > 0:
    print("\n===== MOYENNES PAR QUESTION =====")
    print(f"Moyenne embedding API réelle   : {total_embedding_api_ms / processed_questions:.2f} ms")
    print(f"Moyenne requête Chroma réelle  : {total_chroma_query_ms / processed_questions:.2f} ms")
    print(f"Moyenne pipeline réel total    : {total_real_pipeline_ms / processed_questions:.2f} ms")