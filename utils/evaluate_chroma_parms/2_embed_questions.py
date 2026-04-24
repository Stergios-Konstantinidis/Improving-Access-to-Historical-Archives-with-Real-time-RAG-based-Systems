from __future__ import annotations

"""
Generate question embeddings from a potentially large JSONL file.

Input:
    questions.jsonl
        one JSON object per line, containing at least:
        {
            "question_id": "...",
            "question": "..."
        }

Outputs:
    Q.npy
        L2-normalized question embeddings, shape (Nq, d)
    question_ids.jsonl
        one JSON string per line, aligned with Q.npy row index
    questions_clean.jsonl
        one JSON string per line, aligned with Q.npy row index
    question_matrix_meta.json
        small metadata file

Environment variables:
    ENV_FILE
    QUESTIONS_JSONL
    OUT_DIR
    QUESTION_BATCH_SIZE
    NORMALIZE_QUESTION_EMBEDDINGS

    QUESTIONS_EMBEDDING_PROVIDER   (optional, overrides DEFAULT_EMBEDDING_PROVIDER)
    DEFAULT_EMBEDDING_PROVIDER

    OPENAI_API_KEY
    OPENAI_BASE_URL
    OPENAI_EMBEDDING_MODEL

    OPENROUTER_API_KEY
    OPENROUTER_BASE_URL
    OPENROUTER_EMBEDDING_MODEL

    GOOGLE_EMBEDDING_API_KEY
    GOOGLE_EMBEDDING_MODEL
"""

import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from chromadb.utils.embedding_functions import (
        OpenAIEmbeddingFunction,
        GoogleGenerativeAiEmbeddingFunction,
    )
except Exception:
    OpenAIEmbeddingFunction = None
    GoogleGenerativeAiEmbeddingFunction = None


def load_env(env_path: str | Path | None = None) -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path, override=False)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def normalize_rows_inplace(x: np.ndarray) -> None:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x /= norms


def count_questions(jsonl_path: Path) -> int:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def stream_questions(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Ligne {line_no} invalide dans {jsonl_path}: {e}")

            question_id = str(obj.get("question_id", f"line_{line_no}"))
            question = obj.get("question")

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Ligne {line_no}: champ 'question' manquant ou vide")

            yield question_id, question.strip()


def normalize_google_model_name(model_name: str) -> str:
    """
    Accepte:
      - models/gemini-embedding-2-preview
      - gemini-embedding-2-preview
    et retourne toujours la forme la plus sûre pour Chroma.
    """
    model_name = (model_name or "").strip()
    if not model_name:
        return "gemini-embedding-2-preview"
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]
    return model_name


def ensure_google_api_key_envs(api_key: str) -> None:
    """
    Certaines libs lisent la clé passée explicitement.
    D'autres essaient aussi/à la place de lire GEMINI_API_KEY ou GOOGLE_API_KEY.
    On aligne tout pour éviter les erreurs 'API Key not found'.
    """
    if not api_key:
        return

    os.environ["GOOGLE_EMBEDDING_API_KEY"] = api_key

    if not os.getenv("GEMINI_API_KEY", "").strip():
        os.environ["GEMINI_API_KEY"] = api_key

    if not os.getenv("GOOGLE_API_KEY", "").strip():
        os.environ["GOOGLE_API_KEY"] = api_key


def make_embedding_function():
    provider = os.getenv("QUESTIONS_EMBEDDING_PROVIDER", "").strip().lower()
    if not provider:
        provider = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "openai").strip().lower()

    if provider == "google":
        if GoogleGenerativeAiEmbeddingFunction is None:
            raise RuntimeError("GoogleGenerativeAiEmbeddingFunction indisponible.")

        api_key = os.getenv("GOOGLE_EMBEDDING_API_KEY", "").strip()
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY", "").strip()

        raw_model_name = os.getenv(
            "GOOGLE_EMBEDDING_MODEL",
            "gemini-embedding-2-preview",
        ).strip()
        model_name = normalize_google_model_name(raw_model_name)

        if not api_key:
            raise ValueError(
                "Clé API Google manquante. "
                "Définis GOOGLE_EMBEDDING_API_KEY dans le .env "
                "(ou GEMINI_API_KEY / GOOGLE_API_KEY)."
            )

        ensure_google_api_key_envs(api_key)

        return provider, model_name, GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
        )

    if provider == "openai":
        if OpenAIEmbeddingFunction is None:
            raise RuntimeError("OpenAIEmbeddingFunction indisponible.")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY manquante")
        return provider, model_name, OpenAIEmbeddingFunction(
            api_key=api_key,
            api_base=base_url,
            model_name=model_name,
        )

    if provider == "openrouter":
        if OpenAIEmbeddingFunction is None:
            raise RuntimeError("OpenAIEmbeddingFunction indisponible.")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        model_name = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-large").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY manquante")
        return provider, model_name, OpenAIEmbeddingFunction(
            api_key=api_key,
            api_base=base_url,
            model_name=model_name,
        )

    raise ValueError(
        f"Unknown embedding provider: {provider}. "
        f"Expected one of: google, openai, openrouter."
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = (PROJECT_ROOT / ".env").resolve()

load_env(os.getenv("ENV_FILE", str(ENV_PATH)))

print(ENV_PATH)

DEFAULT_QUESTIONS_JSONL = PROJECT_ROOT / "api_endpoint" / "data" / "baseline" / "generations.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"


def write_batch(
    ef,
    q_memmap,
    q_path: Path,
    batch_ids: list[str],
    batch_questions: list[str],
    ids_out,
    questions_out,
    written: int,
    total_questions: int,
    dim: int | None,
    normalize_questions: bool,
):
    arr = np.asarray(ef(batch_questions), dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Embeddings questions invalides: shape={arr.shape}")

    if q_memmap is None:
        dim = arr.shape[1]
        q_memmap = np.lib.format.open_memmap(
            q_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_questions, dim),
        )

    if arr.shape[1] != dim:
        raise ValueError(f"Dimension incohérente: attendu {dim}, reçu {arr.shape[1]}")

    if normalize_questions:
        normalize_rows_inplace(arr)

    end = written + arr.shape[0]
    q_memmap[written:end] = arr

    for qid in batch_ids:
        ids_out.write(json.dumps(qid, ensure_ascii=False))
        ids_out.write("\n")

    for qtxt in batch_questions:
        questions_out.write(json.dumps(qtxt, ensure_ascii=False))
        questions_out.write("\n")

    return q_memmap, dim, end


def main() -> None:
    questions_jsonl = Path(
        os.getenv("QUESTIONS_JSONL", str(DEFAULT_QUESTIONS_JSONL))
    )
    if not questions_jsonl.is_absolute():
        questions_jsonl = (PROJECT_ROOT / questions_jsonl).resolve()

    out_dir = Path(
        os.getenv("OUT_DIR", str(DEFAULT_OUT_DIR))
    )
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    q_path = out_dir / "Q.npy"
    question_ids_jsonl = out_dir / "question_ids.jsonl"
    questions_clean_jsonl = out_dir / "questions_clean.jsonl"
    meta_path = out_dir / "question_matrix_meta.json"

    batch_size = env_int("QUESTION_BATCH_SIZE", 64)
    normalize_questions = env_bool("NORMALIZE_QUESTION_EMBEDDINGS", True)

    provider, model_name, ef = make_embedding_function()

    print(f".env utilisé: {ENV_PATH}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"questions_jsonl: {questions_jsonl}")
    print(f"out_dir: {out_dir}")
    print(f"QUESTION_BATCH_SIZE: {batch_size}")
    print(f"NORMALIZE_QUESTION_EMBEDDINGS: {normalize_questions}")
    print(f"Embedding provider: {provider}")
    print(f"Embedding model: {model_name}")

    if provider == "google":
        print("GOOGLE_EMBEDDING_API_KEY present:", bool(os.getenv("GOOGLE_EMBEDDING_API_KEY", "").strip()))
        print("GEMINI_API_KEY present:", bool(os.getenv("GEMINI_API_KEY", "").strip()))
        print("GOOGLE_API_KEY present:", bool(os.getenv("GOOGLE_API_KEY", "").strip()))

    if not questions_jsonl.exists():
        raise FileNotFoundError(f"Fichier introuvable: {questions_jsonl}")

    total_questions = count_questions(questions_jsonl)
    if total_questions == 0:
        raise ValueError("Le fichier questions JSONL est vide.")

    q_memmap = None
    dim = None
    written = 0

    batch_ids: list[str] = []
    batch_questions: list[str] = []

    with question_ids_jsonl.open("w", encoding="utf-8") as ids_out, questions_clean_jsonl.open(
        "w", encoding="utf-8"
    ) as questions_out:
        for question_id, question in tqdm(
            stream_questions(questions_jsonl),
            total=total_questions,
            desc="Reading + embedding questions",
            unit="q",
        ):
            batch_ids.append(question_id)
            batch_questions.append(question)

            if len(batch_questions) >= batch_size:
                q_memmap, dim, written = write_batch(
                    ef=ef,
                    q_memmap=q_memmap,
                    q_path=q_path,
                    batch_ids=batch_ids,
                    batch_questions=batch_questions,
                    ids_out=ids_out,
                    questions_out=questions_out,
                    written=written,
                    total_questions=total_questions,
                    dim=dim,
                    normalize_questions=normalize_questions,
                )
                batch_ids = []
                batch_questions = []

        if batch_questions:
            q_memmap, dim, written = write_batch(
                ef=ef,
                q_memmap=q_memmap,
                q_path=q_path,
                batch_ids=batch_ids,
                batch_questions=batch_questions,
                ids_out=ids_out,
                questions_out=questions_out,
                written=written,
                total_questions=total_questions,
                dim=dim,
                normalize_questions=normalize_questions,
            )

    if q_memmap is None or dim is None:
        raise ValueError("Aucune embedding question n'a été générée.")

    if written != total_questions:
        raise ValueError(f"Nombre de questions écrites incorrect: {written} au lieu de {total_questions}")

    q_memmap.flush()

    meta = {
        "source_questions_jsonl": str(questions_jsonl),
        "q_path": str(q_path),
        "question_ids_jsonl": str(question_ids_jsonl),
        "questions_clean_jsonl": str(questions_clean_jsonl),
        "num_questions": total_questions,
        "embedding_dim": dim,
        "dtype": "float32",
        "normalized": normalize_questions,
        "embedding_provider": provider,
        "embedding_model": model_name,
        "question_index_alignment": "line i in question_ids.jsonl / questions_clean.jsonl <-> row i in Q.npy",
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Saved:")
    print(" -", q_path)
    print(" -", question_ids_jsonl)
    print(" -", questions_clean_jsonl)
    print(" -", meta_path)
    print("Q shape:", (total_questions, dim))


if __name__ == "__main__":
    main()