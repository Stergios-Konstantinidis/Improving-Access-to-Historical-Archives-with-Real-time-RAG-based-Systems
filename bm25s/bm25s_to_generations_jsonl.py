#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import bm25s
except Exception as e:
    raise RuntimeError(
        "Le package 'bm25s' n'est pas installé. Installe-le avec: pip install -U bm25s"
    ) from e

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def progress(iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_raw(path: Path) -> List[Any]:
    rows: List[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide dans {path} à la ligne {line_no}: {exc}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes} min {remaining:.2f} s"


def load_corpus_from_json(export_json_path: Path) -> Dict[str, List[Any]]:
    data = read_json(export_json_path)

    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    if not isinstance(ids, list):
        raise ValueError("'ids' doit être une liste")
    if not isinstance(documents, list):
        raise ValueError("'documents' doit être une liste")
    if not isinstance(metadatas, list):
        raise ValueError("'metadatas' doit être une liste")

    if len(ids) != len(documents):
        raise ValueError(
            f"Longueurs incohérentes: len(ids)={len(ids)} != len(documents)={len(documents)}"
        )

    if len(metadatas) != len(documents):
        if len(metadatas) == 0:
            metadatas = [{} for _ in documents]
        else:
            raise ValueError(
                f"Longueurs incohérentes: len(metadatas)={len(metadatas)} != len(documents)={len(documents)}"
            )

    normalized_docs = [normalize_text(doc) for doc in documents]
    normalized_metas = [safe_metadata(md) for md in metadatas]
    normalized_ids = [str(_id) for _id in ids]

    return {
        "ids": normalized_ids,
        "documents": normalized_docs,
        "metadatas": normalized_metas,
    }


def load_corpus_from_jsonl(export_jsonl_path: Path) -> Dict[str, List[Any]]:
    """
    Accepte des lignes JSONL de type:
    1) "texte du document"
    2) {"id": "...", "text": "...", "metadata": {...}}
    3) {"id": "...", "document": "...", "metadata": {...}}
    4) {"ids": "...", "documents": "...", "metadatas": {...}}
    """
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    with export_jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(progress(f, desc="Chargement corpus JSONL", unit="doc"), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide dans {export_jsonl_path} à la ligne {i}: {exc}"
                ) from exc

            if isinstance(row, str):
                text = normalize_text(row)
                if not text:
                    continue

                ids.append(f"chunk_{len(ids):06d}")
                documents.append(text)
                metadatas.append({})
                continue

            if isinstance(row, dict):
                _id = row.get("id")
                if _id is None:
                    _id = row.get("ids")
                if _id is None:
                    _id = f"chunk_{len(ids):06d}"

                text = row.get("text")
                if text is None:
                    text = row.get("document")
                if text is None:
                    text = row.get("documents")

                text = normalize_text(text)
                if not text:
                    continue

                metadata = row.get("metadata")
                if metadata is None:
                    metadata = row.get("metadatas")
                metadata = safe_metadata(metadata)

                ids.append(str(_id))
                documents.append(text)
                metadatas.append(metadata)
                continue

            raise ValueError(
                f"Ligne {i} non supportée dans {export_jsonl_path}: type {type(row).__name__}"
            )

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }


def load_corpus(path: Path) -> Dict[str, List[Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return load_corpus_from_jsonl(path)

    if suffix == ".json":
        return load_corpus_from_json(path)

    raise ValueError(
        f"Format non supporté pour le corpus: {path}. Utilise .json ou .jsonl"
    )


def load_questions(questions_jsonl_path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl_raw(questions_jsonl_path)
    out: List[Dict[str, Any]] = []

    for i, row in enumerate(progress(rows, desc="Chargement questions", unit="q")):
        if not isinstance(row, dict):
            continue

        qid = row.get("question_id") or f"q_{i:04d}"
        question = normalize_text(row.get("question", ""))

        if not question:
            continue

        out.append({
            "question_id": str(qid),
            "question": question,
            "category": row.get("category", "nan"),
            "reference": row.get("reference", "nan"),
            "generated_answer": row.get("generated_answer", ""),
            "retrieved_chunks": row.get("retrieved_chunks", []),
            "raw": row,
        })

    return out


class BM25sRetriever:
    def __init__(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        if len(ids) != len(documents) or len(documents) != len(metadatas):
            raise ValueError("ids, documents et metadatas doivent avoir la même longueur")

        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas

        print("Tokenization BM25s du corpus...")
        self.tokens = bm25s.tokenize(documents)

        print("Indexation BM25s...")
        self.retriever = bm25s.BM25()
        self.retriever.index(self.tokens)

    def search(self, query: str, top_k: int) -> Dict[str, Any]:
        t0 = now_ms()

        query_tokens = bm25s.tokenize([query])

        results, scores = self.retriever.retrieve(
            query_tokens,
            corpus=self.documents,
            k=top_k,
        )

        retrieval_ms = now_ms() - t0

        result_docs = list(results[0])
        result_scores = [float(x) for x in scores[0]]

        positions_by_text: Dict[str, List[int]] = {}
        for idx, text in enumerate(self.documents):
            positions_by_text.setdefault(text, []).append(idx)

        used_positions = set()
        hits = []

        for rank, (doc_text, score) in enumerate(zip(result_docs, result_scores)):
            candidate_positions = positions_by_text.get(doc_text, [])

            chosen_pos: Optional[int] = None
            for pos in candidate_positions:
                if pos not in used_positions:
                    chosen_pos = pos
                    used_positions.add(pos)
                    break

            if chosen_pos is None:
                continue

            metadata = dict(self.metadatas[chosen_pos])
            metadata["score"] = score

            hits.append({
                "id": self.ids[chosen_pos],
                "text": self.documents[chosen_pos],
                "metadata": metadata,
                "distance": score,
                "rank": rank,
            })

        return {
            "hits": hits,
            "retrieval_ms": retrieval_ms,
        }


def build_generation_row(
    question_row: Dict[str, Any],
    hits: List[Dict[str, Any]],
    top_k: int,
    retrieval_ms: float,
) -> Dict[str, Any]:
    return {
        "question_id": question_row["question_id"],
        "question": question_row["question"],
        "category": question_row.get("category", "nan"),
        "reference": question_row.get("reference", "nan"),
        "generated_answer": "",
        "retrieved_chunks": hits,
        "method_metadata": {
            "provider": "bm25s",
            "top_k": top_k,
        },
        "latency_ms": retrieval_ms,
        "latency_breakdown_ms": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": 0.0,
        },
        "question_processing_time_ms": retrieval_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère un fichier gnerations.jsonl à partir d'un corpus texte en utilisant BM25s."
    )

    parser.add_argument(
        "--corpus-json",
        required=True,
        help="Chemin vers le corpus (.json ou .jsonl)",
    )
    parser.add_argument(
        "--questions-jsonl",
        required=True,
        help="Chemin vers le fichier de questions JSONL",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Chemin du fichier generations/gnerations.jsonl de sortie",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Nombre de chunks récupérés par question",
    )

    return parser.parse_args()


def main() -> int:
    global_start_ms = now_ms()

    args = parse_args()

    corpus_path = Path(args.corpus_json)
    questions_path = Path(args.questions_jsonl)
    output_path = Path(args.output_jsonl)

    print("Chargement du corpus...")
    corpus = load_corpus(corpus_path)

    print("Chargement des questions...")
    questions = load_questions(questions_path)

    if not corpus["documents"]:
        raise RuntimeError("Le corpus est vide.")
    if not questions:
        raise RuntimeError("Aucune question valide trouvée.")

    print(f"Corpus chargé : {len(corpus['documents'])} documents")
    print(f"Questions chargées : {len(questions)}")

    retriever = BM25sRetriever(
        ids=corpus["ids"],
        documents=corpus["documents"],
        metadatas=corpus["metadatas"],
    )

    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []

    retrieval_batch_start_ms = now_ms()

    for q in progress(questions, desc="BM25 retrieval", unit="q"):
        result = retriever.search(q["question"], top_k=args.top_k)

        row = build_generation_row(
            question_row=q,
            hits=result["hits"],
            top_k=args.top_k,
            retrieval_ms=result["retrieval_ms"],
        )

        rows.append(row)
        latencies.append(result["retrieval_ms"])

    retrieval_batch_end_ms = now_ms()

    write_jsonl(output_path, rows)

    global_end_ms = now_ms()

    total_retrieval_seconds = (retrieval_batch_end_ms - retrieval_batch_start_ms) / 1000.0
    total_program_seconds = (global_end_ms - global_start_ms) / 1000.0

    print(f"\nFichier généré : {output_path}")
    print(f"Questions traitées : {len(rows)}")

    if latencies:
        print(f"Latency moyenne retrieval : {statistics.mean(latencies):.2f} ms")
        print(f"Latency médiane retrieval : {statistics.median(latencies):.2f} ms")

    print(f"Temps total pour exécuter toutes les requêtes de questions : {format_duration(total_retrieval_seconds)}")
    print(f"Temps total du programme : {format_duration(total_program_seconds)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())