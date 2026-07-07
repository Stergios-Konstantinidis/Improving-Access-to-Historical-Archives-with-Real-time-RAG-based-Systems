#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import bm25s
except Exception as e:
    raise RuntimeError(
        "Le package 'bm25s' n'est pas installé. Installe-le avec: pip install -U bm25s"
    ) from e

try:
    from bm25s.tokenization import Tokenizer
except Exception:
    Tokenizer = None

try:
    import Stemmer
except Exception:
    Stemmer = None

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
        return dict(value)
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


def load_generations(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl_raw(path)
    out: List[Dict[str, Any]] = []

    for i, row in enumerate(progress(rows, desc="Chargement generations", unit="q")):
        if not isinstance(row, dict):
            continue

        question = normalize_text(row.get("question", ""))
        if not question:
            continue

        new_row = dict(row)
        new_row["question_id"] = str(row.get("question_id") or f"q_{i:04d}")
        new_row["question"] = question

        retrieved_chunks = row.get("retrieved_chunks", [])
        if not isinstance(retrieved_chunks, list):
            retrieved_chunks = []
        new_row["retrieved_chunks"] = retrieved_chunks

        out.append(new_row)

    return out


def build_stopwords_candidates(stopwords: Optional[str]) -> List[Optional[str]]:
    if stopwords is None:
        return [None]

    stopwords = str(stopwords).strip()
    if not stopwords:
        return [None]

    candidates: List[Optional[str]] = [stopwords]

    aliases = {
        "fr": "french",
        "french": "fr",
        "en": "english",
        "english": "en",
    }

    alias = aliases.get(stopwords.lower())
    if alias is not None and alias not in candidates:
        candidates.append(alias)

    return candidates


class BM25sRetriever:
    def __init__(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        *,
        stopwords: Optional[str] = "fr",
        stem_language: str = "french",
        use_stemming: bool = True,
    ):
        if len(ids) != len(documents) or len(documents) != len(metadatas):
            raise ValueError("ids, documents et metadatas doivent avoir la même longueur")

        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas

        self.requested_stopwords = stopwords
        self.stopwords: Optional[str] = None
        self.stem_language = stem_language
        self.use_stemming = use_stemming
        self.stemmer = None
        self.tokenizer = None

        if self.use_stemming:
            if Stemmer is None:
                raise RuntimeError(
                    "Le stemming est activé mais PyStemmer n'est pas installé. "
                    "Installe-le avec: pip install -U PyStemmer"
                )
            self.stemmer = Stemmer.Stemmer(self.stem_language)

        print("Tokenization BM25s du corpus...")
        self.tokens = self._tokenize_corpus(self.documents)

        print("Indexation BM25s...")
        self.retriever = bm25s.BM25()
        self.retriever.index(self.tokens)

        print(
            "BM25 configuré avec "
            f"stopwords={self.stopwords!r}, "
            f"stemming={self.use_stemming}, "
            f"stem_language={self.stem_language if self.use_stemming else None}"
        )

    def _tokenize_corpus(self, documents: List[str]):
        stopword_candidates = build_stopwords_candidates(self.requested_stopwords)
        last_exc: Optional[Exception] = None

        for candidate in stopword_candidates:
            try:
                if Tokenizer is not None:
                    tokenizer = Tokenizer(
                        stemmer=self.stemmer,
                        stopwords=candidate,
                    )
                    tokens = tokenizer.tokenize(documents)
                    self.tokenizer = tokenizer
                    self.stopwords = candidate
                    return tokens

                tokens = bm25s.tokenize(
                    documents,
                    stopwords=candidate,
                    stemmer=self.stemmer,
                )
                self.tokenizer = None
                self.stopwords = candidate
                return tokens

            except Exception as exc:
                last_exc = exc

        raise RuntimeError(
            "Impossible de tokeniser le corpus avec les stopwords fournis "
            f"({self.requested_stopwords!r})."
        ) from last_exc

    def _tokenize_query(self, query: str):
        if self.tokenizer is not None:
            return self.tokenizer.tokenize([query])

        return bm25s.tokenize(
            [query],
            stopwords=self.stopwords,
            stemmer=self.stemmer,
        )

    def search(self, query: str, top_k: int) -> Dict[str, Any]:
        if top_k <= 0:
            return {"hits": [], "retrieval_ms": 0.0}

        t0 = now_ms()

        query_tokens = self._tokenize_query(query)

        # Sans corpus=..., bm25s renvoie les indices/doc ids du corpus indexé.
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)

        retrieval_ms = now_ms() - t0

        result_indices = [int(x) for x in results[0]]
        result_scores = [float(x) for x in scores[0]]

        hits: List[Dict[str, Any]] = []

        for rank, (pos, score) in enumerate(zip(result_indices, result_scores)):
            if pos < 0 or pos >= len(self.documents):
                continue

            metadata = dict(self.metadatas[pos])

            retrieval_meta = dict(metadata.get("retrieval", {}))
            retrieval_meta.update({
                "source": "bm25s",
                "rank": rank,
                "score": score,
                "stemming": self.use_stemming,
                "stem_language": self.stem_language if self.use_stemming else None,
                "stopwords": self.stopwords,
            })
            metadata["retrieval"] = retrieval_meta
            metadata["bm25_score"] = score

            hits.append({
                "id": self.ids[pos],
                "text": self.documents[pos],
                "metadata": metadata,
                "distance": score,  # conservé pour compatibilité avec ton format actuel
                "rank": rank,
            })

        return {
            "hits": hits,
            "retrieval_ms": retrieval_ms,
        }


def mark_existing_chunks_as_chroma(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    marked: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue

        new_chunk = dict(chunk)
        new_chunk.setdefault("retrieval_source", "chroma")
        new_chunk.setdefault("merged_position", i)
        marked.append(new_chunk)

    return marked


def mark_bm25_chunks(chunks: List[Dict[str, Any]], start_pos: int) -> List[Dict[str, Any]]:
    marked: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        new_chunk = dict(chunk)
        new_chunk["retrieval_source"] = "bm25s"
        new_chunk["merged_position"] = start_pos + i
        marked.append(new_chunk)

    return marked


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def extract_latency_ms(value: Any) -> float:
    """
    Accepte:
    - un float/int
    - un dict du style {"real_pipeline_ms": ..., ...}
    - None
    """
    if value is None:
        return 0.0

    if is_number(value):
        return float(value)

    if isinstance(value, dict):
        real_pipeline = value.get("real_pipeline_ms")
        if is_number(real_pipeline):
            return float(real_pipeline)

        total = 0.0
        for v in value.values():
            if is_number(v):
                total += float(v)
        return total

    return 0.0


def merge_latency_ms(existing_latency: Any, bm25_ms: float) -> Any:
    """
    Si existing_latency est un dict, on garde ce format.
    Si c'est un nombre, on renvoie un nombre.
    """
    bm25_ms = float(bm25_ms)

    if isinstance(existing_latency, dict):
        out = dict(existing_latency)
        out["bm25s_retrieval_ms"] = bm25_ms
        out["real_pipeline_ms"] = extract_latency_ms(existing_latency) + bm25_ms
        return out

    return extract_latency_ms(existing_latency) + bm25_ms


def append_bm25_to_row(
    row: Dict[str, Any],
    retriever: BM25sRetriever,
    bm25_top_k: int,
    force_empty_answer: bool = False,
) -> Dict[str, Any]:
    out = dict(row)

    existing_chunks_raw = out.get("retrieved_chunks", [])
    if not isinstance(existing_chunks_raw, list):
        existing_chunks_raw = []

    chroma_chunks = mark_existing_chunks_as_chroma(existing_chunks_raw)

    result = retriever.search(out["question"], top_k=bm25_top_k)
    bm25_chunks = mark_bm25_chunks(result["hits"], start_pos=len(chroma_chunks))

    merged_chunks = chroma_chunks + bm25_chunks
    out["retrieved_chunks"] = merged_chunks

    if force_empty_answer:
        out["generated_answer"] = ""

    method_metadata = dict(out.get("method_metadata") or {})
    method_metadata["bm25s_append"] = {
        "provider": "bm25s",
        "appended_count": len(bm25_chunks),
        "top_k": bm25_top_k,
        "deduplicated": False,
        "stemming": retriever.use_stemming,
        "stem_language": retriever.stem_language if retriever.use_stemming else None,
        "stopwords": retriever.stopwords,
    }
    method_metadata["fusion_layout"] = {
        "first_block": {
            "source": "chroma",
            "count": len(chroma_chunks),
        },
        "second_block": {
            "source": "bm25s",
            "count": len(bm25_chunks),
        },
        "total_count": len(merged_chunks),
    }
    out["method_metadata"] = method_metadata

    latency_breakdown = dict(out.get("latency_breakdown_ms") or {})
    latency_breakdown["bm25s_retrieval_ms"] = result["retrieval_ms"]
    out["latency_breakdown_ms"] = latency_breakdown

    previous_latency = out.get("latency_ms")
    previous_processing = extract_latency_ms(out.get("question_processing_time_ms"))

    out["latency_ms"] = merge_latency_ms(previous_latency, result["retrieval_ms"])
    out["question_processing_time_ms"] = previous_processing + result["retrieval_ms"]

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ajoute les top-k BM25s à la suite des retrieved_chunks existants "
            "dans un generations.jsonl, sans déduplication."
        )
    )

    parser.add_argument(
        "--corpus-json",
        required=True,
        help="Chemin vers le corpus (.json ou .jsonl)",
    )
    parser.add_argument(
        "--input-generations-jsonl",
        required=True,
        help="Chemin vers le generations.jsonl existant (déjà rempli côté Chroma)",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Chemin du fichier de sortie",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=100,
        help="Nombre de chunks BM25 à ajouter à la suite",
    )
    parser.add_argument(
        "--force-empty-answer",
        action="store_true",
        help="Vide systématiquement le champ generated_answer dans le fichier de sortie",
    )

    parser.add_argument(
        "--bm25-stopwords",
        default="fr",
        help=(
            "Stopwords à utiliser pour BM25. "
            "Exemples: fr, french, en, english. "
            "Mettre une chaîne vide pour désactiver."
        ),
    )
    parser.add_argument(
        "--bm25-stem-language",
        default="french",
        help="Langue du stemmer PyStemmer. Exemple: french, english, german, ...",
    )
    parser.add_argument(
        "--bm25-no-stemming",
        action="store_true",
        help="Désactive le stemming BM25",
    )

    return parser.parse_args()


def main() -> int:
    global_start_ms = now_ms()

    args = parse_args()

    corpus_path = Path(args.corpus_json)
    input_generations_path = Path(args.input_generations_jsonl)
    output_path = Path(args.output_jsonl)

    requested_stopwords: Optional[str] = args.bm25_stopwords
    if requested_stopwords is not None and requested_stopwords.strip() == "":
        requested_stopwords = None

    print("Chargement du corpus...")
    corpus = load_corpus(corpus_path)

    print("Chargement du fichier generations existant...")
    rows = load_generations(input_generations_path)

    if not corpus["documents"]:
        raise RuntimeError("Le corpus est vide.")
    if not rows:
        raise RuntimeError("Aucune ligne valide trouvée dans le fichier generations.")

    print(f"Corpus chargé : {len(corpus['documents'])} documents")
    print(f"Questions chargées : {len(rows)}")

    retriever = BM25sRetriever(
        ids=corpus["ids"],
        documents=corpus["documents"],
        metadatas=corpus["metadatas"],
        stopwords=requested_stopwords,
        stem_language=args.bm25_stem_language,
        use_stemming=not args.bm25_no_stemming,
    )

    out_rows: List[Dict[str, Any]] = []
    latencies: List[float] = []

    retrieval_batch_start_ms = now_ms()

    for row in progress(rows, desc="Append BM25", unit="q"):
        old_latency = extract_latency_ms(row.get("latency_ms"))

        new_row = append_bm25_to_row(
            row=row,
            retriever=retriever,
            bm25_top_k=args.bm25_top_k,
            force_empty_answer=args.force_empty_answer,
        )

        new_latency = extract_latency_ms(new_row.get("latency_ms"))
        latencies.append(max(0.0, new_latency - old_latency))
        out_rows.append(new_row)

    retrieval_batch_end_ms = now_ms()

    write_jsonl(output_path, out_rows)

    global_end_ms = now_ms()

    total_retrieval_seconds = (retrieval_batch_end_ms - retrieval_batch_start_ms) / 1000.0
    total_program_seconds = (global_end_ms - global_start_ms) / 1000.0

    print(f"\nFichier généré : {output_path}")
    print(f"Questions traitées : {len(out_rows)}")

    if latencies:
        print(f"Latency moyenne BM25 ajoutée : {statistics.mean(latencies):.2f} ms")
        print(f"Latency médiane BM25 ajoutée : {statistics.median(latencies):.2f} ms")

    print(f"Temps total pour exécuter toutes les requêtes de questions : {format_duration(total_retrieval_seconds)}")
    print(f"Temps total du programme : {format_duration(total_program_seconds)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())