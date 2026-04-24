# get_relating_document_bis.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import os
import re
import json
import hashlib
import time

import numpy as np
import chromadb

from .reranker_module import RerankConfig, rerank_with_spans


from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)
DEFAULT_CHROMA_INDEX = os.getenv("DEFAULT_CHROMA_INDEX", "chroma_index")

def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

ENABLE_DEDUP = env_bool("ENABLE_DEDUP", False)
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.95"))
# ============================================================
# Text dedup
# ============================================================
def _normalize_text(text: str) -> str:
    """
    Normalize OCR-ish text for dedup:
    - lowercase
    - collapse whitespace
    - remove hyphen-break artifacts ("- ")
    """
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("- ", "")
    return t.strip()


def _dedup_candidates(
    ids: List[str],
    docs: List[str],
    metas: List[dict],
    dists: List[Optional[float]],
    verbose: bool = True,
) -> Tuple[List[str], List[str], List[dict], List[Optional[float]]]:
    """
    Dedup by normalized text only.
    Keeps the best candidate by smallest distance if duplicates exist.
    """
    before = len(ids)
    seen: Dict[str, int] = {}

    out_ids: List[str] = []
    out_docs: List[str] = []
    out_metas: List[dict] = []
    out_dists: List[Optional[float]] = []

    empty_removed = 0
    duplicate_hits = 0

    for i, doc in enumerate(docs):
        norm = _normalize_text(doc or "")
        if not norm:
            continue

        key = hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()

        if key not in seen:
            seen[key] = len(out_ids)
            out_ids.append(ids[i])
            out_docs.append(docs[i])
            out_metas.append(metas[i] if metas else {})
            out_dists.append(dists[i] if dists else None)
        else:
            out_i = seen[key]
            prev_dist = out_dists[out_i]
            new_dist = dists[i] if dists else None

            if (
                prev_dist is not None
                and new_dist is not None
                and isinstance(prev_dist, (int, float))
                and isinstance(new_dist, (int, float))
                and new_dist < prev_dist
            ):
                out_ids[out_i] = ids[i]
                out_docs[out_i] = docs[i]
                out_metas[out_i] = metas[i] if metas else {}
                out_dists[out_i] = new_dist
    after = len(out_ids)
    removed = before - after

    if verbose:
        print(
            f"[DEDUP_TEXT_CANDIDATE] before={before} "
            f"after={after} "
            f"removed_total={removed} "
            f"empty_removed={empty_removed} "
            f"duplicate_hits={duplicate_hits}"
        )
    return out_ids, out_docs, out_metas, out_dists


def _dedup_by_similarity(
    ids: List[str],
    docs: List[str],
    metas: List[dict],
    dists: List[Optional[float]],
    embeddings: List[Any],
    threshold: float = DEDUP_THRESHOLD,
    verbose: bool = False,
):
    """
    Dedup by embedding cosine similarity > threshold.
    Returns filtered ids/docs/metas/dists/embeddings.
    """
    before = len(ids)

    embs = np.array(embeddings, dtype=float)
    if embs.ndim != 2 or embs.shape[0] == 0:
        return ids, docs, metas, dists, embeddings

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norms + 1e-10)

    kept: List[int] = []
    removed_pairs = []

    for i in range(len(docs)):

        similar_j = None
        sim_score = None

        for j in kept:
            sim = float(np.dot(embs_norm[i], embs_norm[j]))
            if sim > threshold:
                similar_j = j
                sim_score = sim
                break

        if similar_j is None:
            kept.append(i)
        else:
            removed_pairs.append((i, similar_j, sim_score))

    after = len(kept)

    if verbose:
        print(f"[DEDUP_SIM] before={before} after={after} removed={before - after} threshold={threshold}")

        for i, j, sim in removed_pairs:
            print(f"[DEDUP_TEXT_SIMILARITY] ")
            print(f"  - doc[{i}] removed (similarity={sim:.3f})")
            print(f"    removed: '{docs[i][:120]}'")
            print(f"    kept   : '{docs[j][:120]}'")
            print()

    return (
        [ids[i] for i in kept],
        [docs[i] for i in kept],
        [metas[i] for i in kept],
        [dists[i] for i in kept],
        [embeddings[i] for i in kept],
    )

# ============================================================
# run_file cache (JSONL runs with retrieved_chunks)
# ============================================================
_RUN_FILE_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_run_file_index(run_file_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Index a run JSONL file by question_id.
    Cached by file mtime to avoid re-reading.
    """
    run_file_path = Path(run_file_path).expanduser().resolve()
    cache_key = str(run_file_path)
    mtime = run_file_path.stat().st_mtime

    cached = _RUN_FILE_CACHE.get(cache_key)
    if cached is not None and cached.get("mtime") == mtime:
        return cached["by_qid"]

    by_qid: Dict[str, Dict[str, Any]] = {}
    with run_file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("question_id")
            if qid:
                by_qid[str(qid)] = rec

    _RUN_FILE_CACHE[cache_key] = {"mtime": mtime, "by_qid": by_qid}
    return by_qid


def _chunk_text(c: Dict[str, Any]) -> str:
    t = c.get("text")
    if isinstance(t, str) and t.strip():
        return t
    t = c.get("content")
    if isinstance(t, str) and t.strip():
        return t
    return ""


def _from_run_file(
    run_file_path: str | Path,
    question_id: str,
    n_results: int,
) -> Tuple[List[str], List[str], List[dict], List[Optional[float]], List[Any]]:
    idx = _load_run_file_index(run_file_path)
    rec = idx.get(str(question_id))
    if rec is None:
        return [], [], [], [], []

    chunks = rec.get("retrieved_chunks", []) or []
    chunks = chunks[:n_results]

    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []
    dists: List[Optional[float]] = []
    embs: List[Any] = []

    for i, c in enumerate(chunks):
        txt = _chunk_text(c)
        if not txt:
            continue

        cid = c.get("id") or c.get("doc_id") or f"run_{question_id}_{i}"
        md = dict(c.get("metadata") or {})

        # Champs déjà gérés
        if c.get("doc_id") is not None:
            md["doc_id"] = c.get("doc_id")
        md["id"] = str(cid)

        # Champs à transporter depuis le run_file
        for key in [
            "rank",
            "score",
            "chunk_answer",
            "relevance_score",
            "justification",
            "is_relevant",
            "original_rank",
        ]:
            if key in c:
                md[key] = c[key]

        ids.append(str(cid))
        docs.append(txt)
        metas.append(md)
        dists.append(c.get("distance") if "distance" in c else None)

    return ids, docs, metas, dists, embs


# ============================================================
# Main retrieval function
# ============================================================
def get_relating_documents_bis(
    query: str,
    filters: Optional[dict],
    *,
    chroma_base_path: str | Path,
    embedding_fn,
    k_value: int = 10,
    filter_document_text: Optional[str] = None,
    index: str = DEFAULT_CHROMA_INDEX,
    use_reranker: bool = False,
    rerank_config: Optional[RerankConfig] = None,
    n_candidates: Optional[int] = None,      # pool initial if reranker enabled
    chroma_persist_mode: str = "per_index",  # "per_index" or "base"
    return_timing=False,

    # Retrieval source
    source: str = "chroma",                  # "chroma" or "run_file"
    run_file_path: Optional[str | Path] = None,     # required if source="run_file"
    question_id: Optional[str] = None,       # required if source="run_file"
    verbose: bool = True,
    generate_embeddings: bool = True,
) -> Dict[str, List[List[Any]]]:
    """
    Returns a dict compatible with Chroma query format:
      {
        "ids": [[...]],
        "documents": [[...]],
        "metadatas": [[...]],
        "distances": [[...]]
      }

    source:
      - "chroma": query ChromaDB (current behavior)
      - "run_file": read retrieved_chunks from a JSONL export indexed by question_id
    """
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    def _finalize(payload):
        timings["retrieval_total"] = time.perf_counter() - t_total
        if return_timing:
            payload["timings"] = timings
            payload["stats"] = {
                "source": source,
                "use_reranker": use_reranker,
                "k_value": k_value,
            }
        return payload

    # 1) Retrieval params
    if use_reranker:
        pool = n_candidates if n_candidates is not None else max(50, k_value * 8)
        n_results = pool
    else:
        n_results = k_value

    ids = []
    docs = []
    metas = []
    dists = []
    embs = []

    # ----------------------------
    # SOURCE SWITCH
    # ----------------------------
    if source == "run_file":
        if not run_file_path:
            raise ValueError('source="run_file" requires run_file_path="...jsonl"')
        if not question_id:
            raise ValueError('source="run_file" requires question_id="q_0000"')

        t = time.perf_counter()
        ids, docs, metas, dists, embs = _from_run_file(
            run_file_path=run_file_path,
            question_id=str(question_id),
            n_results=n_results,
        )
        timings["run_file_load"] = time.perf_counter() - t



        if source == "run_file" and docs:
            t = time.perf_counter()
            if generate_embeddings:
                embs = embedding_fn(docs)
            timings["run_file_doc_embeddings"] = time.perf_counter() - t

        if verbose:
            print("\n[DEBUG] embeddings shape:", len(embs), "x", len(embs[0]) if embs else 0)

    elif source == "chroma":
        # Resolve Chroma persist path + collection name
        t = time.perf_counter()
        if chroma_persist_mode == "base":
            persist_path = chroma_base_path
        else:
            persist_path = os.path.join(chroma_base_path, index)

        collection_name = index

        # Create client and get collection
        chroma_client = chromadb.PersistentClient(path=persist_path)
        chroma_index = chroma_client.get_collection(name=collection_name)

        include = ["documents", "metadatas", "distances", "embeddings"]
        timings["chroma_prepare"] = time.perf_counter() - t

        print("\n[EMBEDDING CHECK BIS]")
        print("index:", index)
        print("chroma_base_path:", chroma_base_path)
        print("embedding_fn_type:", type(embedding_fn).__name__)

        if hasattr(embedding_fn, "model_name"):
            print("embedding_model_name:", embedding_fn.model_name)

        if hasattr(embedding_fn, "api_base"):
            print("embedding_api_base:", embedding_fn.api_base)

        t = time.perf_counter()
        query_embedding = embedding_fn([query])
        timings["query_embedding"] = time.perf_counter() - t

        t = time.perf_counter()
        query_kwargs: Dict[str, Any] = dict(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=include,
        )
        if filters:
            query_kwargs["where"] = filters

        results = chroma_index.query(**query_kwargs)
        timings["chroma_query"] = time.perf_counter() - t

        # Normalize outputs
        t = time.perf_counter()
        ids = results.get("ids", [[]])[0] or []
        docs = results.get("documents", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or [{} for _ in ids]
        dists = results.get("distances", [[]])[0] or [None] * len(ids)
        embs_raw = results.get("embeddings", None)
        embs = embs_raw[0] if embs_raw is not None and len(embs_raw) > 0 else []


        # Add id in metadata (stable)
        for i in range(len(ids)):
            if metas[i] is None:
                metas[i] = {}
            metas[i]["id"] = ids[i]
        timings["normalize_results"] = time.perf_counter() - t

    else:
        raise ValueError('source must be "chroma" or "run_file"')

    # Early return if empty
    if not ids:
        return _finalize({"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})

    # Optional text filter before rerank
    if filter_document_text:
        t = time.perf_counter()
        needle = filter_document_text
        keep_idx = [i for i, d in enumerate(docs) if d and needle in d]
        ids = [ids[i] for i in keep_idx]
        docs = [docs[i] for i in keep_idx]
        metas = [metas[i] for i in keep_idx]
        dists = [dists[i] for i in keep_idx]
        if embs:
            embs = [embs[i] for i in keep_idx]
        timings["filter_document_text"] = time.perf_counter() - t

    # Early return if empty
    if not ids:
        return _finalize({"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})

    # 5) No rerank: sort by distance (optional)
    if not use_reranker:
        t = time.perf_counter()
        order = list(range(len(ids)))
        # smaller is usually better
        if any(x is not None for x in dists):
            order = sorted(order, key=lambda i: dists[i] if dists[i] is not None else 1e9)
        order = order[:k_value]
        timings["sort_without_rerank"] = time.perf_counter() - t

        return _finalize({
            "ids": [[ids[i] for i in order]],
            "documents": [[docs[i] for i in order]],
            "metadatas": [[metas[i] for i in order]],
            "distances": [[dists[i] for i in order]],
        })

    # # 6) Dedup before rerank (important for OCR)
    t = time.perf_counter()
    before_dedup = len(ids)

    if ENABLE_DEDUP:
        if embs is not None and len(embs) > 0 and len(embs) == len(docs):
            ids, docs, metas, dists, embs = _dedup_by_similarity(ids, docs, metas, dists, embs, threshold=DEDUP_THRESHOLD, verbose=verbose)
        else:
            ids, docs, metas, dists = _dedup_candidates(ids, docs, metas, dists, verbose=verbose)

    if not ids:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    timings["dedup_method"] = 0.0
    timings["dedup"] = time.perf_counter() - t
    after_dedup = len(ids)

    # 7) Rerank (Option B: windowing only for long docs)
    if not ids:
        return _finalize({"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})

    cfg = rerank_config or RerankConfig()

    rerank_inputs = [
        {
            "id": ids[i],
            "text": docs[i] or "",
            "metadata": metas[i] if i < len(metas) and isinstance(metas[i], dict) else {},
        }
        for i in range(len(docs))
    ]

    t = time.perf_counter()
    scores, spans = rerank_with_spans(query, rerank_inputs, config=cfg)
    timings["rerank"] = time.perf_counter() - t

    # 8) Build final order
    # If everything is -inf, fallback to distance
    t = time.perf_counter()
    if not any(np.isfinite(s) for s in scores):
        order = list(range(len(ids)))
        if any(x is not None for x in dists):
            order = sorted(order, key=lambda i: dists[i] if dists[i] is not None else 1e9)
        order = order[:k_value]
    else:
        # Hybrid stable sorting: score desc, then distance asc
        order = sorted(
            range(len(ids)),
            key=lambda i: (
                -scores[i],
                dists[i] if dists[i] is not None else 1e9,
            ),
        )[:k_value]
    timings["build_final_order"] = time.perf_counter() - t

    # 9) Construct output
    t = time.perf_counter()
    new_ids: List[str] = []
    new_docs: List[str] = []
    new_metas: List[dict] = []
    new_dists: List[Optional[float]] = []

    for i in order:
        new_ids.append(ids[i])
        new_docs.append(docs[i])
        new_dists.append(dists[i])

        md = dict(metas[i] or {})
        md["id"] = ids[i]

        if getattr(cfg, "add_scores_to_metadata", True):
            md["cross_score"] = float(scores[i])

        if spans is not None and i < len(spans) and spans[i] is not None:
            md["best_span"] = {"start_char": int(spans[i][0]), "end_char": int(spans[i][1])}

        new_metas.append(md)

    timings["construct_output"] = time.perf_counter() - t


    payload = {
        "ids": [new_ids],
        "documents": [new_docs],
        "metadatas": [new_metas],
        "distances": [new_dists],
    }

    payload = _finalize(payload)

    if return_timing:
        payload["stats"].update({
            "n_results_requested": n_results,
            "n_before_dedup": before_dedup,
            "n_after_dedup": after_dedup,
            "n_final": len(new_ids),
        })

    return payload