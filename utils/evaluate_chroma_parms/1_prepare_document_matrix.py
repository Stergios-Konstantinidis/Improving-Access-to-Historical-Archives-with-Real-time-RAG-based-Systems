from __future__ import annotations

"""
Prepare a binary document matrix from a large Chroma export JSON.

Input:
    export_chroma.json
        {
            "ids": [...],
            "documents": [...],
            "metadatas": [...],
            "embeddings": [[...], [...], ...],
            "meta": {}
        }

Outputs:
    X.npy
        L2-normalized document embeddings, shape (N, d)

    documents.jsonl
        One raw document string per line, aligned with X.npy row index

    documents_with_metadata.jsonl
        One JSON object per line, aligned with X.npy row index
        line i <-> X[i]
        {"id": ..., "text": ..., "metadata": {...}}

    document_matrix_meta.json
        Small metadata file
"""

import json
import os
from pathlib import Path
from itertools import zip_longest

import ijson
import numpy as np
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def guess_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / ".env").exists():
            return candidate
    return here.parent


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


def count_items(json_path: Path, prefix: str) -> int:
    count = 0
    with json_path.open("rb") as f:
        for _ in ijson.items(f, prefix):
            count += 1
    return count


def detect_embedding_dim(json_path: Path) -> int:
    with json_path.open("rb") as f:
        for emb in ijson.items(f, "embeddings.item"):
            if emb is None:
                raise ValueError("La première embedding est None.")
            return len(emb)
    raise ValueError("Aucune embedding trouvée dans le JSON.")


def normalize_rows_inplace(x: np.ndarray) -> None:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x /= norms

def write_documents_jsonl(export_json: Path, documents_jsonl: Path, n_docs: int) -> None:
    with (
        export_json.open("rb") as f_ids,
        export_json.open("rb") as f_docs,
        export_json.open("rb") as f_metas,
        documents_jsonl.open("w", encoding="utf-8") as dst,
    ):
        ids_iter = ijson.items(f_ids, "ids.item")
        docs_iter = ijson.items(f_docs, "documents.item")
        metas_iter = ijson.items(f_metas, "metadatas.item")

        for _id, doc, meta in tqdm(
            zip_longest(ids_iter, docs_iter, metas_iter, fillvalue=None),
            total=n_docs,
            desc="Writing documents.jsonl",
            unit="doc",
        ):
            row = {
                "id": _id,
                "text": doc if doc is not None else "",
                "metadata": meta if isinstance(meta, dict) else {},
            }
            dst.write(json.dumps(row, ensure_ascii=False))
            dst.write("\n")


def write_embeddings_npy(
    export_json: Path,
    x_path: Path,
    n_embeddings: int,
    dim: int,
    block_size: int,
    normalize: bool,
) -> None:
    x_memmap = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_embeddings, dim),
    )

    block_rows: list[list[float]] = []
    write_idx = 0

    def flush_block(rows: list[list[float]], start_idx: int) -> int:
        if not rows:
            return start_idx

        block = np.asarray(rows, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != dim:
            raise ValueError(f"Bloc embedding invalide: shape={block.shape}, dim attendue={dim}")

        if normalize:
            normalize_rows_inplace(block)

        end_idx = start_idx + block.shape[0]
        x_memmap[start_idx:end_idx] = block
        return end_idx

    with export_json.open("rb") as src:
        for emb in tqdm(
            ijson.items(src, "embeddings.item"),
            total=n_embeddings,
            desc="Writing X.npy",
            unit="vec",
        ):
            if emb is None:
                raise ValueError(f"Embedding None détectée à l'index {write_idx + len(block_rows)}")
            if len(emb) != dim:
                raise ValueError(
                    f"Dimension incohérente à l'index {write_idx + len(block_rows)}: "
                    f"attendu {dim}, reçu {len(emb)}"
                )

            block_rows.append(emb)

            if len(block_rows) >= block_size:
                write_idx = flush_block(block_rows, write_idx)
                block_rows = []

    if block_rows:
        write_idx = flush_block(block_rows, write_idx)

    if write_idx != n_embeddings:
        raise ValueError(f"Nombre d'embeddings écrits incorrect: {write_idx} au lieu de {n_embeddings}")

    x_memmap.flush()


def main() -> None:
    project_root = guess_project_root()
    env_path = project_root / ".env"
    load_env(os.getenv("ENV_FILE", str(env_path)))

    export_json = Path(
        os.getenv("CHROMA_EXPORT_JSON", str(project_root / "api_endpoint" / "data" / "processed" / "export_chroma.json"))
    )
    if not export_json.is_absolute():
        export_json = (project_root / export_json).resolve()

    out_dir = Path(os.getenv("OUT_DIR", str(project_root / "data" / "processed")))
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path = out_dir / "X.npy"
    documents_jsonl = out_dir / "documents.jsonl"
    meta_path = out_dir / "document_matrix_meta.json"

    block_size = env_int("DOC_WRITE_BLOCK_SIZE", 1024)
    normalize_docs = env_bool("NORMALIZE_DOCUMENT_EMBEDDINGS", True)

    print(f".env utilisé: {env_path}")
    print(f"export_json: {export_json}")
    print(f"out_dir: {out_dir}")
    print(f"DOC_WRITE_BLOCK_SIZE: {block_size}")
    print(f"NORMALIZE_DOCUMENT_EMBEDDINGS: {normalize_docs}")

    if not export_json.exists():
        raise FileNotFoundError(f"Fichier introuvable: {export_json}")

    print("Counting documents...")
    n_docs = count_items(export_json, "documents.item")

    print("Counting metadatas...")
    n_metas = count_items(export_json, "metadatas.item")

    print("Counting ids...")
    n_ids = count_items(export_json, "ids.item")

    print("Counting embeddings...")
    n_embeddings = count_items(export_json, "embeddings.item")

    print("Detecting embedding dimension...")
    dim = detect_embedding_dim(export_json)

    print(f"Nb documents: {n_docs}")
    print(f"Nb metadatas: {n_metas}")
    print(f"Nb ids: {n_ids}")
    print(f"Nb embeddings: {n_embeddings}")
    print(f"Embedding dim: {dim}")

    if n_docs != n_embeddings:
        raise ValueError(f"Mismatch documents / embeddings: {n_docs} vs {n_embeddings}")
    if n_metas != n_docs:
        raise ValueError(f"Mismatch metadatas / documents: {n_metas} vs {n_docs}")
    if n_ids != n_docs:
        raise ValueError(f"Mismatch ids / documents: {n_ids} vs {n_docs}")

    write_documents_jsonl(export_json, documents_jsonl, n_docs)
    write_embeddings_npy(export_json, x_path, n_embeddings, dim, block_size, normalize_docs)

    meta = {
        "source_export_json": str(export_json),
        "x_path": str(x_path),
        "documents_jsonl": str(documents_jsonl),
        "num_documents": n_docs,
        "embedding_dim": dim,
        "dtype": "float32",
        "normalized": normalize_docs,
        "doc_index_alignment": (
            "line i in documents.jsonl and documents_with_metadata.jsonl <-> row i in X.npy"
        ),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Saved:")
    print(" -", x_path)
    print(" -", documents_jsonl)
    print(" -", meta_path)


if __name__ == "__main__":
    main()