from __future__ import annotations

"""
Export documents, metadatas, ids and embeddings from a ChromaDB collection to a JSON file.

Output:
    {
        "ids": [...],
        "documents": [...],
        "metadatas": [...],
        "embeddings": [[...], [...], ...],
        "meta": {}
    }
"""

import json
import os
from pathlib import Path

import chromadb
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def load_env(env_path: str | Path | None = None) -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path, override=False)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = (PROJECT_ROOT / ".env").resolve()

load_env(os.getenv("ENV_FILE", str(ENV_PATH)))

DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "api_endpoint" / "data" / "processed" / "export_chroma.json"

CHROMA_BASE_PATH = Path(os.getenv("CHROMA_BASE_PATH", "workspace/chroma"))
if not CHROMA_BASE_PATH.is_absolute():
    CHROMA_BASE_PATH = (PROJECT_ROOT / CHROMA_BASE_PATH).resolve()

DEFAULT_CHROMA_INDEX = os.getenv("DEFAULT_CHROMA_INDEX", "chroma_index")
PERSIST_DIR = (CHROMA_BASE_PATH / DEFAULT_CHROMA_INDEX).resolve()
COLLECTION_NAME = DEFAULT_CHROMA_INDEX

OUTPUT_FILE = Path(os.getenv("CHROMA_OUTPUT_FILE", str(DEFAULT_OUTPUT_FILE)))
if not OUTPUT_FILE.is_absolute():
    OUTPUT_FILE = (PROJECT_ROOT / OUTPUT_FILE).resolve()

BATCH_SIZE = env_int("CHROMA_BATCH_SIZE", 1000)

print(f".env utilisé: {ENV_PATH}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"CHROMA_BASE_PATH: {CHROMA_BASE_PATH}")
print(f"DEFAULT_CHROMA_INDEX: {DEFAULT_CHROMA_INDEX}")
print(f"PERSIST_DIR: {PERSIST_DIR}")
print(f"COLLECTION_NAME: {COLLECTION_NAME}")
print(f"OUTPUT_FILE: {OUTPUT_FILE}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"PERSIST_DIR exists: {PERSIST_DIR.exists()}")


def export_collection(
    persist_dir: Path,
    collection_name: str,
    output_file: Path,
    batch_size: int = 1000,
) -> None:
    if not persist_dir.exists():
        raise FileNotFoundError(f"Chroma persist dir introuvable: {persist_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_output_file = output_file.with_suffix(output_file.suffix + ".tmp")

    client = chromadb.PersistentClient(path=str(persist_dir))
    print("Collections:", client.list_collections())

    col = client.get_collection(collection_name)
    total = col.count()

    try:
        with temp_output_file.open("w", encoding="utf-8") as f:
            # ids
            f.write('{"ids":[')
            first = True
            with tqdm(total=total, desc="Export ids", unit="id") as pbar:
                for offset in range(0, total, batch_size):
                    res = col.get(
                        limit=batch_size,
                        offset=offset,
                        include=[],
                    )
                    ids = res["ids"]
                    for _id in ids:
                        if not first:
                            f.write(",")
                        f.write(json.dumps(_id, ensure_ascii=False))
                        first = False
                    pbar.update(len(ids))

            # documents
            f.write('],"documents":[')
            first = True
            with tqdm(total=total, desc="Export documents", unit="doc") as pbar:
                for offset in range(0, total, batch_size):
                    res = col.get(
                        limit=batch_size,
                        offset=offset,
                        include=["documents"],
                    )

                    ids = res["ids"]
                    docs = res.get("documents")
                    if docs is None:
                        docs = [None] * len(ids)

                    for doc in docs:
                        if not first:
                            f.write(",")
                        f.write(json.dumps(doc, ensure_ascii=False))
                        first = False

                    pbar.update(len(ids))

            # metadatas
            f.write('],"metadatas":[')
            first = True
            with tqdm(total=total, desc="Export metadatas", unit="meta") as pbar:
                for offset in range(0, total, batch_size):
                    res = col.get(
                        limit=batch_size,
                        offset=offset,
                        include=["metadatas"],
                    )

                    ids = res["ids"]
                    metas = res.get("metadatas")
                    if metas is None:
                        metas = [None] * len(ids)

                    for meta in metas:
                        if not first:
                            f.write(",")
                        f.write(json.dumps(meta, ensure_ascii=False))
                        first = False

                    pbar.update(len(ids))

            # embeddings
            f.write('],"embeddings":[')
            first = True
            with tqdm(total=total, desc="Export embeddings", unit="vec") as pbar:
                for offset in range(0, total, batch_size):
                    res = col.get(
                        limit=batch_size,
                        offset=offset,
                        include=["embeddings"],
                    )

                    ids = res["ids"]
                    embs = res.get("embeddings")
                    if embs is None:
                        embs = [None] * len(ids)

                    for emb in embs:
                        if emb is not None and hasattr(emb, "tolist"):
                            emb = emb.tolist()

                        if not first:
                            f.write(",")

                        f.write(json.dumps(emb))
                        first = False

                    pbar.update(len(ids))

            # meta
            f.write('],"meta":{}}')

        temp_output_file.replace(output_file)
        print(f"Export terminé -> {output_file}")

    except Exception:
        if temp_output_file.exists():
            temp_output_file.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    export_collection(
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        output_file=OUTPUT_FILE,
        batch_size=BATCH_SIZE,
    )