from __future__ import annotations

"""
Build exact top-k ground truth between document embeddings X.npy and question embeddings Q.npy.

Inputs:
    X.npy
        normalized document embeddings
    Q.npy
        normalized question embeddings

Outputs:
    gt_idx.npy
        shape (Nq, k), exact top-k document indices for each question
    gt_scores.npy
        shape (Nq, k), cosine similarities aligned with gt_idx
    ground_truth_meta.json
        small metadata file

Environment variables:
    ENV_FILE
    OUT_DIR
    GT_TOP_K
    GT_QUERY_BLOCK_SIZE
    GT_DB_BLOCK_SIZE
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


def exact_topk_cosine_streaming(
    x_path: Path,
    q_path: Path,
    gt_idx_path: Path,
    gt_scores_path: Path,
    k: int,
    query_block_size: int,
    db_block_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    X = np.load(x_path, mmap_mode="r")
    Q = np.load(q_path, mmap_mode="r")

    if X.ndim != 2 or Q.ndim != 2:
        raise ValueError(f"X et Q doivent être 2D. Reçu X={X.shape}, Q={Q.shape}")

    n_docs, dim_x = X.shape
    n_questions, dim_q = Q.shape

    if dim_x != dim_q:
        raise ValueError(f"Dimension mismatch: X={X.shape}, Q={Q.shape}")

    if k <= 0:
        raise ValueError("GT_TOP_K doit être >= 1")
    if k > n_docs:
        raise ValueError(f"GT_TOP_K={k} > nombre de documents={n_docs}")

    gt_idx = np.lib.format.open_memmap(
        gt_idx_path,
        mode="w+",
        dtype=np.int64,
        shape=(n_questions, k),
    )
    gt_scores = np.lib.format.open_memmap(
        gt_scores_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_questions, k),
    )

    query_ranges = range(0, n_questions, query_block_size)

    for q_start in tqdm(query_ranges, desc="Ground truth", unit="qblock"):
        q_end = min(q_start + query_block_size, n_questions)
        Qb = np.asarray(Q[q_start:q_end], dtype=np.float32)
        bq = Qb.shape[0]

        best_scores = np.full((bq, k), -np.inf, dtype=np.float32)
        best_idx = np.full((bq, k), -1, dtype=np.int64)

        for x_start in range(0, n_docs, db_block_size):
            x_end = min(x_start + db_block_size, n_docs)
            Xb = np.asarray(X[x_start:x_end], dtype=np.float32)

            sims = Qb @ Xb.T  # (bq, xb)

            db_indices = np.arange(x_start, x_end, dtype=np.int64)
            db_indices = np.broadcast_to(db_indices, sims.shape)

            merged_scores = np.concatenate([best_scores, sims], axis=1)
            merged_idx = np.concatenate([best_idx, db_indices], axis=1)

            keep = np.argpartition(-merged_scores, kth=k - 1, axis=1)[:, :k]
            best_scores = np.take_along_axis(merged_scores, keep, axis=1)
            best_idx = np.take_along_axis(merged_idx, keep, axis=1)

            order = np.argsort(-best_scores, axis=1)
            best_scores = np.take_along_axis(best_scores, order, axis=1)
            best_idx = np.take_along_axis(best_idx, order, axis=1)

        gt_idx[q_start:q_end] = best_idx
        gt_scores[q_start:q_end] = best_scores

    gt_idx.flush()
    gt_scores.flush()

    return X.shape, Q.shape


def main() -> None:
    project_root = guess_project_root()
    env_path = project_root / ".env"
    load_env(os.getenv("ENV_FILE", str(env_path)))

    out_dir = Path(os.getenv("OUT_DIR", str(project_root / "data" / "processed")))
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path = out_dir / "X.npy"
    q_path = out_dir / "Q.npy"
    gt_idx_path = out_dir / "gt_idx.npy"
    gt_scores_path = out_dir / "gt_scores.npy"
    meta_path = out_dir / "ground_truth_meta.json"

    k = env_int("GT_TOP_K", 10)
    query_block_size = env_int("GT_QUERY_BLOCK_SIZE", 128)
    db_block_size = env_int("GT_DB_BLOCK_SIZE", 50000)

    print(f".env utilisé: {env_path}")
    print(f"out_dir: {out_dir}")
    print(f"X.npy: {x_path}")
    print(f"Q.npy: {q_path}")
    print(f"GT_TOP_K: {k}")
    print(f"GT_QUERY_BLOCK_SIZE: {query_block_size}")
    print(f"GT_DB_BLOCK_SIZE: {db_block_size}")

    if not x_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {x_path}")
    if not q_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {q_path}")

    x_shape, q_shape = exact_topk_cosine_streaming(
        x_path=x_path,
        q_path=q_path,
        gt_idx_path=gt_idx_path,
        gt_scores_path=gt_scores_path,
        k=k,
        query_block_size=query_block_size,
        db_block_size=db_block_size,
    )

    meta = {
        "x_path": str(x_path),
        "q_path": str(q_path),
        "gt_idx_path": str(gt_idx_path),
        "gt_scores_path": str(gt_scores_path),
        "x_shape": list(x_shape),
        "q_shape": list(q_shape),
        "k": k,
        "query_block_size": query_block_size,
        "db_block_size": db_block_size,
        "similarity": "cosine (assumes X and Q are already L2-normalized)",
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Saved:")
    print(" -", gt_idx_path)
    print(" -", gt_scores_path)
    print(" -", meta_path)


if __name__ == "__main__":
    main()