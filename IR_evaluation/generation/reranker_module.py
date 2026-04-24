# reranker_module.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any
import numpy as np

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

@dataclass
class RerankConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    device: Optional[str] = None  # "mps" / "cuda" / "cpu" / None

    # safety cap
    max_doc_chars: int = 20000

    # windowing (used only if doc is longer than window_chars)
    window_chars: int = 1200
    window_overlap: int = 200
    min_window_chars: int = 50

    # aggregation for windowed docs
    agg: str = "max"               # recommend "max" for stability
    topk_windows: int = 1          # ignored for "max" (keep for future)
    softmax_temp: float = 1.0      # ignored for "max"

    normalize_whitespace: bool = True
    return_spans: bool = True      # best window span (only for long docs)

    add_scores_to_metadata: bool = True


def _auto_device() -> str:
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"
    except Exception:
        return "cpu"


def _get_model(model_name: str, device: Optional[str]):
    from sentence_transformers import CrossEncoder
    if device is None:
        device = _auto_device()
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = CrossEncoder(model_name, device=device)
    return _MODEL_CACHE[key]


def _clean_text(t: str, normalize_whitespace: bool) -> str:
    if not t:
        return ""
    if normalize_whitespace:
        return " ".join(t.split())
    return t


def _make_windows(text: str, cfg: RerankConfig) -> List[Tuple[int, int, str]]:
    """
    Char windows with overlap. Called only when len(text) > cfg.window_chars.
    """
    n = len(text)
    step = max(1, cfg.window_chars - cfg.window_overlap)
    windows: List[Tuple[int, int, str]] = []
    for start in range(0, n, step):
        end = min(n, start + cfg.window_chars)
        if end - start < cfg.min_window_chars:
            continue
        windows.append((start, end, text[start:end]))
        if end >= n:
            break
    return windows


def _aggregate(scores: np.ndarray, cfg: RerankConfig) -> float:
    if scores.size == 0:
        return float("-inf")
    if cfg.agg == "max":
        return float(scores.max())

    k = min(cfg.topk_windows, scores.size)
    top = np.sort(scores)[-k:]
    if cfg.agg == "mean_topk":
        return float(top.mean())
    if cfg.agg == "softmax_topk":
        x = top / max(cfg.softmax_temp, 1e-6)
        x = x - x.max()
        w = np.exp(x)
        w = w / (w.sum() + 1e-12)
        return float((w * top).sum())

    return float(scores.max())


def rerank_with_spans(
    query: str,
    docs: Sequence[str],
    config: Optional[RerankConfig] = None,
) -> Tuple[List[float], Optional[List[Optional[Tuple[int, int]]]]]:
    """
    Option B: windowing only if doc is longer than window_chars.
    Returns doc_scores and optional best span (for long docs only).
    """
    cfg = config or RerankConfig()
    model = _get_model(cfg.model_name, cfg.device)

    q = _clean_text(query, cfg.normalize_whitespace)

    # We will score in two groups:
    # - short docs: one pair (q, doc)
    # - long docs: multiple pairs (q, window)
    short_pairs: List[Tuple[str, str]] = []
    short_map: List[int] = []

    win_pairs: List[Tuple[str, str]] = []
    win_map: List[Tuple[int, int, int]] = []  # (doc_i, start, end)

    cleaned_docs: List[str] = []
    for i, d in enumerate(docs):
        t = _clean_text((d or "")[: cfg.max_doc_chars], cfg.normalize_whitespace)
        cleaned_docs.append(t)

        if len(t) <= cfg.window_chars:
            if len(t) >= cfg.min_window_chars:
                short_pairs.append((q, t))
                short_map.append(i)
        else:
            for (s, e, wtxt) in _make_windows(t, cfg):
                win_pairs.append((q, wtxt))
                win_map.append((i, s, e))

    doc_scores = [float("-inf")] * len(docs)
    spans = [None] * len(docs) if cfg.return_spans else None

    # Score short docs
    if short_pairs:
        short_scores = model.predict(short_pairs, batch_size=cfg.batch_size, show_progress_bar=False)
        for idx, sc in zip(short_map, short_scores):
            doc_scores[idx] = float(sc)
        # spans for short docs: we can set (0, len) or None; keep None for clarity

    # Score windows for long docs
    if win_pairs:
        try:
            win_scores = model.predict(win_pairs, batch_size=cfg.batch_size, show_progress_bar=False)
            win_scores = np.asarray(win_scores, dtype=float)
        except RuntimeError as e:
            msg = str(e).lower()
            current_device = cfg.device or _auto_device()

            # fallback CPU pour CUDA ET MPS
            if ("out of memory" in msg or "oom" in msg) and current_device in {"cuda", "mps"}:
                model_cpu = _get_model(cfg.model_name, "cpu")
                win_scores = model_cpu.predict(
                    win_pairs,
                    batch_size=max(4, cfg.batch_size // 4),
                    show_progress_bar=False
                )
                win_scores = np.asarray(win_scores, dtype=float)
            else:
                raise

        per_doc: Dict[int, List[float]] = {}
        per_doc_best: Dict[int, Tuple[float, int, int]] = {}  # (best_score, s, e)

        for (doc_i, s, e), sc in zip(win_map, win_scores):
            scf = float(sc)
            per_doc.setdefault(doc_i, []).append(scf)
            if cfg.return_spans:
                prev = per_doc_best.get(doc_i)
                if prev is None or scf > prev[0]:
                    per_doc_best[doc_i] = (scf, s, e)

        for doc_i, arr_list in per_doc.items():
            agg_score = _aggregate(np.asarray(arr_list, dtype=float), cfg)
            doc_scores[doc_i] = float(agg_score)
            if cfg.return_spans and doc_i in per_doc_best and spans is not None:
                _, s, e = per_doc_best[doc_i]
                spans[doc_i] = (s, e)

    return doc_scores, spans


def rerank(query: str, docs: Sequence[str], config: Optional[RerankConfig] = None) -> List[float]:
    scores, _ = rerank_with_spans(query, docs, config=config)
    return scores


def top_k_indices(scores: Sequence[float], k: int) -> List[int]:
    if k <= 0:
        return []
    arr = np.asarray(scores, dtype=float)
    order = np.argsort(-arr)
    return order[:k].tolist()