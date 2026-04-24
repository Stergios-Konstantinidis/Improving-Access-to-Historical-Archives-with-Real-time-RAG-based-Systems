# reranker_module.py
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

DocumentInput = Union[str, Dict[str, Any]]


@dataclass
class RerankConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    device: Optional[str] = None  # "mps" / "cuda" / "cpu" / None
    backend: str = "torch"

    # safety cap
    max_doc_chars: int = 20000

    # windowing
    window_chars: int = 1200
    window_overlap: int = 200
    min_window_chars: int = 50

    # aggregation
    # "max", "mean_topk", "softmax_topk", "logsumexp_topk"
    agg: str = "softmax_topk"
    topk_windows: int = 3
    softmax_temp: float = 0.25

    # text cleanup
    normalize_whitespace: bool = False

    # outputs
    return_spans: bool = True
    add_scores_to_metadata: bool = True

    # metadata
    use_metadata: bool = False
    metadata_max_chars: int = 500

    # fixed whitelist
    metadata_fields: Tuple[str, ...] = (
        "date",
        "year",
        "topic",
        "organization",
    )

    metadata_aliases: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "date": ("date", "Date"),
        "year": ("year", "Year"),
        "topic": ("topic", "Topic", "topics", "Topics"),
        "organization": ("organization", "Organization", "org", "Org"),
    })

    metadata_labels: Dict[str, str] = field(default_factory=lambda: {
        "date": "DATE",
        "year": "YEAR",
        "topic": "TOPIC",
        "organization": "ORG",
    })


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

def _clean_text(t: str, normalize_whitespace: bool) -> str:
    if not t:
        return ""
    if normalize_whitespace:
        return " ".join(t.split()).strip()
    return t.strip()

def _get_model(model_name: str, device: Optional[str], backend: str = "torch"):
    from sentence_transformers import CrossEncoder

    if backend == "torch":
        if device is None:
            device = _auto_device()

        key = (model_name, backend, device)
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = CrossEncoder(
                model_name,
                device=device,
                backend=backend,
            )
        return _MODEL_CACHE[key]

    # openvino / onnx
    key = (model_name, backend, "na")
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = CrossEncoder(
            model_name,
            backend=backend,
            model_kwargs={"device": "GPU"},
        )

        try:
            backend_model = _MODEL_CACHE[key].model
            print(f"[RERANK] internal model type={type(backend_model)}")

            if hasattr(backend_model, "request"):
                print(f"[RERANK] infer request type={type(backend_model.request)}")
                print("[RERANK] execution devices:",
                      backend_model.request.get_property("EXECUTION_DEVICES"))
        except Exception as e:
            print(f"[RERANK] could not inspect backend model: {e}")

    return _MODEL_CACHE[key]


def _make_windows(text: str, cfg: RerankConfig) -> List[Tuple[int, int, str]]:
    """
    Character windows with overlap. Called only when len(text) > cfg.window_chars.
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


def _is_empty_metadata_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, tuple, set, dict)) and len(v) == 0:
        return True
    return False


def _try_parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    s = value.strip()
    if not s:
        return s

    if (
        (s.startswith("[") and s.endswith("]"))
        or (s.startswith("{") and s.endswith("}"))
        or (s.startswith("(") and s.endswith(")"))
    ):
        try:
            return ast.literal_eval(s)
        except Exception:
            return value

    return value


def _normalize_metadata_value(value: Any) -> str:
    value = _try_parse_literal(value)

    if _is_empty_metadata_value(value):
        return ""

    if isinstance(value, dict):
        parts: List[str] = []
        for k, v in value.items():
            nv = _normalize_metadata_value(v)
            if nv:
                parts.append(f"{k}: {nv}")
        return "; ".join(parts)

    if isinstance(value, (list, tuple, set)):
        parts: List[str] = []
        for item in value:
            ni = _normalize_metadata_value(item)
            if ni:
                parts.append(ni)
        return ", ".join(parts)

    return str(value).strip()


def _coerce_metadata_dict(metadata: Any) -> Dict[str, Any]:
    """
    Accepts:
      - dict
      - nested lists containing a dict
      - anything else -> {}
    """
    if metadata is None:
        return {}

    if isinstance(metadata, dict):
        return metadata

    if isinstance(metadata, list):
        stack = list(metadata)
        while stack:
            item = stack.pop(0)
            if isinstance(item, dict):
                return item
            if isinstance(item, list):
                stack = list(item) + stack

    return {}


def _get_metadata_value(
    metadata: Dict[str, Any],
    canonical_key: str,
    cfg: RerankConfig,
) -> Optional[Any]:
    if not metadata:
        return None

    aliases = cfg.metadata_aliases.get(canonical_key, (canonical_key,))
    lowered = {str(k).lower(): v for k, v in metadata.items()}

    for alias in aliases:
        if alias in metadata:
            return metadata[alias]
        if alias.lower() in lowered:
            return lowered[alias.lower()]

    return None


def _format_metadata(metadata: Optional[Dict[str, Any]], cfg: RerankConfig) -> str:
    if not cfg.use_metadata or not metadata:
        return ""

    lines: List[str] = []

    for field in cfg.metadata_fields:
        raw = _get_metadata_value(metadata, field, cfg)
        value = _normalize_metadata_value(raw)
        if not value:
            continue

        # simple filter to avoid junk year values
        if field == "year":
            if not value.isdigit():
                continue
            if len(value) != 4:
                continue

        label = cfg.metadata_labels.get(field, field.upper())
        lines.append(f"[{label}] {value}")

    if not lines:
        return ""

    return "\n".join(lines)[: cfg.metadata_max_chars]


def _prepare_doc_text(
    text: str,
    metadata: Optional[Dict[str, Any]],
    cfg: RerankConfig,
) -> str:
    text = text or ""
    meta_block = _format_metadata(metadata, cfg)

    if meta_block:
        full = f"{meta_block}\n\n[TEXT]\n{text}"
    else:
        full = text

    return _clean_text(full[: cfg.max_doc_chars], cfg.normalize_whitespace)


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

    if cfg.agg == "logsumexp_topk":
        tau = max(cfg.softmax_temp, 1e-6)
        x = top / tau
        m = x.max()
        return float(tau * (m + np.log(np.exp(x - m).sum())))

    return float(scores.max())


def rerank_with_spans(
    query: str,
    docs: Sequence[DocumentInput],
    config: Optional[RerankConfig] = None,
) -> Tuple[List[float], Optional[List[Optional[Tuple[int, int]]]]]:
    """
    Supports:
      - docs as list[str]
      - docs as list[{"text": "...", "metadata": {...}}]
    """
    cfg = config or RerankConfig()
    model = _get_model(cfg.model_name, cfg.device, cfg.backend)

    print(f"[RERANK] effective backend={cfg.backend}")

    q = _clean_text(query, cfg.normalize_whitespace)

    short_pairs: List[Tuple[str, str]] = []
    short_map: List[int] = []

    win_pairs: List[Tuple[str, str]] = []
    win_map: List[Tuple[int, int, int]] = []  # (doc_i, start, end)

    for i, d in enumerate(docs):
        if isinstance(d, str):
            raw_text = d or ""
            metadata = {}
        else:
            raw_text = d.get("text", "") or ""
            metadata = _coerce_metadata_dict(d.get("metadata", {}))

        prepared = _prepare_doc_text(raw_text, metadata, cfg)

        if len(prepared) <= cfg.window_chars:
            if len(prepared) >= cfg.min_window_chars:
                short_pairs.append((q, prepared))
                short_map.append(i)
        else:
            for s, e, wtxt in _make_windows(prepared, cfg):
                win_pairs.append((q, wtxt))
                win_map.append((i, s, e))

    doc_scores = [float("-inf")] * len(docs)
    spans = [None] * len(docs) if cfg.return_spans else None

    # score short docs
    if short_pairs:
        short_scores = model.predict(
            short_pairs,
            batch_size=cfg.batch_size,
            show_progress_bar=False,
        )
        for idx, sc in zip(short_map, short_scores):
            doc_scores[idx] = float(sc)

    # score windowed docs
    if win_pairs:
        try:
            win_scores = model.predict(
                win_pairs,
                batch_size=cfg.batch_size,
                show_progress_bar=False,
            )
            win_scores = np.asarray(win_scores, dtype=float)

        except RuntimeError as e:
            msg = str(e).lower()
            current_device = cfg.device or _auto_device()

            if ("out of memory" in msg or "oom" in msg) and current_device in {"cuda", "mps"}:
                model_cpu = _get_model(cfg.model_name, "cpu")
                win_scores = model_cpu.predict(
                    win_pairs,
                    batch_size=max(4, cfg.batch_size // 4),
                    show_progress_bar=False,
                )
                win_scores = np.asarray(win_scores, dtype=float)
            else:
                raise

        per_doc: Dict[int, List[float]] = {}
        per_doc_best: Dict[int, Tuple[float, int, int]] = {}

        for (doc_i, s, e), sc in zip(win_map, win_scores):
            scf = float(sc)
            per_doc.setdefault(doc_i, []).append(scf)

            prev = per_doc_best.get(doc_i)
            if cfg.return_spans and (prev is None or scf > prev[0]):
                per_doc_best[doc_i] = (scf, s, e)

        for doc_i, arr_list in per_doc.items():
            agg_score = _aggregate(np.asarray(arr_list, dtype=float), cfg)
            doc_scores[doc_i] = float(agg_score)

            if cfg.return_spans and spans is not None and doc_i in per_doc_best:
                _, s, e = per_doc_best[doc_i]
                spans[doc_i] = (s, e)

    return doc_scores, spans


def rerank(
    query: str,
    docs: Sequence[DocumentInput],
    config: Optional[RerankConfig] = None,
) -> List[float]:
    scores, _ = rerank_with_spans(query, docs, config=config)
    return scores


def top_k_indices(scores: Sequence[float], k: int) -> List[int]:
    if k <= 0:
        return []
    arr = np.asarray(scores, dtype=float)
    order = np.argsort(-arr)
    return order[:k].tolist()