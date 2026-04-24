import json
import math
import csv
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional
from adjustText import adjust_text
import matplotlib.pyplot as plt
import argparse

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_GLOB = str(BASE_DIR / "data" / "reranker" / "runs" / "*.jsonl")

OUTPUT_CSV = BASE_DIR / "plot" / "reranker"/ "generated" / "reranker_metrics_vs_time_summary.csv"

OUTPUT_NDCG_PNG = BASE_DIR / "plot" / "generated" / "reranker_ndcg_vs_time.png"
OUTPUT_PRECISION_PNG = BASE_DIR / "plot" / "generated" / "reranker_precision_vs_time.png"


# IMPORTANT: aligned with your reference NDCG script
TOP_K = 10
GAIN_FIELD = "relevance_score"
USE_EXPONENTIAL_GAIN = False
EMPTY_IDCG_VALUE = 0.0

RECURSIVE_GLOB = False

GENERIC_FILENAMES = {
    "generations",
    "generations_final",
    "generations_enriched_with_ndcg",
    "output",
    "results",
}

# Scientific/publication-friendly distinct palette
DISTINCT_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#332288",  # indigo
    "#117733",  # dark green
    "#882255",  # dark magenta
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # magenta
]


# ============================================================
# I/O
# ============================================================

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    items = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL line in {path} at line {i}: {e}") from e

    return items


def save_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Stats helpers
# ============================================================

def mean(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def std(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return var ** 0.5


def safe_float(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# Chunk field access
# ============================================================

def get_chunk_field(chunk: Dict[str, Any], field_name: str, default: Any = None) -> Any:
    if field_name in chunk:
        return chunk[field_name]

    metadata = chunk.get("metadata")
    if isinstance(metadata, dict) and field_name in metadata:
        return metadata[field_name]

    return default


def extract_gain(chunk: Dict[str, Any], gain_field: str) -> float:
    if gain_field == "is_relevant":
        value = get_chunk_field(chunk, "is_relevant", False)
        return 1.0 if bool(value) else 0.0

    value = get_chunk_field(chunk, gain_field, 0.0)
    return safe_float(value)


def extract_binary(chunk: Dict[str, Any]) -> int:
    value = get_chunk_field(chunk, "is_relevant", False)
    return 1 if bool(value) else 0


# ============================================================
# Ranking metrics
# ============================================================

def dcg(scores: List[float], use_exponential_gain: bool = False) -> float:
    total = 0.0
    for i, rel in enumerate(scores):
        denom = math.log2(i + 2)
        gain = (2 ** rel - 1) if use_exponential_gain else rel
        total += gain / denom
    return total


def compute_ndcg_for_record(
    record: Dict[str, Any],
    top_k: Optional[int] = None,
    gain_field: str = "relevance_score",
    use_exponential_gain: bool = False,
    empty_idcg_value: float = 0.0,
) -> float:
    all_chunks = record.get("retrieved_chunks", []) or []
    if not all_chunks:
        return empty_idcg_value

    ranked_chunks = all_chunks[:top_k] if top_k is not None else all_chunks

    actual_scores = [extract_gain(chunk, gain_field) for chunk in ranked_chunks]
    all_scores = [extract_gain(chunk, gain_field) for chunk in all_chunks]

    if top_k is not None:
        ideal_scores = sorted(all_scores, reverse=True)[:top_k]
    else:
        ideal_scores = sorted(all_scores, reverse=True)

    actual_dcg = dcg(actual_scores, use_exponential_gain=use_exponential_gain)
    ideal_dcg = dcg(ideal_scores, use_exponential_gain=use_exponential_gain)

    if ideal_dcg == 0.0:
        return empty_idcg_value

    return actual_dcg / ideal_dcg


def precision_at_k(binary_labels: List[int], k: int) -> float:
    if not binary_labels:
        return 0.0
    observed = binary_labels[:k]
    denom = k if k > 0 else 1
    return sum(observed) / denom


def hit_at_k(binary_labels: List[int], k: int) -> float:
    return 1.0 if any(binary_labels[:k]) else 0.0


def reciprocal_rank(binary_labels: List[int]) -> float:
    for i, value in enumerate(binary_labels):
        if value == 1:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================
# Timing / pool / returned-k
# ============================================================

def infer_pool_size(record: Dict[str, Any]) -> int:
    mm = record.get("method_metadata", {}) or {}

    for key in ("rerank_pool", "fetch_k", "retrieve_k", "n_candidates", "candidate_pool"):
        value = mm.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    chunks = record.get("retrieved_chunks", []) or []
    return max(1, len(chunks))


def infer_returned_k(record: Dict[str, Any]) -> int:
    mm = record.get("method_metadata", {}) or {}

    for key in ("k_value", "rerank_k", "top_k"):
        value = mm.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    chunks = record.get("retrieved_chunks", []) or []
    return len(chunks)


def infer_rerank_time_ms(record: Dict[str, Any]) -> Optional[float]:
    bd = record.get("latency_breakdown_ms", {}) or {}

    for key in ("rerank_ms", "candidate_loading_and_rerank_ms", "retrieval_ms"):
        value = bd.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    return None


def infer_time_per_candidate_ms(record: Dict[str, Any]) -> Optional[float]:
    total_ms = infer_rerank_time_ms(record)
    if total_ms is None:
        return None

    pool = infer_pool_size(record)
    if pool <= 0:
        return None

    return total_ms / pool


# ============================================================
# Labels
# ============================================================

def infer_label_from_path(path_str: str) -> str:
    p = Path(path_str)
    stem = p.stem

    if stem in GENERIC_FILENAMES and p.parent.name:
        return p.parent.name

    if stem.startswith("generations_"):
        return stem[len("generations_"):]

    return stem


# ============================================================
# Summary per file
# ============================================================

def summarize_file(path_str: str) -> Optional[Dict[str, Any]]:
    records = load_json_or_jsonl(path_str)

    if not records:
        return None

    ndcg_values = []
    precision_values = []
    hit_values = []
    mrr_values = []

    rerank_time_values = []
    time_per_candidate_values = []

    pool_sizes = []
    returned_ks = []

    for record in records:
        chunks = record.get("retrieved_chunks", []) or []

        binary_labels = [extract_binary(c) for c in chunks]

        precision_values.append(precision_at_k(binary_labels, TOP_K))
        hit_values.append(hit_at_k(binary_labels, TOP_K))
        mrr_values.append(reciprocal_rank(binary_labels))

        ndcg = compute_ndcg_for_record(
            record,
            top_k=TOP_K,
            gain_field=GAIN_FIELD,
            use_exponential_gain=USE_EXPONENTIAL_GAIN,
            empty_idcg_value=EMPTY_IDCG_VALUE,
        )
        ndcg_values.append(ndcg)

        total_ms = infer_rerank_time_ms(record)
        if total_ms is not None:
            rerank_time_values.append(total_ms)

        per_candidate_ms = infer_time_per_candidate_ms(record)
        if per_candidate_ms is not None:
            time_per_candidate_values.append(per_candidate_ms)

        pool_sizes.append(infer_pool_size(record))
        returned_ks.append(infer_returned_k(record))

    return {
        "file": path_str,
        "label": infer_label_from_path(path_str),
        "n_questions": len(records),

        "mean_ndcg": mean(ndcg_values),
        "std_ndcg": std(ndcg_values),

        "mean_precision": mean(precision_values),
        "std_precision": std(precision_values),

        "mean_hit": mean(hit_values),
        "std_hit": std(hit_values),

        "mean_mrr": mean(mrr_values),
        "std_mrr": std(mrr_values),

        "mean_time_per_query_ms": mean(rerank_time_values),
        "std_time_per_query_ms": std(rerank_time_values),

        "mean_time_per_candidate_ms": mean(time_per_candidate_values),
        "std_time_per_candidate_ms": std(time_per_candidate_values),

        "mean_pool_size": mean(pool_sizes),
        "mean_returned_k": mean(returned_ks),
    }


# ============================================================
# Generic plotter
# ============================================================

def plot_metric(
    rows: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    output_png: str | Path,
    title: str,
    show_legend: bool = False,
    show_point_labels: bool = False,
) -> None:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    valid_rows = [
        r for r in rows
        if r is not None and r["n_questions"] > 0 and r["mean_time_per_query_ms"] > 0
    ]

    if not valid_rows:
        print(f"No data to plot for {metric_key}.")
        return

    valid_rows = sorted(valid_rows, key=lambda r: r[metric_key], reverse=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "legend.edgecolor": "black",
    })

    fig, ax = plt.subplots(figsize=(11, 7))

    texts = []
    handles = []
    labels = []

    for i, row in enumerate(valid_rows):
        x = row["mean_time_per_query_ms"]
        y = row[metric_key]
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

        handle = ax.scatter(
            [x],
            [y],
            s=95,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

        if show_point_labels:
            txt = ax.text(
                x,
                y,
                row["label"],
                fontsize=9,
                color="black",
                ha="left",
                va="bottom",
            )
            texts.append(txt)

        if show_legend:
            legend_text = (
                f"{row['label']} | "
                f"{metric_label}={y:.4f} | "
                f"t/q={x:.1f} ms"
            )
            handles.append(handle)
            labels.append(legend_text)

    if show_point_labels and texts:
        adjust_text(
            texts,
            ax=ax,
            expand_points=(1.01, 1.03),
            expand_text=(1.05, 1.10),
            force_points=0.15,
            force_text=0.20,
            only_move={"text": "y", "points": "y"},
        )

    if show_legend and handles:
        ax.legend(
            handles,
            labels,
            loc="best",
            fontsize=8.4,
            title_fontsize=9.0,
            frameon=False,
        )

    ax.set_xlabel("Mean cross-encoder time per query (ms)", color="black")
    ax.set_ylabel(metric_label, color="black")
    ax.set_title(title, color="black")

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.subplots_adjust(bottom=0.14)

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reranker metrics vs time."
    )

    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Display legend on plots."
    )
    parser.add_argument(
        "--show-point-labels",
        action="store_true",
        help="Display text labels next to points."
    )

    return parser.parse_args()

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    files = sorted(glob.glob(INPUT_GLOB, recursive=RECURSIVE_GLOB))

    if not files:
        raise FileNotFoundError(f"No files found with pattern: {INPUT_GLOB}")

    print(f"{len(files)} file(s) found.")
    print(f"Evaluating metrics at k={TOP_K}")
    print(f"NDCG gain field: {GAIN_FIELD}")
    print(f"NDCG gain mode: {'exponential' if USE_EXPONENTIAL_GAIN else 'linear'}")

    rows = []

    for path in files:
        try:
            summary = summarize_file(path)
            if summary is None:
                continue

            rows.append(summary)

            print(
                f"- {summary['label']}: "
                f"n={summary['n_questions']}, "
                f"NDCG@{TOP_K}={summary['mean_ndcg']:.4f}, "
                f"Precision@{TOP_K}={summary['mean_precision']:.4f}, "
                f"Hit@{TOP_K}={summary['mean_hit']:.4f}, "
                f"MRR={summary['mean_mrr']:.4f}, "
                f"t/q={summary['mean_time_per_query_ms']:.2f} ms, "
                f"t/c={summary['mean_time_per_candidate_ms']:.2f} ms, "
                f"pool≈{summary['mean_pool_size']:.0f}, "
                f"returned≈{summary['mean_returned_k']:.0f}"
            )
        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    rows.sort(key=lambda r: r["mean_ndcg"], reverse=True)
    save_csv(rows, OUTPUT_CSV)


    plot_metric(
        rows,
        metric_key="mean_ndcg",
        metric_label=f"NDCG@{TOP_K}",
        output_png=OUTPUT_NDCG_PNG,
        title="Cross-encoder comparison",
        show_legend=args.show_legend,
        show_point_labels=args.show_point_labels,
    )

    plot_metric(
        rows,
        metric_key="mean_precision",
        metric_label=f"Precision@{TOP_K}",
        output_png=OUTPUT_PRECISION_PNG,
        title="Cross-encoder comparison",
        show_legend=args.show_legend,
        show_point_labels=args.show_point_labels,
    )



    print(f"\nSummary CSV written to: {OUTPUT_CSV}")
    print(f"NDCG plot written to: {OUTPUT_NDCG_PNG}")
    print(f"Precision plot written to: {OUTPUT_PRECISION_PNG}")


if __name__ == "__main__":
    main()