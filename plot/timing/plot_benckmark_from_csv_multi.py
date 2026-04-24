import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_PLOT_OUTPUT = PROJECT_ROOT / "plot" / "generated" / "benchmark_timing_breakdown_multi.png"


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def read_csv_rows(csv_path: str | Path) -> List[Dict[str, Any]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def list_csv_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Ce n'est pas un dossier: {folder}")

    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"Aucun fichier CSV trouvé dans: {folder}")
    return files


def filter_rows(
    rows: List[Dict[str, Any]],
    *,
    label: Optional[str] = None,
    batch_id: Optional[str] = None,
    query_contains: Optional[str] = None,
    question_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out = rows

    if label is not None:
        out = [r for r in out if r.get("label") == label]

    if batch_id is not None:
        out = [r for r in out if r.get("batch_id") == batch_id]

    if query_contains is not None:
        q = query_contains.lower()
        out = [r for r in out if q in str(r.get("query", "")).lower()]

    if question_index is not None:
        out = [r for r in out if str(r.get("question_index")) == str(question_index)]

    return out


def build_component_runs(
    rows: List[Dict[str, Any]],
    *,
    stream_first_chunk_only: bool = False,
) -> List[Dict[str, float]]:
    component_runs: List[Dict[str, float]] = []

    for row in rows:
        llm_mode = str(row.get("llm_mode", "")).strip().lower()

        query_embedding = _to_float(row.get("timings_ms.query_embedding_ms"))

        knn_search = (
            _to_float(row.get("timings_ms.run_file_load_ms")) +
            _to_float(row.get("timings_ms.run_file_doc_embeddings_ms")) +
            _to_float(row.get("timings_ms.chroma_prepare_ms")) +
            _to_float(row.get("timings_ms.chroma_query_ms")) +
            _to_float(row.get("timings_ms.normalize_results_ms"))
        )

        reranker = (
            _to_float(row.get("timings_ms.filter_document_text_ms")) +
            _to_float(row.get("timings_ms.sort_without_rerank_ms")) +
            _to_float(row.get("timings_ms.dedup_ms")) +
            _to_float(row.get("timings_ms.dedup_method_ms")) +
            _to_float(row.get("timings_ms.rerank_ms")) +
            _to_float(row.get("timings_ms.build_final_order_ms")) +
            _to_float(row.get("timings_ms.construct_output_ms"))
        )

        full_total = _to_float(row.get("timings_ms.total_ms"))

        llm_first_text = 0.0
        llm_stream_rest = 0.0
        llm_non_stream = 0.0

        displayed_total = full_total

        if llm_mode == "stream":
            first_text = _to_float(row.get("timings_ms.llm_first_text_chunk_ms"))
            stream_total = _to_float(row.get("timings_ms.llm_stream_total_ms"))

            llm_first_text = first_text
            llm_stream_rest = max(0.0, stream_total - first_text)

            if stream_first_chunk_only:
                displayed_total = max(0.0, full_total - llm_stream_rest)
        else:
            llm_non_stream = _to_float(row.get("timings_ms.llm_total_ms"))

        counted_stream_rest = 0.0 if stream_first_chunk_only else llm_stream_rest

        component_runs.append({
            "query_embedding": query_embedding,
            "knn_search": knn_search,
            "reranker": reranker,
            "llm_first_text": llm_first_text,
            "llm_stream_rest": counted_stream_rest,
            "llm_non_stream": llm_non_stream,
            "total": displayed_total,
        })

    return component_runs


def compute_component_stats(
    rows: List[Dict[str, Any]],
    *,
    stream_first_chunk_only: bool = False,
) -> Dict[str, Dict[str, float]]:
    component_runs = build_component_runs(
        rows,
        stream_first_chunk_only=stream_first_chunk_only,
    )
    if not component_runs:
        return {}

    keys = [
        "query_embedding",
        "knn_search",
        "reranker",
        "llm_first_text",
        "llm_stream_rest",
        "llm_non_stream",
        "total",
    ]

    stats: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values_sec = [run[key] / 1000.0 for run in component_runs]
        avg, std = _mean_std(values_sec)
        stats[key] = {
            "mean": avg,
            "std": std,
        }

    return stats


def infer_plot_mode(rows: List[Dict[str, Any]]) -> str:
    modes = {str(r.get("llm_mode", "")).strip().lower() for r in rows}
    modes.discard("")
    if modes == {"stream"}:
        return "stream"
    if modes == {"non_stream"}:
        return "non_stream"
    return "mixed"


def components_for_mode(mode: str, stream_first_chunk_only: bool) -> List[Tuple[str, str]]:
    if mode == "stream":
        if stream_first_chunk_only:
            return [
                ("query_embedding", "Embedding"),
                ("knn_search", "Vector retrieval"),
                ("reranker", "Reranker"),
                ("llm_first_text", "LLM first text chunk"),
            ]
        return [
            ("query_embedding", "Embedding"),
            ("knn_search", "Vector retrieval"),
            ("reranker", "Reranker"),
            ("llm_first_text", "LLM first text chunk"),
            ("llm_stream_rest", "LLM stream rest"),
        ]

    if mode == "non_stream":
        return [
            ("query_embedding", "Embedding"),
            ("knn_search", "Vector retrieval"),
            ("reranker", "Reranker"),
            ("llm_non_stream", "LLM answer generation"),
        ]

    if stream_first_chunk_only:
        return [
            ("query_embedding", "Embedding"),
            ("knn_search", "Vector retrieval"),
            ("reranker", "Reranker"),
            ("llm_first_text", "LLM first text chunk"),
            ("llm_non_stream", "LLM non-stream"),
        ]

    return [
        ("query_embedding", "Embedding"),
        ("knn_search", "Vector retrieval"),
        ("reranker", "Reranker"),
        ("llm_first_text", "LLM first text chunk"),
        ("llm_stream_rest", "LLM stream rest"),
        ("llm_non_stream", "LLM non-stream"),
    ]


def plot_multiple_csvs_breakdown(
    datasets: List[Tuple[str, List[Dict[str, Any]]]],
    output_path: str | Path,
    *,
    stream_first_chunk_only: bool = False,
    xmax: Optional[float] = None,
    title: str = "",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = []
    all_totals = []

    for dataset_name, rows in datasets:
        if not rows:
            continue

        mode = infer_plot_mode(rows)
        stats = compute_component_stats(
            rows,
            stream_first_chunk_only=stream_first_chunk_only,
        )
        if not stats:
            continue

        components = components_for_mode(mode, stream_first_chunk_only)

        prepared.append({
            "name": dataset_name,
            "rows": rows,
            "mode": mode,
            "stats": stats,
            "components": components,
        })
        all_totals.append(stats["total"]["mean"])

    if not prepared:
        raise ValueError("Aucune donnée exploitable à tracer.")

    colors = {
        "query_embedding": "#4C78A8",
        "knn_search": "#F58518",
        "reranker": "#E45756",
        "llm_first_text": "#72B7B2",
        "llm_stream_rest": "#9FBEBB",
        "llm_non_stream": "#72B7B2",
    }

    fig_height = max(3.0, 1.2 * len(prepared) + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    y_positions = list(range(len(prepared)))

    for y, item in zip(y_positions, prepared):
        left = 0.0
        total_mean = item["stats"]["total"]["mean"]
        total_std = item["stats"]["total"]["std"]

        for key, label in item["components"]:
            width = item["stats"].get(key, {}).get("mean", 0.0)
            if width <= 0:
                continue

            ax.barh(
                [y],
                [width],
                left=[left],
                height=0.55,
                color=colors.get(key, "#999999"),
                edgecolor="black",
                linewidth=0.8,
            )
            #
            # pct = 100.0 * width / total_mean if total_mean > 0 else 0.0
            # if width >= 0.18:
            #     ax.text(
            #         left + 0.02,
            #         y,
            #         f"{pct:.0f}%",
            #         ha="left",
            #         va="center",
            #         fontsize=10,
            #         color="white",
            #         fontweight="bold",
            #     )

            left += width

        # Show error bars
        # ax.errorbar(
        #     x=total_mean,
        #     y=y,
        #     xerr=total_std,
        #     fmt="none",
        #     ecolor="black",
        #     elinewidth=1.2,
        #     capsize=5,
        #     capthick=1.2,
        #     zorder=5,
        # )

        # ax.text(
        #     total_mean + 0.03,
        #     y,
        #     f"{total_mean:.2f}s",
        #     va="center",
        #     ha="left",
        #     fontsize=10,
        # )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([item["name"] for item in prepared])
    ax.set_xlabel("Time (sec)")
    ax.set_title(title or "Benchmark comparison across CSV files")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    if xmax is not None:
        ax.set_xlim(0, xmax)
    else:
        ax.set_xlim(0, max(all_totals) * 1.20 if all_totals else 1.0)

    legend_keys = []
    for item in prepared:
        for key, _ in item["components"]:
            if key not in legend_keys:
                legend_keys.append(key)

    legend_labels = {
        "query_embedding": "Embedding",
        "knn_search": "Vector retrieval",
        "reranker": "Reranker",
        "llm_first_text": "LLM first text chunk",
        "llm_stream_rest": "LLM stream rest",
        "llm_non_stream": "LLM non-stream",
    }

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[k], ec="black")
        for k in legend_keys
    ]
    labels = [legend_labels[k] for k in legend_keys]
    #
    # if handles:
    #     ax.legend(handles, labels, loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lit tous les CSV d'un dossier et génère un graphique comparatif unique."
    )
    parser.add_argument(
        "--csv-dir",
        default=DEFAULT_EVAL_DIR,
        help="Dossier contenant les CSV à comparer (par défaut: dossier courant)."
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "plot" / "generated" / "benchmark_timing_breakdown_multi.png",
        help="Chemin du PNG de sortie"
    )
    parser.add_argument("--label", default=None, help="Filtrer par label exact")
    parser.add_argument("--batch-id", default=None, help="Filtrer par batch_id exact")
    parser.add_argument("--query-contains", default=None, help="Filtrer si la question contient ce texte")
    parser.add_argument("--question-index", type=int, default=None, help="Filtrer par question_index")
    parser.add_argument("--title", default="", help="Titre du graphique")
    parser.add_argument(
        "--stream-first-chunk-only",
        action="store_true",
        help="Pour les runs stream, tronque le graphe au premier chunk texte reçu."
    )
    parser.add_argument("--xmax", type=float, default=None, help="Force la limite max de l'axe X (sec)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        csv_files = list_csv_files(args.csv_dir)
        print("CSV trouvés :")
        for f in csv_files:
            print(f" - {f}")

        datasets: List[Tuple[str, List[Dict[str, Any]]]] = []

        for csv_file in csv_files:
            rows = read_csv_rows(csv_file)
            filtered = filter_rows(
                rows,
                label=args.label,
                batch_id=args.batch_id,
                query_contains=args.query_contains,
                question_index=args.question_index,
            )

            if not filtered:
                print(f"[SKIP] {csv_file.name}: aucune ligne après filtrage")
                continue

            datasets.append((csv_file.stem, filtered))
            print(f"[OK] {csv_file.name}: {len(filtered)} runs retenus")

        if not datasets:
            print("Aucune donnée à tracer après filtrage.", file=sys.stderr)
            return 1

        plot_multiple_csvs_breakdown(
            datasets,
            args.output,
            stream_first_chunk_only=args.stream_first_chunk_only,
            xmax=args.xmax,
            title=args.title,
        )

        print(f"Graphique sauvegardé : {args.output}")
        return 0

    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())