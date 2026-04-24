import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def list_distinct_values(rows: List[Dict[str, Any]], key: str) -> List[str]:
    values = sorted({str(r.get(key, "")).strip() for r in rows if str(r.get(key, "")).strip()})
    return values


def build_component_runs(
    rows: List[Dict[str, Any]],
    *,
    stream_first_chunk_only: bool = False,
) -> List[Dict[str, float]]:
    """
    Construit les composantes par run, en millisecondes.

    Si llm_mode=stream:
      - llm_first_text
      - llm_stream_rest

    Si --stream-first-chunk-only est activé:
      - le temps total affiché devient le temps jusqu'au premier chunk texte
      - llm_stream_rest n'est plus compté dans le total affiché
      - le résiduel est recalculé en conséquence
    """
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

        residual_pipeline_time = max(
            0.0,
            displayed_total
            - query_embedding
            - knn_search
            - reranker
            - llm_first_text
            - counted_stream_rest
            - llm_non_stream
        )

        component_runs.append({
            "query_embedding": query_embedding,
            "knn_search": knn_search,
            "reranker": reranker,
            "llm_first_text": llm_first_text,
            "llm_stream_rest": counted_stream_rest,
            "llm_non_stream": llm_non_stream,
            #"residual_pipeline_time": residual_pipeline_time,
            "total": displayed_total,
            "full_total": full_total,
        })

    return component_runs


def compute_component_stats(
    rows: List[Dict[str, Any]],
    *,
    stream_first_chunk_only: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Retourne mean/std en secondes pour chaque composante.
    """
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
        #"residual_pipeline_time",
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


def plot_response_time_breakdown(
    rows: List[Dict[str, Any]],
    output_path: str | Path,
    title: str,
    *,
    stream_first_chunk_only: bool = False,
    alternate_labels: bool = False,
    xmax: Optional[float] = None,
    edge_label_side: str = "auto",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = compute_component_stats(
        rows,
        stream_first_chunk_only=stream_first_chunk_only,
    )
    if not stats:
        raise ValueError("Aucune donnée à tracer.")

    mode = infer_plot_mode(rows)

    total_mean = stats["total"]["mean"]
    total_std = stats["total"]["std"]

    if mode == "stream":
        if stream_first_chunk_only:
            components = [
                ("query_embedding", "Embedding\ninference"),
                ("knn_search", "Vector\nretrieval"),
                ("reranker", "Reranker"),
                ("llm_first_text", "LLM\nfirst text chunk"),
                #("residual_pipeline_time", "Residual\nprocessing\ntime"),
            ]
        else:
            components = [
                ("query_embedding", "Embedding\ninference"),
                ("knn_search", "Vector\nretrieval"),
                ("reranker", "Reranker"),
                ("llm_first_text", "LLM\nfirst text chunk"),
                ("llm_stream_rest", "LLM\nstream rest"),
                #("residual_pipeline_time", "Residual\nprocessing\ntime"),
            ]
    elif mode == "non_stream":
        components = [
            ("query_embedding", "Embedding\ninference"),
            ("knn_search", "Vector\nretrieval"),
            ("reranker", "Reranker"),
            ("llm_non_stream", "Answer\ngeneration (LLM)"),
            #("residual_pipeline_time", "Residual\nprocessing\ntime"),
        ]
    else:
        if stream_first_chunk_only:
            components = [
                ("query_embedding", "Embedding\ninference"),
                ("knn_search", "Vector\nretrieval"),
                ("reranker", "Reranker"),
                ("llm_first_text", "LLM\nfirst text chunk"),
                ("llm_non_stream", "LLM\nnon-stream"),
                #("residual_pipeline_time", "Residual\nprocessing\ntime"),
            ]
        else:
            components = [
                ("query_embedding", "Embedding\ninference"),
                ("knn_search", "Vector\nretrieval"),
                ("reranker", "Reranker"),
                ("llm_first_text", "LLM\nfirst text chunk"),
                ("llm_stream_rest", "LLM\nstream rest"),
                ("llm_non_stream", "LLM\nnon-stream"),
                #("residual_pipeline_time", "Residual\nprocessing\ntime"),
            ]

    print("\nBreakdown des parties du graphe :")
    for key, label in components:
        mean_val = stats[key]["mean"]
        std_val = stats[key]["std"]
        print(f"  - {label.replace(chr(10), ' ')}: mean={mean_val:.4f}s, std={std_val:.4f}s")
    print(f"  - Total: mean={stats['total']['mean']:.4f}s, std={stats['total']['std']:.4f}s")

    colors = {
        "query_embedding": "#4C78A8",
        "knn_search": "#F58518",
        "reranker": "#E45756",
        "llm_first_text": "#B279A2",
        "llm_stream_rest": "#FF9DA6",
        "llm_non_stream": "#B1A0DE",
        "residual_pipeline_time": "#9D9D9D",
    }

    fig, ax = plt.subplots(figsize=(15, 3))

    y = 0
    outer_height = 0.44
    inner_height = 0.18
    outer_left = 0.0
    label_edge_margin = 0.15 # Modifer cette valeur pour que la ligne aparaisse ou pas
    label_offset = 0.15

    ax.barh(
        [y],
        [total_mean],
        #xerr=[total_std],
        height=outer_height,
        color="white",
        edgecolor="none",
        linewidth=1.5,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.4,
            "capsize": 8,
            "capthick": 1.4,
        },
        zorder=1,
    )

    # ax.errorbar(
    #        x=total_mean,
    #    y=y,
    #    xerr=total_std,
    #    fmt="none",
    #    ecolor="black",
    #    elinewidth=1.6,
    #    capsize=8,
    #    capthick=1.6,
    #    zorder=10,
    #)

    # Selon cette valeur, s'affiche ou pas dans la légende
    min_label_width = 0.0005
    legend_items = []
    left = 0.0

    for i, (key, label) in enumerate(components):
        width = stats[key]["mean"]
        if width <= 0:
            continue

        bar_color = colors[key]

        ax.barh(
            [y],
            [width],
            left=[left],
            height=inner_height,
            color=bar_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )

        center_x = left + width / 2.0

        # Pourcentage de la composante par rapport au total
        pct = 100.0 * width / total_mean if total_mean > 0 else 0.0
        pct_text = f"{pct:.0f}%"

        # Label du composant au-dessus / dessous
        if alternate_labels:
            if i % 2 == 0:
                text_y = y + 0.12
                va = "bottom"
            else:
                text_y = y - 0.12
                va = "top"
        else:
            text_y = y + 0.12
            va = "bottom"

        too_close_edge = (
            center_x < (outer_left + label_edge_margin)
            or center_x > (total_mean - label_edge_margin)
        )

        if too_close_edge and edge_label_side != "none":
            if edge_label_side == "left":
                text_x = center_x - label_offset
                ha = "right"
            elif edge_label_side == "right":
                text_x = center_x + label_offset
                ha = "left"
            else:  # auto
                if center_x < total_mean / 2:
                    text_x = center_x + label_offset
                    ha = "left"
                else:
                    text_x = center_x - label_offset
                    ha = "right"

            ax.annotate(
                label,
                xy=(center_x, text_y),
                xytext=(text_x, text_y),
                ha=ha,
                va=va,
                fontsize=10,
                color=bar_color,   # même couleur que la boîte
                arrowprops=dict(
                    arrowstyle="-",
                    color="black",
                    lw=1.0,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=4,
            )
        else:
            ax.text(
                center_x,
                text_y,
                label,
                ha="center",
                va=va,
                fontsize=10,
                color=bar_color,   # même couleur que la boîte
                clip_on=False,
                zorder=4,
            )

        # Pourcentage à l'intérieur de la barre si assez d'espace
        # aligné à gauche, centré verticalement
        pct_left_padding = 0.03
        min_width_for_pct = 0.22

        if width >= min_width_for_pct:
            ax.text(
                left + pct_left_padding,
                y,
                pct_text,
                ha="left",
                va="center",
                fontsize=18,
                color="white",
                fontweight="bold",
                zorder=5,
            )

        left += width

    if legend_items:
        ax.legend(
            handles=legend_items,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(len(legend_items), 3),
            frameon=False,
            fontsize=10,
        )

    ax.set_yticks([y])
    ax.set_yticklabels(["Total\nResponse"])
    ax.set_xlabel("Time (sec)")
    ax.set_title(title, pad=16)

    if xmax is not None:
        ax.set_xlim(0, xmax)
    else:
        xmax_auto = max(total_mean + total_std, left, 0.5)
        ax.set_xlim(0, xmax_auto * 1.06)

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_default_title(
    rows: List[Dict[str, Any]],
    *,
    stream_first_chunk_only: bool = False,
) -> str:
    mode = infer_plot_mode(rows)
    labels = list_distinct_values(rows, "label")

    label_part = labels[0] if len(labels) == 1 else f"{len(labels)} labels"
    n = len(rows)

    if mode == "stream" and stream_first_chunk_only:
        return f"Mean Time to First Text Chunk"
    if mode == "stream" or mode == "non_stream":
        return f"Mean Total Response Time"
    return f"Mean Total Response Time (with std) and subcomponents — mixed modes — {label_part} — n={n}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lit le CSV de benchmark et génère un graphique de breakdown des temps.")
    parser.add_argument("--csv", default=PROJECT_ROOT / "data" / "evaluation" / "benchmark_timing_results.csv", help="Chemin du CSV produit par benchmark_to_csv.py")

    parser.add_argument("--output", default=PROJECT_ROOT / "plot" / "generated" / "benchmark_timing_breakdown.png", help="Chemin du PNG de sortie")
    parser.add_argument("--label", default=None, help="Filtrer par label exact")
    parser.add_argument("--batch-id", default=None, help="Filtrer par batch_id exact")
    parser.add_argument("--query-contains", default=None, help="Filtrer si la question contient ce texte")
    parser.add_argument("--question-index", type=int, default=None, help="Filtrer par question_index")
    parser.add_argument("--title", default=None, help="Titre du graphique")
    parser.add_argument("--list-labels", action="store_true", help="Afficher les labels disponibles puis quitter")
    parser.add_argument("--list-batches", action="store_true", help="Afficher les batch_id disponibles puis quitter")
    parser.add_argument("--stream-first-chunk-only", action="store_true", help="Pour les runs stream, tronque le graphe au premier chunk texte reçu et ignore le reste du stream.")
    parser.add_argument("--alternate-labels", action="store_true", help="Alterne les labels des composants au-dessus et en-dessous de la barre.")
    parser.add_argument("--xmax", type=float, default=None, help="Force la limite maximale de l'axe X (en secondes).")
    parser.add_argument("--edge-label-side", choices=["none","auto", "left", "right"], default="auto", help="Gestion des labels proches du bord : none, auto, left, right.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        rows = read_csv_rows(args.csv)

        if args.list_labels:
            print("Labels disponibles:")
            for value in list_distinct_values(rows, "label"):
                print(value)
            return 0

        if args.list_batches:
            print("Batch IDs disponibles:")
            for value in list_distinct_values(rows, "batch_id"):
                print(value)
            return 0

        filtered = filter_rows(
            rows,
            label=args.label,
            batch_id=args.batch_id,
            query_contains=args.query_contains,
            question_index=args.question_index,
        )

        if not filtered:
            print("Aucune ligne après filtrage.", file=sys.stderr)
            return 1

        title = args.title or build_default_title(
            filtered,
            stream_first_chunk_only=args.stream_first_chunk_only,
        )
        title = ""

        print(f"Nombre de runs utilisés : {len(filtered)}")
        print(f"Mode LLM détecté       : {infer_plot_mode(filtered)}")
        plot_response_time_breakdown(
            filtered,
            args.output,
            title,
            stream_first_chunk_only=args.stream_first_chunk_only,
            alternate_labels=args.alternate_labels,
            xmax=args.xmax,
            edge_label_side=args.edge_label_side,
        )
        print(f"Graphique sauvegardé   : {args.output}")

        return 0

    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())