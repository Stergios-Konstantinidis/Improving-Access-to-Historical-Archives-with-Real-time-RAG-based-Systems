#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

SUMMARY_FILENAME = "ragas_summary.json"

method_labels = {
    "bm25s": "BM25s",
    "semantic": "Semantic",
    "corrected": "Ours (Semantic+LLM)",
    "reranker": "Ours (Semantic+LLM+Reranker)",
}

metric_labels = {
    "chunk_relevance_ndcg_at_k": "Chunk relevance NDCG@k",
    "chunk_relevance_precision_at_k": "Chunk relevance Precision@k",
    "answer_relevancy": "Answer relevancy",
    "context_relevance": "Context relevance",
    "answer_correctness": "Answer correctness",
    "faithfulness": "Faithfulness",
    "llm_correct_or_not": "LLM correctness",
}

method_styles = {
    "bm25s": {
        "color": "0.7",
        "linestyle": ":",
        "linewidth": 2.2,
        "marker": "s",
    },
    "semantic": {
        "color": "0.6",
        "linestyle": "-",
        "linewidth": 2.2,
        "marker": "^",
    },
    "corrected": {
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.2,
        "marker": "o",
    },
    "reranker": {
        "color": "black",
        "linestyle": "-",
        "linewidth": 3.0,
        "marker": "D",
    },
}
pretty_titles = {
    "answer_correctness": "Answer correctness by retrieval depth (k)",
    "answer_relevancy": "Answer relevancy by retrieval depth (k)",
    "context_relevance": "Context relevance by retrieval depth (k)",
    "faithfulness": "Faithfulness by retrieval depth (k)",
    "llm_correct_or_not": "LLM correctness by retrieval depth (k)",
    "chunk_relevance_ndcg_at_k": "Chunk relevance NDCG@k by retrieval depth (k)",
    "chunk_relevance_precision_at_k": "Chunk relevance Precision@k by retrieval depth (k)",
}

pretty_y = {
    "answer_correctness": "Answer correctness",
    "answer_relevancy": "Answer relevancy",
    "context_relevance": "Context relevance",
    "faithfulness": "Faithfulness",
    "llm_correct_or_not": "LLM correctness",
    "chunk_relevance_ndcg_at_k": "Chunk relevance NDCG@k",
    "chunk_relevance_precision_at_k": "Chunk relevance Precision@k",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_name(folder_name: str) -> Optional[Dict[str, Any]]:
    """
    Exemples gérés :
      - generations_k5_answered_bm25s
      - generations_k10_answered_corrected
      - generations_k20_answered_semantic
      - generations_reranker_k5_answered
      - generations_reranker_k20_answered
    """
    folder = folder_name.strip()

    m = re.match(r"^generations_reranker_k(\d+)_answered$", folder)
    if m:
        return {
            "method": "reranker",
            "k": int(m.group(1)),
        }

    m = re.match(r"^generations_k(\d+)_answered_([a-zA-Z0-9_]+)$", folder)
    if m:
        return {
            "method": m.group(2),
            "k": int(m.group(1)),
        }

    return None


def collect_summaries(root_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue

        parsed = parse_run_name(subdir.name)
        if parsed is None:
            continue

        summary_path = subdir / SUMMARY_FILENAME
        if not summary_path.exists():
            print(f"[WARN] summary absent: {summary_path}")
            continue

        try:
            summary = load_json(summary_path)
        except Exception as e:
            print(f"[WARN] impossible de lire {summary_path}: {e}")
            continue

        row = {
            "folder": subdir.name,
            "method": parsed["method"],
            "k": parsed["k"],
            **summary,
        }
        rows.append(row)

    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    columns = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: List[Dict[str, Any]], metric_name: str, output_path: Path) -> None:
    """
    Un graphique par métrique, avec comparaison de toutes les méthodes.
    C'est cette fonction qui colle le mieux à la consigne reçue.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if metric_name in row and isinstance(row.get("k"), int):
            grouped[row["method"]].append(row)

    if not grouped:
        print(f"[WARN] aucune donnée trouvée pour {metric_name}")
        return

    plt.figure(figsize=(9, 5.8))
    method_order = ["bm25s", "semantic", "corrected", "reranker"]

    for method in method_order:
        if method not in grouped:
            continue

        items = sorted(grouped[method], key=lambda x: x["k"])
        xs = [item["k"] for item in items]
        ys = [item[metric_name] for item in items]

        style = method_styles.get(method, {})
        plt.plot(
            xs,
            ys,
            label=method_labels.get(method, method),
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            **style,
        )

    plt.xlabel("Top-k retrieved chunks")
    plt.ylabel(pretty_y.get(metric_name, metric_name))
    plt.title(pretty_titles.get(metric_name, f"{metric_name} by retrieval depth (k)"))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.25)
    plt.legend(
        title="Method",
        loc="lower right",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        handlelength=3.2,
        handletextpad=0.9,
        borderpad=0.9,
        labelspacing=0.7,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_all_metrics_across_methods(
    rows: List[Dict[str, Any]],
    metrics: List[str],
    output_dir: Path,
) -> None:
    """
    Génère un fichier par métrique, tous les methods sur le même graphique.
    """
    for metric in metrics:
        plot_metric(rows, metric, output_dir / f"{metric}.png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parcourt les dossiers d'expériences et lit directement les ragas_summary.json."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Dossier racine, par ex: data/ragas_erag",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/plots_from_summary",
        help="Dossier de sortie pour csv/json/plots",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {root_dir}")

    rows = collect_summaries(root_dir)

    if not rows:
        raise RuntimeError("Aucun summary valide trouvé.")

    rows = sorted(rows, key=lambda x: (x["method"], x["k"]))

    save_json(output_dir / "all_summaries.json", rows)
    save_csv(output_dir / "all_summaries.csv", rows)

    metrics_to_plot = [
        "chunk_relevance_ndcg_at_k",
        "chunk_relevance_precision_at_k",
        "answer_relevancy",
        "context_relevance",
        "answer_correctness",
        "faithfulness",
        "llm_correct_or_not",
    ]

    plot_all_metrics_across_methods(
        rows=rows,
        metrics=metrics_to_plot,
        output_dir=output_dir,
    )

    print("\nRuns trouvés :")
    for row in rows:
        print(
            f"  {row['folder']} | method={row['method']} | k={row['k']} | "
            f"NDCG={row.get('chunk_relevance_ndcg_at_k', 'NA')}"
        )

    print(f"\nFichiers écrits dans : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())