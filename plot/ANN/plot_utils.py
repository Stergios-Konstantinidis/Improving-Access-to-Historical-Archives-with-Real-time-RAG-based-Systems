# src/plot_utils.py
import os
import matplotlib.ticker as ticker


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


import matplotlib.ticker as ticker

import matplotlib.ticker as ticker

def _apply_log_qps_axis(ax):
    ax.set_yscale("log")

    # --- MAJOR ticks: 10^k ---
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    # --- MINOR ticks: 2–9 × 10^k ---
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=range(2, 10))
    )

    # Pas de labels pour les minor ticks
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # --- Grid ---
    ax.grid(True, which="major", axis="y",
            linestyle="--", linewidth=0.7, alpha=0.8)

    ax.grid(True, which="minor", axis="y",
            linestyle=":", linewidth=0.4, alpha=0.5)

    ax.tick_params(axis="y", which="major", labelsize=9)



def plot_hnsw_tradeoff(rows, out_png: str, title: str):
    """
    rows: list of tuples saved by run_benchmarks (only HNSW rows are used)
    Groups by (M, ef_construction). Points along curve are different ef_search.
    """
    h = [r for r in rows if r[0] == "HNSW"]
    if not h:
        return

    recs = []
    for r in h:
        params = ast.literal_eval(r[1])
        query_ms = float(r[3])
        recall = float(r[4])
        qps = 1000.0 / max(query_ms, 1e-12)
        recs.append({
            "M": int(params["M"]),
            "efC": int(params["ef_construction"]),
            "efS": int(params["ef_search"]),
            "recall": recall,
            "qps": qps,
        })

    groups = {}
    for d in recs:
        key = (d["M"], d["efC"])
        groups.setdefault(key, []).append(d)

    plt.figure(figsize=(9, 6), dpi=150)
    ax = plt.gca()

    for (M, efC), pts in sorted(groups.items()):
        pts = sorted(pts, key=lambda x: x["recall"])
        x = [p["recall"] for p in pts]
        y = [p["qps"] for p in pts]
        ax.plot(x, y, marker="o", linewidth=1, markersize=3, label=f"M={M}, efC={efC}")

    _apply_log_qps_axis(ax)
    ax.set_xlabel("Recall@k")
    ax.set_ylabel("Queries per second (QPS)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.text(
        0.02,
        0.02,
        "Each curve: fixed (M, efConstruction)\nPoints along curve: increasing efSearch",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax.legend(fontsize=8)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_scatter_tradeoff(rows, method_name: str, out_png: str, title: str, max_legend: int = 12):
    """
    Generic scatter plot: each point = one parameter setting.
    Useful for methods where there's no canonical "curve" parameter.
    """
    m = [r for r in rows if r[0] == method_name]
    if not m:
        return

    pts = []
    for r in m:
        params_str = r[1]
        query_ms = float(r[3])
        recall = float(r[4])
        qps = 1000.0 / max(query_ms, 1e-12)
        pts.append((recall, qps, params_str))

    # Sort by recall for nicer look
    pts.sort(key=lambda x: x[0])

    plt.figure(figsize=(9, 6), dpi=150)
    ax = plt.gca()

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    ax.scatter(x, y, s=20)

    _apply_log_qps_axis(ax)
    ax.set_xlabel("Recall@k")
    ax.set_ylabel("Queries per second (QPS)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # Put a compact legend with only a few param strings (optional)
    # Otherwise legend becomes unreadable
    show = pts[:max_legend]
    if show:
        legend_lines = ["Example params (first few points):"]
        for _, _, ps in show:
            legend_lines.append(ps)
        ax.text(
            0.02,
            0.02,
            "\n".join(legend_lines),
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def read_rows_from_csv(csv_path: str):
    """
    Read rows from a results CSV produced by run_benchmarks.py.
    Returns the same tuple-rows format used by plotting functions.
    """
    import csv
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)  # skip header
        for row in r:
            # Keep types simple; downstream parses params and casts numeric fields.
            # row columns match HEADER in run_benchmarks.py
            rows.append((
                row[0],              # method
                row[1],              # params (string)
                float(row[2]),       # build_seconds
                float(row[3]),       # query_ms_per_q_median
                float(row[4]),       # recall_at_k
                float(row[5]),       # FDR
                float(row[6]),       # rss_before_build_mb
                float(row[7]),       # rss_after_build_mb
                float(row[8]),       # rss_build_delta_mb
                float(row[9]),       # rss_after_queries_mb
                int(row[10]),        # warmup_runs
                int(row[11]),        # timed_repeats
                int(row[12]),        # n_queries
                int(row[13]),        # n_base
                int(row[14]),        # dim
                int(row[15]),        # k
            ))
    return rows


def plot_from_latest_csvs(out_dir: str, method: str = "all"):
    """
    Convenience function: reads *_latest.csv in out_dir and writes plots into out_dir/plots.
    method: "hnsw" | "pynndescent" | "annoy" | "all"
    """
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    plots_dir = os.path.join(out_dir, "plots")

    def exists(p): return os.path.exists(p)

    if method in ["hnsw", "all"]:
        csv_path = os.path.join(out_dir, "results_HNSW_latest.csv")
        if exists(csv_path):
            rows = read_rows_from_csv(csv_path)
            out_png = os.path.join(plots_dir, f"HNSW_tradeoff_latest_{ts}.png")
            plot_hnsw_tradeoff(rows, out_png, title=f"HNSW Recall vs QPS (latest, {ts})")

    if method in ["pynndescent", "all"]:
        csv_path = os.path.join(out_dir, "results_PyNNDescent_latest.csv")
        if exists(csv_path):
            rows = read_rows_from_csv(csv_path)
            out_png = os.path.join(plots_dir, f"PyNNDescent_tradeoff_latest_{ts}.png")
            plot_scatter_tradeoff(rows, "PyNNDescent", out_png, title=f"PyNNDescent Recall vs QPS (latest, {ts})")

    if method in ["annoy", "all"]:
        csv_path = os.path.join(out_dir, "results_Annoy_latest.csv")
        if exists(csv_path):
            rows = read_rows_from_csv(csv_path)
            out_png = os.path.join(plots_dir, f"Annoy_tradeoff_latest_{ts}.png")
            plot_scatter_tradeoff(rows, "Annoy", out_png, title=f"Annoy Recall vs QPS (latest, {ts})")

    # Optional: if you want one plot for ALL together, you can add it later.

def plot_pynndescent_tradeoff(rows, out_png: str, title: str):
    """
    PyNNDescent plot in 'curve' style:
      curve = (n_neighbors, pruning_degree_multiplier, diversify_prob, metric)
      points along curve = increasing max_candidates
    """
    import ast
    import matplotlib.pyplot as plt

    m = [r for r in rows if r[0] == "PyNNDescent"]
    if not m:
        return

    recs = []
    for r in m:
        params = ast.literal_eval(r[1])
        query_ms = float(r[3])
        recall = float(r[4])
        qps = 1000.0 / max(query_ms, 1e-12)

        recs.append({
            "metric": params.get("metric", "cosine"),
            "n_neighbors": int(params.get("n_neighbors", 30)),
            "max_candidates": int(params.get("max_candidates", 60)),
            "pruning_degree_multiplier": float(params.get("pruning_degree_multiplier", 2.0)),
            "diversify_prob": float(params.get("diversify_prob", 1.0)),
            "recall": recall,
            "qps": qps,
        })

    # group: fixed build/structure params; vary max_candidates as "query effort"
    groups = {}
    for d in recs:
        key = (
            d["metric"],
            d["n_neighbors"],
            d["pruning_degree_multiplier"],
            d["diversify_prob"],
        )
        groups.setdefault(key, []).append(d)

    plt.figure(figsize=(9, 6), dpi=150)
    ax = plt.gca()

    # Sort groups for stable legend
    for key, pts in sorted(groups.items(), key=lambda x: x[0]):
        metric, nn, it, pdm, dp = key
        # points ordered by increasing max_candidates
        pts = sorted(pts, key=lambda x: x["max_candidates"])
        x = [p["recall"] for p in pts]
        y = [p["qps"] for p in pts]
        ax.plot(x, y, marker="o", linewidth=1, markersize=3,
                label=f"nn={nn}, it={it}, pdm={pdm}, dp={dp}")

    _apply_log_qps_axis(ax)
    ax.set_xlabel("Recall@k")
    ax.set_ylabel("Queries per second (QPS)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.text(
        0.02,
        0.02,
        "Each curve: fixed (n_neighbors, pruning_degree_multiplier, diversify_prob)\n"
        "Points along curve: increasing max_candidates",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax.legend(fontsize=8)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# --- ADD BELOW in src/plot_utils.py ---
import ast
import numpy as np
import matplotlib.pyplot as plt


def plot_pynndescent_experiment(rows, out_png: str, title: str, curve_by: str, point_by: str):
    """
    rows: tuples from results list (same as CSV rows in run_benchmarks)
    curve_by: one param name used to create multiple curves (e.g., "n_neighbors")
    point_by: one param name used for ordering points along each curve (e.g., "max_candidates")
    Other params are assumed fixed inside the experiment.
    """
    m = [r for r in rows if r[0] == "PyNNDescent"]
    if not m:
        return

    recs = []
    for r in m:
        params = ast.literal_eval(r[1])
        query_ms = float(r[3])
        recall = float(r[4])
        qps = 1000.0 / max(query_ms, 1e-12)

        recs.append({
            "params": params,
            "curve": params.get(curve_by),
            "point": params.get(point_by),
            "recall": recall,
            "qps": qps,
        })

    # group by curve key
    groups = {}
    for d in recs:
        groups.setdefault(d["curve"], []).append(d)

    plt.figure(figsize=(9, 6), dpi=150)
    ax = plt.gca()

    for curve_val in sorted(groups.keys()):
        pts = groups[curve_val]
        # order by point_by (e.g., max_candidates increasing)
        pts = sorted(pts, key=lambda x: x["point"])
        x = [p["recall"] for p in pts]
        y = [p["qps"] for p in pts]
        ax.plot(x, y, marker="o", linewidth=1, markersize=3, label=f"{curve_by}={curve_val}")

    # log QPS axis formatting (reuse your helper)
    _apply_log_qps_axis(ax)
    ax.set_ylim(bottom=pow(10, 4))
    ax.set_xlabel("Recall@k")
    ax.set_ylabel("Queries per second (QPS)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.text(
        0.02, 0.02,
        f"Curves: {curve_by}\nPoints: increasing {point_by}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax.legend(fontsize=8)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    # --- ADD in src/plot_utils.py ---

    from typing import Dict, List, Tuple, Optional

    def _pynnd_exp_curve_point(exp: str) -> Tuple[str, str]:
        """
        Returns (curve_by, point_by) mapping for each PyNNDescent experiment.
        """
        if exp == "epsilon":
            return "metric", "epsilon"
        if exp == "n_neighbors":
            return "metric", "n_neighbors"
        if exp == "max_candidates":
            return "metric", "max_candidates"
        if exp == "pruning":
            return "metric", "pruning_degree_multiplier"
        if exp == "diversify":
            return "metric", "diversify_prob"
        # fallback
        return "metric", exp

    def plot_pynndescent_latest_overlay(
            out_dir: str,
            out_png: str,
            title: str,
            exps: Optional[List[str]] = None,
            y_min_qps: Optional[float] = None,
            max_legend: int = 30,
    ):
        """
        Superpose sur un seul graphique toutes les courbes issues des fichiers:
          results_PyNNDescent_{exp}_latest.csv

        - Chaque expérience 'exp' produit plusieurs courbes (curve_by, généralement metric)
          et des points ordonnés par point_by.
        - Label = "{exp} | {curve_by}={curve_val}"

        Args:
            out_dir: dossier où se trouvent les CSV latest
            out_png: chemin du PNG final
            title: titre du plot
            exps: liste d’expériences (default: toutes celles connues)
            y_min_qps: si set, force une borne inférieure sur l’axe QPS
            max_legend: limite de labels affichés (sinon la légende devient illisible)
        """
        import os
        import ast
        import matplotlib.pyplot as plt

        if exps is None:
            exps = ["epsilon", "n_neighbors", "max_candidates", "pruning", "diversify"]

        # Markers différents par expérience (facilite la lecture)
        exp_markers = {
            "epsilon": "o",
            "n_neighbors": "s",
            "max_candidates": "^",
            "pruning": "D",
            "diversify": "v",
        }

        plt.figure(figsize=(10, 7), dpi=150)
        ax = plt.gca()

        # collect labels to potentially truncate legend
        plotted_labels = []

        any_data = False

        for exp in exps:
            csv_path = os.path.join(out_dir, f"results_PyNNDescent_{exp}_latest.csv")
            if not os.path.exists(csv_path):
                continue

            rows = read_rows_from_csv(csv_path)
            m = [r for r in rows if r[0] == "PyNNDescent"]
            if not m:
                continue

            any_data = True
            curve_by, point_by = _pynnd_exp_curve_point(exp)

            # parse into records
            recs = []
            for r in m:
                params = ast.literal_eval(r[1])
                query_ms = float(r[3])
                recall = float(r[4])
                qps = 1000.0 / max(query_ms, 1e-12)

                recs.append({
                    "curve": params.get(curve_by, None),
                    "point": params.get(point_by, None),
                    "recall": recall,
                    "qps": qps,
                })

            # group by curve value
            groups: Dict[object, List[dict]] = {}
            for d in recs:
                groups.setdefault(d["curve"], []).append(d)

            marker = exp_markers.get(exp, "o")

            # plot each curve
            for curve_val in sorted(groups.keys(), key=lambda x: str(x)):
                pts = groups[curve_val]

                # order by point_by when possible
                def _safe_sort_key(x):
                    v = x["point"]
                    # None last
                    return (v is None, v)

                pts = sorted(pts, key=_safe_sort_key)
                x = [p["recall"] for p in pts]
                y = [p["qps"] for p in pts]

                label = f"{exp} | {curve_by}={curve_val}"
                ax.plot(x, y, marker=marker, linewidth=1, markersize=3, label=label)
                plotted_labels.append(label)

        if not any_data:
            plt.close()
            return

        _apply_log_qps_axis(ax)
        if y_min_qps is not None:
            ax.set_ylim(bottom=y_min_qps)

        ax.set_xlabel("Recall@k")
        ax.set_ylabel("Queries per second (QPS)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        ax.text(
            0.02, 0.02,
            "Overlay of PyNNDescent latest experiments\nEach curve = one (exp, curve_by value)",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

        # Legend control: keep first max_legend entries only (readable)
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > max_legend:
            handles = handles[:max_legend]
            labels = labels[:max_legend]
            labels.append(f"... (+{len(ax.get_legend_handles_labels()[1]) - max_legend} more)")
            handles.append(handles[-1])  # duplicate last handle just to show the "...", ok-ish

        ax.legend(handles, labels, fontsize=8, loc="best")

        plt.tight_layout()
        ensure_dir(os.path.dirname(out_png))
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()

# --- ADD in src/plot_utils.py ---

from typing import Dict, List, Tuple, Optional

def _pynnd_exp_curve_point(exp: str) -> Tuple[str, str]:
    """
    Returns (curve_by, point_by) mapping for each PyNNDescent experiment.
    """
    if exp == "epsilon":
        return "metric", "epsilon"
    if exp == "n_neighbors":
        return "metric", "n_neighbors"
    if exp == "max_candidates":
        return "metric", "max_candidates"
    if exp == "pruning":
        return "metric", "pruning_degree_multiplier"
    if exp == "diversify":
        return "metric", "diversify_prob"
    # fallback
    return "metric", exp


def plot_pynndescent_all_latest_one_plot(
    out_dir: str,
    out_png: str,
    title: str,
    exps=None,
    y_min_qps=None,
    max_legend: int = 40,
):
    """
    Un SEUL graphique contenant TOUTES les courbes PyNNDescent issues des fichiers:
      results_PyNNDescent_{exp}_latest.csv

    Chaque fichier exp définit:
      - curve_by : quel param fait "une courbe"
      - point_by : quel param ordonne les points le long de la courbe
    """
    import os
    import ast
    import matplotlib.pyplot as plt

    if exps is None:
        exps = ["epsilon", "n_neighbors", "max_candidates", "pruning", "diversify"]

    def exp_curve_point(exp: str):
        if exp == "epsilon":
            return "metric", "epsilon"
        if exp == "n_neighbors":
            return "metric", "n_neighbors"
        if exp == "max_candidates":
            return "metric", "max_candidates"
        if exp == "pruning":
            return "metric", "pruning_degree_multiplier"
        if exp == "diversify":
            return "metric", "diversify_prob"
        return "metric", exp

    # marqueurs différents par exp (lisible en overlay)
    exp_markers = {
        "epsilon": "o",
        "n_neighbors": "s",
        "max_candidates": "^",
        "pruning": "D",
        "diversify": "v",
    }

    plt.figure(figsize=(10, 7), dpi=150)
    ax = plt.gca()

    any_data = False

    for exp in exps:
        csv_path = os.path.join(out_dir, f"results_PyNNDescent_{exp}_latest.csv")
        if not os.path.exists(csv_path):
            continue

        rows = read_rows_from_csv(csv_path)
        m = [r for r in rows if r[0] == "PyNNDescent"]
        if not m:
            continue

        any_data = True
        curve_by, point_by = exp_curve_point(exp)
        marker = exp_markers.get(exp, "o")

        # parse records
        recs = []
        for r in m:
            params = ast.literal_eval(r[1])
            query_ms = float(r[3])
            recall = float(r[4])
            qps = 1000.0 / max(query_ms, 1e-12)

            recs.append({
                "curve": params.get(curve_by, None),
                "point": params.get(point_by, None),
                "recall": recall,
                "qps": qps,
            })

        # group by curve
        groups = {}
        for d in recs:
            groups.setdefault(d["curve"], []).append(d)

        for curve_val in sorted(groups.keys(), key=lambda x: str(x)):
            pts = groups[curve_val]

            # order along curve by point_by (None last)
            pts = sorted(pts, key=lambda x: (x["point"] is None, x["point"]))

            x = [p["recall"] for p in pts]
            y = [p["qps"] for p in pts]

            # label unique par (exp, curve_val)
            label = f"{exp} | {curve_by}={curve_val}"
            ax.plot(x, y, marker=marker, linewidth=1, markersize=3, label=label)

    if not any_data:
        plt.close()
        return

    _apply_log_qps_axis(ax)
    if y_min_qps is not None:
        ax.set_ylim(bottom=y_min_qps)

    ax.set_xlabel("Recall@k")
    ax.set_ylabel("Queries per second (QPS)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # limiter la légende si trop de courbes
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > max_legend:
        handles = handles[:max_legend]
        labels = labels[:max_legend]
        ax.legend(handles, labels, fontsize=8, loc="best")
    else:
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_faiss_ivf_tradeoff(rows, out_png: str, title: str,
                            method_name: str = "FAISS_IVF_FLAT",
                            curve_by: str = "nlist",
                            point_by: str = "nprobe",
                            y_min_qps: float | None = None,
                            max_legend: int = 20):
    """
    FAISS IVF 'curve' plot:
      - each curve: fixed `curve_by` (default nlist)
      - points along curve: increasing `point_by` (default nprobe)

    rows: tuples from read_rows_from_csv()
    """
    import ast
    import matplotlib.pyplot as plt

    m = [r for r in rows if r[0] == method_name]
    if not m:
        return

    recs = []
    for r in m:
        params = ast.literal_eval(r[1])
        query_ms = float(r[3])
        recall = float(r[4])
        qps = 1000.0 / max(query_ms, 1e-12)

        recs.append({
            "curve": params.get(curve_by, None),
            "point": params.get(point_by, None),
            "recall": recall,
            "qps": qps,
        })

    # group by curve key (e.g., nlist)
    groups = {}
    for d in recs:
        groups.setdefault(d["curve"], []).append(d)

    plt.figure(figsize=(9, 6), dpi=150)
    ax = plt.gca()

    # stable ordering of curves
    for curve_val in sorted(groups.keys(), key=lambda x: (x is None, x)):
        pts = groups[curve_val]

        # order points by nprobe (None last)
        pts = sorted(pts, key=lambda x: (x["point"] is None, x["point"]))

        x = [p["recall"] for p in pts]
        y = [p["qps"] for p in pts]

        ax.plot(x, y, marker="o", linewidth=1, markersize=3, label=f"{curve_by}={curve_val}")

    _apply_log_qps_axis(ax)
    if y_min_qps is not None:
        ax.set_ylim(bottom=y_min_qps)

    ax.set_xlabel("Recall@k")
    ax.set_ylabel("Queries per second (QPS)")
    ax.set_title(title)

    # grille lisible: majeures + mineures (si ton _apply_log_qps_axis garde les minor ticks)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.text(
        0.02, 0.02,
        f"Each curve: fixed {curve_by}\nPoints along curve: increasing {point_by}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    # limiter légende si trop de nlist
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > max_legend:
        handles = handles[:max_legend]
        labels = labels[:max_legend]
    ax.legend(handles, labels, fontsize=8)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_faiss_ivfpq_tradeoff(
    rows,
    out_png: str,
    title: str,
    method_name: str = "FAISS_IVF_PQ",
    curve_by=("nlist", "m", "nbits"),
    point_by: str = "nprobe",
    y_min_qps: float | None = None,
    max_legend: int = 20,
):
    """
    FAISS IVFPQ curve plot:
      - each curve: fixed (nlist, m, nbits) by default
      - points along curve: increasing nprobe

    rows: tuples from read_rows_from_csv()
    """
    import ast
    import matplotlib.pyplot as plt

    mrows = [r for r in rows if r[0] == method_name]
    if not mrows:
        return

    recs = []
    for r in mrows:
        params = ast.literal_eval(r[1])
        query_ms = float(r[3])
        recall = float(r[4])
        qps = 1000.0 / max(query_ms, 1e-12)

        # curve key: (nlist, m, nbits)
        curve_key = tuple(params.get(k, None) for k in curve_by)
        point_val = params.get(point_by, None)

        recs.append({
            "curve": curve_key,
            "point": point_val,
            "recall": recall,
            "qps": qps,
        })

    # group by curve
    groups = {}
    for d in recs:
        groups.setdefault(d["curve"], []).append(d)

    plt.figure(figsize=(9, 6), dpi=150)
    ax = plt.gca()

    def _curve_sort_key(k):
        # stable sorting even with None
        return tuple((v is None, v) for v in k)

    for curve_key in sorted(groups.keys(), key=_curve_sort_key):
        pts = groups[curve_key]
        pts = sorted(pts, key=lambda x: (x["point"] is None, x["point"]))

        x = [p["recall"] for p in pts]
        y = [p["qps"] for p in pts]

        # label readable
        label = ", ".join(f"{name}={val}" for name, val in zip(curve_by, curve_key))
        ax.plot(x, y, marker="o", linewidth=1, markersize=3, label=label)

    _apply_log_qps_axis(ax)
    if y_min_qps is not None:
        ax.set_ylim(bottom=y_min_qps)

    ax.set_xlabel("Recall@k")
    ax.set_ylabel("Queries per second (QPS)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.text(
        0.02, 0.02,
        f"Each curve: fixed {curve_by}\nPoints along curve: increasing {point_by}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    # limit legend
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > max_legend:
        handles = handles[:max_legend]
        labels = labels[:max_legend]
    ax.legend(handles, labels, fontsize=8)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()