# src/plot_latest.py
import os
import argparse
from datetime import datetime

from ..utils.config import CFG

from plot_utils import (
    read_rows_from_csv,
    plot_hnsw_tradeoff,
    plot_pynndescent_experiment,
    plot_scatter_tradeoff,
    plot_pynndescent_all_latest_one_plot,
    plot_faiss_ivf_tradeoff,
    plot_faiss_ivfpq_tradeoff,
)

PYNND_EXPS = ["epsilon", "n_neighbors", "max_candidates", "pruning", "diversify"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["hnsw", "pynndescent", "annoy", "faiss", "all"], default="all")
    p.add_argument(
        "--experiment",
        default="all",
        choices=["all"] + PYNND_EXPS,
        help="For PyNNDescent: which latest experiment plot to generate (default: all).",
    )
    p.add_argument(
        "--pynndescent_all_in_one",
        action="store_true",
        help="Generate ONE plot overlaying all PyNNDescent latest experiments.",
    )
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    plots_dir = os.path.join(CFG.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- HNSW ---
    if args.method in ["hnsw", "all"]:
        csv_path = os.path.join(CFG.out_dir, "results_HNSW_latest.csv")
        if os.path.exists(csv_path):
            rows = read_rows_from_csv(csv_path)
            out_png = os.path.join(plots_dir, f"HNSW_latest_{ts}.png")
            plot_hnsw_tradeoff(rows, out_png, title=f"HNSW Recall vs QPS (latest, {ts})")
            print("Saved plot:", out_png)
        else:
            print("[skip] Missing:", csv_path)

    # --- PyNNDescent ---

    if args.method in ["pynndescent", "all"] and args.pynndescent_all_in_one:
        out_png = os.path.join(plots_dir, f"PyNNDescent_ALL_latest_{ts}.png")
        plot_pynndescent_all_latest_one_plot(
            out_dir=CFG.out_dir,
            out_png=out_png,
            title=f"PyNNDescent — ALL latest experiments (Recall vs QPS, {ts})",
            exps=PYNND_EXPS,
            y_min_qps=None,
            max_legend=40,
        )
        print("Saved plot:", out_png)
        return

    if args.method in ["pynndescent", "all"]:
        exps = PYNND_EXPS if args.experiment == "all" else [args.experiment]
        for exp in exps:
            csv_path = os.path.join(CFG.out_dir, f"results_PyNNDescent_{exp}_latest.csv")
            if not os.path.exists(csv_path):
                print("[skip] Missing:", csv_path)
                continue

            rows = read_rows_from_csv(csv_path)

            # define curve_by/point_by mapping (like your experiments)
            if exp == "epsilon":
                curve_by, point_by = "metric", "epsilon"
            elif exp == "n_neighbors":
                curve_by, point_by = "metric", "n_neighbors"
            elif exp == "max_candidates":
                curve_by, point_by = "metric", "max_candidates"
            elif exp == "pruning":
                curve_by, point_by = "metric", "pruning_degree_multiplier"
            elif exp == "diversify":
                curve_by, point_by = "metric", "diversify_prob"
            else:
                curve_by, point_by = "metric", exp

            out_png = os.path.join(plots_dir, f"PyNNDescent_{exp}_latest_{ts}.png")
            plot_pynndescent_experiment(
                rows,
                out_png=out_png,
                title=f"PyNNDescent {exp} — Recall vs QPS (latest, {ts})",
                curve_by=curve_by,
                point_by=point_by,
            )
            print("Saved plot:", out_png)

    # --- FAISS (IVF Flat + IVF PQ) ---
    if args.method in ["faiss", "all"]:
        csv_path = os.path.join(CFG.out_dir, "results_FAISS_latest.csv")
        if os.path.exists(csv_path):
            rows = read_rows_from_csv(csv_path)

            # IVF Flat
            out_png = os.path.join(plots_dir, f"FAISS_IVF_FLAT_latest_{ts}.png")
            plot_faiss_ivf_tradeoff(
                rows,
                out_png=out_png,
                title=f"FAISS IVFFlat — Recall vs QPS (latest, {ts})",
                method_name="FAISS_IVF_FLAT",
                curve_by="nlist",
                point_by="nprobe",
            )
            print("Saved plot:", out_png)

            # IVFPQ
            out_png = os.path.join(plots_dir, f"FAISS_IVF_PQ_latest_{ts}.png")
            plot_faiss_ivfpq_tradeoff(
                rows,
                out_png=out_png,
                title=f"FAISS IVFPQ — Recall vs QPS (latest, {ts})",
                method_name="FAISS_IVF_PQ",
                curve_by=("nlist", "m", "nbits"),  # courbe = (nlist, m, nbits)
                point_by="nprobe",
            )
            print("Saved plot:", out_png)

        else:
            print("[skip] Missing:", csv_path)

    # --- Annoy ---
    if args.method in ["annoy", "all"]:
        csv_path = os.path.join(CFG.out_dir, "results_Annoy_latest.csv")
        if os.path.exists(csv_path):
            rows = read_rows_from_csv(csv_path)
            out_png = os.path.join(plots_dir, f"Annoy_latest_{ts}.png")
            plot_scatter_tradeoff(rows, "Annoy", out_png, title=f"Annoy Recall vs QPS (latest, {ts})")
            print("Saved plot:", out_png)
        else:
            print("[skip] Missing:", csv_path)

    print("Done. Check:", f"{CFG.out_dir}/plots/")


if __name__ == "__main__":
    main()