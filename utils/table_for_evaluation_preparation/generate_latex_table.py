#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un tableau LaTeX depuis un dossier d'expériences RAG, avec :
- moyenne
- intervalle de confiance à 95%
- test t apparié (paired t-test)
- marqueurs de significativité (*, **, ***)
- soulignement d'une méthode vs une baseline

Structure attendue :
root/
├── generations_k5_answered/
│   ├── ragas_summary.json
│   └── ragas_metrics.json
├── generations_k10_answered/
│   ├── ragas_summary.json
│   └── ragas_metrics.json
├── generations_reranker_k5_answered/
│   ├── ragas_summary.json
│   └── ragas_metrics.json
└── ...

Usage :
    python generate_latex_table.py /chemin/vers/ragas_erag -o table_results.tex

Exemple :
    python generate_latex_table.py ./ragas_erag -o table_results.tex

Remarques :
- L'alignement des tests appariés se fait via question_id.
- Si une expérience manque pour un k, la ligne reste vide.
- Par défaut :
    * "Semantic" = dossier du type generations_k5_answered
    * "LLM corrected + reranker" = dossier contenant "reranker"
    * "LLM corrected" = dossier contenant "corrected"
    * "BM25s" = dossier contenant "bm25"
- Tu peux forcer le mapping exact via EXACT_FOLDER_METHOD_MAP.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import t as student_t


# =====================================================================
# CONFIGURATION
# =====================================================================

# Ordre des méthodes dans le tableau
METHOD_ORDER = [
    "BM25s",
    "Semantic",
    "LLM corrected",
    "LLM corrected + reranker",
]

# Si tu veux forcer certains noms de dossiers
# Exemple :
# EXACT_FOLDER_METHOD_MAP = {
#     "generations_k5_answered": "Semantic",
#     "generations_corrected_k5_answered": "LLM corrected",
#     "generations_reranker_k5_answered": "LLM corrected + reranker",
#     "bm25_k5_answered": "BM25s",
# }
EXACT_FOLDER_METHOD_MAP: Dict[str, str] = {}

# Méthode à souligner si meilleure que la baseline à p < 0.05
UNDERLINE_METHOD = "LLM corrected"
UNDERLINE_BASELINE = "Semantic"

# Test t apparié :
# "greater" = test unilatéral : amélioration stricte
# "two-sided" = test bilatéral
PAIRED_TEST_ALTERNATIVE = "two-sided"

# Faut-il afficher les IC95 dans les cellules ?
SHOW_CONFIDENCE_INTERVALS = True

# Placeholder si une ligne d'expérience est absente
# Par ex. "running" si tu veux reproduire ce style
MISSING_ROW_FIRST_METRIC_TEXT = ""

# Comparaison du temps : en général plus petit = meilleur
TIME_LOWER_IS_BETTER = True

# Tolérance pour égalité numérique
EPS = 1e-12

# Colonnes du tableau
METRIC_SPECS = [
    {
        "key": "answer_relevancy",
        "label": "Answer Relevancy",
        "kind": "percent",
        "higher_is_better": True,
        "allow_significance": True,
        "allow_underline": True,
    },
    {
        "key": "context_relevance",
        "label": "Context Relevance",
        "kind": "percent",
        "higher_is_better": True,
        "allow_significance": True,
        "allow_underline": True,
    },
    {
        "key": "answer_correctness",
        "label": "Answer Correctness",
        "kind": "percent",
        "higher_is_better": True,
        "allow_significance": True,
        "allow_underline": True,
    },
    {
        "key": "chunk_relevance_ndcg_at_k",
        "label": "NDCG@k",
        "kind": "percent",
        "higher_is_better": True,
        "allow_significance": True,
        "allow_underline": True,
    },
    {
        "key": "avg_time_seconds",
        "label": "Avg. t/q(s)",
        "kind": "seconds",
        "higher_is_better": not TIME_LOWER_IS_BETTER,
        "allow_significance": False,
        "allow_underline": False,
    },
]


# =====================================================================
# STRUCTURES
# =====================================================================

@dataclass
class Experiment:
    k: int
    method: str
    folder_name: str
    summary: dict
    sample_metrics: Dict[str, Dict[str, float]]  # metric_key -> {question_id: value}


# =====================================================================
# OUTILS
# =====================================================================

def extract_k(folder_name: str) -> Optional[int]:
    m = re.search(r"(?:^|[_-])k(\d+)(?:[_-]|$)", folder_name)
    if m:
        return int(m.group(1))
    return None


def infer_method(folder_name: str) -> str:
    if folder_name in EXACT_FOLDER_METHOD_MAP:
        return EXACT_FOLDER_METHOD_MAP[folder_name]

    name = folder_name.lower()

    if "bm25" in name:
        return "BM25s"

    if "reranker" in name:
        return "LLM corrected + reranker"

    if "corrected" in name or "llm_corrected" in name or "correction" in name:
        return "LLM corrected"

    # cas typique : generations_k5_answered
    if name.startswith("generations_k") and "reranker" not in name:
        return "Semantic"

    return folder_name


def load_json(path: Path) -> Optional[object]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def approx_equal(a: Optional[float], b: Optional[float], eps: float = EPS) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= eps


def significance_marker(pvalue: Optional[float]) -> str:
    if pvalue is None:
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return ""


# =====================================================================
# CHARGEMENT DES EXPÉRIENCES
# =====================================================================

def get_display_mean(exp: Experiment, metric_key: str) -> Optional[float]:
    """
    Moyenne affichée dans le tableau.
    On privilégie ragas_summary.json pour garder les valeurs officielles.
    """
    if metric_key == "avg_time_seconds":
        return compute_summary_avg_time_seconds(exp.summary)

    raw = exp.summary.get(metric_key)
    if is_number(raw):
        return float(raw)

    # fallback si jamais la summary est absente/incomplète
    values_by_qid = get_metric_values(exp, metric_key)
    if values_by_qid:
        arr = np.asarray(list(values_by_qid.values()), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            return float(np.mean(arr))

    return None


def get_metric_ci(exp: Experiment, metric_key: str) -> Optional[float]:
    """
    Demi-largeur de l'IC95 calculée depuis ragas_metrics.json.
    """
    values_by_qid = get_metric_values(exp, metric_key)
    if not values_by_qid:
        return None

    arr = np.asarray(list(values_by_qid.values()), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None

    sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    tcrit = float(student_t.ppf(0.975, df=arr.size - 1))
    return tcrit * sem

def compute_summary_avg_time_seconds(summary: dict) -> Optional[float]:
    avg_ms = summary.get("pipeline_avg_time_ms")
    if is_number(avg_ms):
        return float(avg_ms) / 1000.0

    total_s = summary.get("pipeline_total_time_seconds")
    n = summary.get("num_samples")
    if is_number(total_s) and is_number(n) and float(n) > 0:
        return float(total_s) / float(n)

    return None


def parse_experiment(folder: Path) -> Optional[Experiment]:
    if not folder.is_dir():
        return None

    k = extract_k(folder.name)
    if k is None:
        return None

    summary_path = folder / "ragas_summary.json"
    metrics_path = folder / "ragas_metrics.json"

    if not summary_path.exists() or not metrics_path.exists():
        return None

    summary = load_json(summary_path)
    metrics_rows = load_json(metrics_path)

    if not isinstance(summary, dict) or not isinstance(metrics_rows, list):
        return None

    method = infer_method(folder.name)

    sample_metrics: Dict[str, Dict[str, float]] = {
        spec["key"]: {} for spec in METRIC_SPECS
    }

    # Metrics par question
    for row in metrics_rows:
        if not isinstance(row, dict):
            continue

        qid = row.get("question_id")
        if not qid:
            continue

        for spec in METRIC_SPECS:
            key = spec["key"]

            if key == "avg_time_seconds":
                latency_ms = row.get("latency_ms")
                if is_number(latency_ms):
                    sample_metrics[key][qid] = float(latency_ms) / 1000.0
                continue

            value = row.get(key)
            if is_number(value):
                sample_metrics[key][qid] = float(value)

    # Si pas de latence par sample, on garde vide et on utilisera le summary en fallback
    if not sample_metrics["avg_time_seconds"]:
        summary_avg_t = compute_summary_avg_time_seconds(summary)
        if summary_avg_t is not None:
            # pas de qid, donc pas de CI ni de t-test, seulement une valeur de fallback
            sample_metrics["avg_time_seconds"] = {}

    return Experiment(
        k=k,
        method=method,
        folder_name=folder.name,
        summary=summary,
        sample_metrics=sample_metrics,
    )


def collect_experiments(root: Path) -> List[Experiment]:
    experiments = []
    for child in root.iterdir():
        exp = parse_experiment(child)
        if exp is not None:
            experiments.append(exp)
    return experiments


# =====================================================================
# STATISTIQUES
# =====================================================================

def get_metric_values(exp: Experiment, metric_key: str) -> Dict[str, float]:
    return exp.sample_metrics.get(metric_key, {})


def get_metric_mean_ci(exp: Experiment, metric_key: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Retourne (moyenne, demi-largeur IC95).
    Si on n'a pas les valeurs par sample pour le temps, fallback sur summary sans CI.
    """
    values_by_qid = get_metric_values(exp, metric_key)

    if values_by_qid:
        arr = np.asarray(list(values_by_qid.values()), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, None

        mean = float(np.mean(arr))
        if arr.size == 1:
            return mean, None

        sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
        tcrit = float(student_t.ppf(0.975, df=arr.size - 1))
        ci = tcrit * sem
        return mean, ci

    # fallback spécifique temps
    if metric_key == "avg_time_seconds":
        summary_avg = compute_summary_avg_time_seconds(exp.summary)
        if summary_avg is not None:
            return summary_avg, None

    # fallback summary pour les autres métriques si besoin
    raw = exp.summary.get(metric_key)
    if is_number(raw):
        return float(raw), None

    return None, None


def paired_ttest_pvalue(
    exp_a: Experiment,
    exp_b: Experiment,
    metric_key: str,
    alternative: str = PAIRED_TEST_ALTERNATIVE,
) -> Optional[float]:
    """
    Test t apparié sur les samples communs, alignés par question_id.
    alternative:
        - "greater" : H1 = mean(a - b) > 0
        - "less"    : H1 = mean(a - b) < 0
        - "two-sided"
    """
    a_map = get_metric_values(exp_a, metric_key)
    b_map = get_metric_values(exp_b, metric_key)

    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()))
    if len(common_ids) < 2:
        return None

    a = np.asarray([a_map[qid] for qid in common_ids], dtype=float)
    b = np.asarray([b_map[qid] for qid in common_ids], dtype=float)
    diff = a - b

    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n < 2:
        return None

    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))

    # cas dégénéré : toutes les différences identiques
    if sd_diff <= EPS:
        if alternative == "greater":
            return 0.0 if mean_diff > 0 else 1.0
        if alternative == "less":
            return 0.0 if mean_diff < 0 else 1.0
        return 0.0 if abs(mean_diff) > EPS else 1.0

    t_stat = mean_diff / (sd_diff / math.sqrt(n))
    df = n - 1

    if alternative == "greater":
        pvalue = 1.0 - float(student_t.cdf(t_stat, df))
    elif alternative == "less":
        pvalue = float(student_t.cdf(t_stat, df))
    elif alternative == "two-sided":
        cdf = float(student_t.cdf(t_stat, df))
        pvalue = 2.0 * min(cdf, 1.0 - cdf)
    else:
        raise ValueError(f"alternative invalide: {alternative}")

    return max(0.0, min(1.0, pvalue))


# =====================================================================
# AGRÉGATION / ANNOTATIONS
# =====================================================================

def group_by_k(experiments: List[Experiment]) -> Dict[int, Dict[str, Experiment]]:
    grouped: Dict[int, Dict[str, Experiment]] = {}
    for exp in experiments:
        grouped.setdefault(exp.k, {})
        grouped[exp.k][exp.method] = exp
    return grouped


def is_better(a: float, b: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return a > b + EPS
    return a < b - EPS


def sort_metric_stats(
    metric_stats: List[Tuple[str, float]],
    higher_is_better: bool,
) -> List[Tuple[str, float]]:
    return sorted(metric_stats, key=lambda x: x[1], reverse=higher_is_better)


def compute_annotations_for_k(rows_for_k: Dict[str, Experiment]) -> Dict[Tuple[str, str], Dict[str, object]]:
    """
    Règles :
    - bold = meilleure méthode à ce k pour cette métrique
    - superscript = significativité de la ligne vs la ligne précédente
    - underline = LLM corrected significativement meilleur que Semantic (p < 0.05)
    """
    annotations: Dict[Tuple[str, str], Dict[str, object]] = {}

    for method in METHOD_ORDER:
        for spec in METRIC_SPECS:
            annotations[(method, spec["key"])] = {
                "bold": False,
                "underline": False,
                "superscript": "",
            }

    for spec in METRIC_SPECS:
        metric_key = spec["key"]
        higher_is_better = spec["higher_is_better"]

        # ---------- bold sur la meilleure valeur affichée ----------
        metric_stats: List[Tuple[str, float]] = []
        for method in METHOD_ORDER:
            exp = rows_for_k.get(method)
            if exp is None:
                continue

            mean = get_display_mean(exp, metric_key)
            if mean is not None:
                metric_stats.append((method, mean))

        if metric_stats:
            sorted_stats = sort_metric_stats(metric_stats, higher_is_better=higher_is_better)
            best_value = sorted_stats[0][1]
            best_methods = [m for m, v in sorted_stats if approx_equal(v, best_value)]
            for method in best_methods:
                annotations[(method, metric_key)]["bold"] = True

        # ---------- étoiles : chaque ligne vs la précédente ----------
        if spec["allow_significance"]:
            for idx in range(1, len(METHOD_ORDER)):
                prev_method = METHOD_ORDER[idx - 1]
                curr_method = METHOD_ORDER[idx]

                exp_prev = rows_for_k.get(prev_method)
                exp_curr = rows_for_k.get(curr_method)

                if exp_prev is None or exp_curr is None:
                    continue

                mean_prev = get_display_mean(exp_prev, metric_key)
                mean_curr = get_display_mean(exp_curr, metric_key)

                if mean_prev is None or mean_curr is None:
                    continue

                improved = is_better(mean_curr, mean_prev, higher_is_better=higher_is_better)
                if not improved:
                    continue

                pval = paired_ttest_pvalue(
                    exp_curr,
                    exp_prev,
                    metric_key,
                    alternative=PAIRED_TEST_ALTERNATIVE,
                )
                annotations[(curr_method, metric_key)]["superscript"] = significance_marker(pval)

        # ---------- soulignement : LLM corrected vs Semantic ----------
        if spec["allow_underline"]:
            exp_u = rows_for_k.get(UNDERLINE_METHOD)
            exp_b = rows_for_k.get(UNDERLINE_BASELINE)

            if exp_u is not None and exp_b is not None:
                mean_u = get_display_mean(exp_u, metric_key)
                mean_b = get_display_mean(exp_b, metric_key)

                if mean_u is not None and mean_b is not None:
                    improved = is_better(mean_u, mean_b, higher_is_better=True)
                    if improved:
                        pval = paired_ttest_pvalue(
                            exp_u,
                            exp_b,
                            metric_key,
                            alternative=PAIRED_TEST_ALTERNATIVE,
                        )
                        if pval is not None and pval < 0.05:
                            annotations[(UNDERLINE_METHOD, metric_key)]["underline"] = True

    return annotations


# =====================================================================
# FORMATAGE LATEX
# =====================================================================

def format_metric_value(mean: Optional[float], ci: Optional[float], kind: str) -> str:
    if mean is None:
        return ""

    if kind == "percent":
        if SHOW_CONFIDENCE_INTERVALS and ci is not None:
            return f"{mean * 100:.2f}$\\pm${ci * 100:.2f}\\%"
        return f"{mean * 100:.2f}\\%"

    if kind == "seconds":
        if SHOW_CONFIDENCE_INTERVALS and ci is not None:
            return f"{mean:.2f}$\\pm${ci:.2f}"
        return f"{mean:.2f}"

    return str(mean)


def apply_cell_styles(text: str, bold: bool, underline: bool, superscript: str) -> str:
    styled = text

    if bold:
        styled = rf"\textbf{{{styled}}}"

    if underline:
        styled = rf"\underline{{{styled}}}"

    if superscript:
        styled = styled + rf"\textsuperscript{{{superscript}}}"

    return styled


def render_missing_row(k_str: str, method: str) -> str:
    cells = [MISSING_ROW_FIRST_METRIC_TEXT] + [""] * (len(METRIC_SPECS) - 1)
    return f"{k_str} & {latex_escape(method)} & " + " & ".join(cells) + r" \\"


def render_row(
    k_str: str,
    method: str,
    exp: Optional[Experiment],
    annotations: Dict[Tuple[str, str], Dict[str, object]],
) -> str:
    if exp is None:
        return render_missing_row(k_str, method)

    cells = []
    for spec in METRIC_SPECS:
        metric_key = spec["key"]
        mean = get_display_mean(exp, metric_key)
        ci = get_metric_ci(exp, metric_key)
        text = format_metric_value(mean, ci, spec["kind"])

        ann = annotations[(method, metric_key)]
        text = apply_cell_styles(
            text=text,
            bold=bool(ann["bold"]),
            underline=bool(ann["underline"]),
            superscript=str(ann["superscript"]),
        )
        cells.append(text)

    return f"{k_str} & {latex_escape(method)} & " + " & ".join(cells) + r" \\"


def generate_latex_table(
    experiments: List[Experiment],
    caption: str = "",
    label: str = "tab:IRexperimentresults",
) -> str:
    grouped = group_by_k(experiments)
    ks = sorted(grouped.keys())

    lines: List[str] = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{clccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{$k$} & \textbf{Method} & \textbf{Answer Relevancy} & "
        r"\textbf{Context Relevance} & \textbf{Answer Correctness} & "
        r"\textbf{NDCG@k} & \textbf{Avg. t/q(s)} \\"
    )
    lines.append(r"\midrule")

    for idx_k, k in enumerate(ks):
        rows_for_k = grouped[k]
        annotations = compute_annotations_for_k(rows_for_k)

        for idx_method, method in enumerate(METHOD_ORDER):
            exp = rows_for_k.get(method)
            k_str = str(k) if idx_method == 0 else ""
            lines.append(render_row(k_str, method, exp, annotations))

        if idx_k < len(ks) - 1:
            lines.append(r"\hline")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{latex_escape(label)}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Génère un tableau LaTeX depuis un dossier d'expériences.")
    parser.add_argument("root", type=str, help="Dossier racine contenant les sous-dossiers d'expériences")
    parser.add_argument("-o", "--output", type=str, default="table_results.tex", help="Fichier .tex de sortie")
    parser.add_argument("--caption", type=str, default="", help="Caption LaTeX")
    parser.add_argument("--label", type=str, default="tab:IRexperimentresults", help="Label LaTeX")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Erreur: dossier invalide ou introuvable: {root}")

    experiments = collect_experiments(root)
    if not experiments:
        raise SystemExit("Erreur: aucune expérience valide trouvée.")

    latex = generate_latex_table(
        experiments=experiments,
        caption=args.caption,
        label=args.label,
    )

    output_path = Path(args.output)
    output_path.write_text(latex, encoding="utf-8")

    print(f"Tableau LaTeX généré dans : {output_path}")
    print()
    print(latex)


if __name__ == "__main__":
    main()