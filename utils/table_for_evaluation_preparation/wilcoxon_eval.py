import json
import numpy as np
from scipy.stats import wilcoxon


METRICS = [
    "answer_relevancy",
    "context_relevance",
    "answer_correctness",
    "chunk_relevance_ndcg_at_k",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(data):
    return {item["question_id"]: item for item in data if "question_id" in item}


def extract_paired_metric(data_a, data_b, metric, invalid_values=None):
    if invalid_values is None:
        invalid_values = {-1.0, None}

    index_a = build_index(data_a)
    index_b = build_index(data_b)

    common_ids = sorted(set(index_a.keys()) & set(index_b.keys()))

    values_a = []
    values_b = []
    kept_ids = []

    for qid in common_ids:
        va = index_a[qid].get(metric)
        vb = index_b[qid].get(metric)

        if va in invalid_values or vb in invalid_values:
            continue

        try:
            va = float(va)
            vb = float(vb)
        except (TypeError, ValueError):
            continue

        values_a.append(va)
        values_b.append(vb)
        kept_ids.append(qid)

    return np.array(values_a, dtype=float), np.array(values_b, dtype=float), kept_ids


def safe_wilcoxon(a, b):
    diffs = b - a

    # Si toutes les différences sont nulles, le test n'est pas applicable
    if np.allclose(diffs, 0):
        return None, None

    # zero_method="wilcox" ignore les différences exactement nulles
    stat, p_value = wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
    return stat, p_value


def compare_runs(file_a, file_b, label_a="Baseline", label_b="New"):
    data_a = load_json(file_a)
    data_b = load_json(file_b)

    print(f"\nComparaison : {label_a} vs {label_b}")
    print("=" * 70)

    for metric in METRICS:
        a, b, kept_ids = extract_paired_metric(data_a, data_b, metric)

        print(f"\nMetric: {metric}")
        print("-" * 70)

        if len(a) == 0:
            print("Aucune donnée valide pour cette métrique.")
            continue

        mean_a = np.mean(a)
        mean_b = np.mean(b)
        diff = b - a
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)

        stat, p_value = safe_wilcoxon(a, b)

        print(f"N commun            : {len(a)}")
        print(f"Moyenne {label_a:<10}: {mean_a:.6f}")
        print(f"Moyenne {label_b:<10}: {mean_b:.6f}")
        print(f"Diff moyenne (B-A)  : {mean_diff:.6f}")
        print(f"Diff médiane (B-A)  : {median_diff:.6f}")

        if p_value is None:
            print("Wilcoxon            : non applicable (différences nulles partout)")
            print("Conclusion          : pas de différence mesurable entre les deux runs")
            continue

        print(f"Wilcoxon statistic   : {stat:.6f}")
        print(f"p-value              : {p_value:.6g}")

        if p_value < 0.05:
            print("Conclusion          : différence statistiquement significative (p < 0.05)")
        else:
            print("Conclusion          : différence non significative (p >= 0.05)")


if __name__ == "__main__":
    # Remplace ces chemins par les tiens
    file_a = "generations_k5_answered/ragas_metrics.json"
    file_b = "generations_reranker_k5_answered/ragas_metrics.json"

    compare_runs(file_a, file_b, label_a="Baseline", label_b="New")