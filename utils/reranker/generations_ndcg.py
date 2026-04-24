import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]


INPUT_FILE = PROJECT_ROOT / "data" / "reranker" / "processed" / "generations_enriched.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "reranker" / "baseline" / "generations.jsonl"
# Nombre de chunks à considérer dans le ranking observé.
# None = tous les retrieved_chunks
TOP_K = 10

# Champ utilisé comme gain de pertinence :
# - "relevance_score"  -> score continu
# - "is_relevant"      -> binaire (True/False)
GAIN_FIELD = "relevance_score"

# Formule DCG :
# True  -> (2^rel - 1) / log2(rank + 2)
# False -> rel / log2(rank + 2)
USE_EXPONENTIAL_GAIN = False

# Si aucun document pertinent (IDCG = 0), renvoyer :
EMPTY_IDCG_VALUE = 0.0


# ============================================================
# Helpers I/O
# ============================================================

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Cas JSON classique
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Cas JSONL
    items = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Ligne JSONL invalide dans {path} à la ligne {i}: {e}") from e
    return items


def save_as_jsonl(data: List[Dict[str, Any]], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


# ============================================================
# Helpers gains / champs
# ============================================================

def safe_float(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def get_chunk_field(chunk: Dict[str, Any], field_name: str, default: Any = None) -> Any:
    """
    Lit un champ d'abord au niveau top-level du chunk,
    puis dans chunk["metadata"].
    Compatible avec les deux schémas JSON.
    """
    if field_name in chunk:
        return chunk[field_name]

    metadata = chunk.get("metadata")
    if isinstance(metadata, dict) and field_name in metadata:
        return metadata[field_name]

    return default


def extract_gain(chunk: Dict[str, Any], gain_field: str) -> float:
    """
    Extrait le gain de pertinence depuis :
    - chunk[gain_field]
    - ou chunk["metadata"][gain_field]
    """
    if gain_field == "is_relevant":
        value = get_chunk_field(chunk, "is_relevant", False)
        return 1.0 if bool(value) else 0.0

    value = get_chunk_field(chunk, gain_field, 0.0)
    return safe_float(value)


# ============================================================
# NDCG
# ============================================================

def dcg(scores: List[float], use_exponential_gain: bool = True) -> float:
    total = 0.0
    for i, rel in enumerate(scores):
        denom = math.log2(i + 2)
        gain = (2 ** rel - 1) if use_exponential_gain else rel
        total += gain / denom
    return total


def ndcg_from_actual_and_all_scores(
    actual_scores: List[float],
    all_scores: List[float],
    top_k: Optional[int] = None,
    use_exponential_gain: bool = True,
    empty_idcg_value: float = 0.0,
) -> float:
    """
    NDCG@k standard :
    - actual_scores = scores du ranking observé
    - all_scores = tous les scores disponibles pour construire l'idéal
    """
    if not actual_scores:
        return empty_idcg_value

    if top_k is not None:
        actual_scores = actual_scores[:top_k]
        ideal_scores = sorted(all_scores, reverse=True)[:top_k]
    else:
        ideal_scores = sorted(all_scores, reverse=True)

    actual_dcg = dcg(actual_scores, use_exponential_gain=use_exponential_gain)
    ideal_dcg = dcg(ideal_scores, use_exponential_gain=use_exponential_gain)

    if ideal_dcg == 0.0:
        return empty_idcg_value

    return actual_dcg / ideal_dcg


# ============================================================
# Calcul principal
# ============================================================

def compute_ndcg_for_record(
    record: Dict[str, Any],
    top_k: Optional[int] = None,
    gain_field: str = "relevance_score",
    use_exponential_gain: bool = True,
    empty_idcg_value: float = 0.0
) -> Dict[str, Any]:
    out = dict(record)

    all_chunks = record.get("retrieved_chunks", []) or []
    ranked_chunks = all_chunks[:top_k] if top_k is not None else all_chunks

    actual_scores = [extract_gain(chunk, gain_field) for chunk in ranked_chunks]
    all_scores = [extract_gain(chunk, gain_field) for chunk in all_chunks]

    ndcg_value = ndcg_from_actual_and_all_scores(
        actual_scores=actual_scores,
        all_scores=all_scores,
        top_k=top_k,
        use_exponential_gain=use_exponential_gain,
        empty_idcg_value=empty_idcg_value,
    )

    out["computed_ndcg"] = ndcg_value
    out["computed_ndcg_gain_field"] = gain_field
    out["computed_ndcg_top_k"] = len(ranked_chunks) if top_k is not None else len(all_chunks)
    out["computed_ndcg_formula"] = "exp2" if use_exponential_gain else "linear"

    # Debug utile
    out["computed_ndcg_actual_scores"] = actual_scores
    out["computed_ndcg_ideal_scores"] = sorted(all_scores, reverse=True)[:len(actual_scores)] if actual_scores else []

    return out


def compute_dataset_average_ndcg(records: List[Dict[str, Any]]) -> float:
    values = []
    for record in records:
        value = record.get("computed_ndcg")
        if value is not None:
            values.append(float(value))
    if not values:
        return 0.0
    return sum(values) / len(values)


# ============================================================
# Main
# ============================================================

def main():
    print(f"Lecture de {INPUT_FILE} ...")
    records = load_json_or_jsonl(INPUT_FILE)

    print("Calcul du NDCG ...")
    enriched_records = [
        compute_ndcg_for_record(
            record,
            top_k=TOP_K,
            gain_field=GAIN_FIELD,
            use_exponential_gain=USE_EXPONENTIAL_GAIN,
            empty_idcg_value=EMPTY_IDCG_VALUE
        )
        for record in records
    ]

    avg_ndcg = compute_dataset_average_ndcg(enriched_records)

    save_as_jsonl(enriched_records, OUTPUT_FILE)

    print(f"OK: {len(enriched_records)} question(s) traitée(s)")
    print(f"NDCG moyen: {avg_ndcg:.6f}")
    print(f"Fichier écrit: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()