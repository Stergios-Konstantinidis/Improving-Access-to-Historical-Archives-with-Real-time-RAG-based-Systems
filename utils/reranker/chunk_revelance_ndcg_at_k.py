import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ce script permet dajouter au JSON la revelance de chaque chunk et directement de le mettre dans le JSON depuis le metric.json produit lors de l'évaluation (run d'alexi)


# ============================================================
# Configuration : fichiers dans le dossier courant
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVED_FILE = PROJECT_ROOT / "data" /  "baseline" / "generations.jsonl"
ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "reranker" / "processed" / "metrics.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "reranker" / "processed"/ "generations_enriched.jsonl"

DEBUG_MATCH = False  # True pour ajouter _match_strategy

# Nouveau : mélange l'ordre des chunks en sortie
SHUFFLE = False
SHUFFLE_SEED = 42

# Si SHUFFLE=True :
# - True  -> réécrit rank selon le nouvel ordre
# - False -> garde le rank d'origine même si l'ordre est changé
RENUMBER_RANK_AFTER_SHUFFLE = True

# Si on renumérote, on garde l'ancien rank ici
KEEP_ORIGINAL_RANK = True


# ============================================================
# Helpers de chargement
# ============================================================

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Charge un fichier JSON ou JSONL.
    - JSON array -> retourne la liste
    - JSON object -> retourne [obj]
    - JSONL -> retourne une liste d'objets
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Essai JSON classique
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Format JSON non supporté dans {path}")
    except json.JSONDecodeError:
        pass

    # Sinon JSONL
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


def save_as_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """
    Écrit un fichier JSONL :
    un objet JSON compact par ligne.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.split()).strip()


# ============================================================
# Index des annotations
# ============================================================

def build_annotation_index(annotation_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}

    for record in annotation_records:
        qid = record.get("question_id")
        if not qid:
            continue

        chunk_details = record.get("chunk_details", [])

        by_rank = {}
        by_text = {}
        by_rank_and_text = {}

        for chunk in chunk_details:
            rank = chunk.get("rank")
            chunk_text = normalize_text(chunk.get("chunk_text", ""))

            if rank is not None:
                by_rank[rank] = chunk

            if chunk_text:
                by_text[chunk_text] = chunk

            if rank is not None and chunk_text:
                by_rank_and_text[(rank, chunk_text)] = chunk

        index[qid] = {
            "record": record,
            "by_rank": by_rank,
            "by_text": by_text,
            "by_rank_and_text": by_rank_and_text,
        }

    return index


# ============================================================
# Matching
# ============================================================

def find_matching_chunk(
    qid: str,
    retrieved_chunk: Dict[str, Any],
    ann_index: Dict[str, Dict[str, Any]]
):
    qdata = ann_index.get(qid)
    if not qdata:
        return {}, "no_question_match"

    rank = retrieved_chunk.get("rank")
    text = normalize_text(retrieved_chunk.get("text", ""))

    if rank is not None and text:
        match = qdata["by_rank_and_text"].get((rank, text))
        if match:
            return match, "question_id+rank+text"

    return {}, "no_chunk_match"


# ============================================================
# Shuffle
# ============================================================

def shuffle_chunks(
    chunks: List[Dict[str, Any]],
    rng: random.Random,
    renumber_rank: bool = True,
    keep_original_rank: bool = True
) -> List[Dict[str, Any]]:
    shuffled = [dict(chunk) for chunk in chunks]
    rng.shuffle(shuffled)

    if renumber_rank:
        for new_rank, chunk in enumerate(shuffled):
            if keep_original_rank and "original_rank" not in chunk:
                chunk["original_rank"] = chunk.get("rank")
            chunk["rank"] = new_rank

    return shuffled


# ============================================================
# Enrichissement
# ============================================================

def enrich_records(
    retrieved_records: List[Dict[str, Any]],
    annotation_records: List[Dict[str, Any]],
    include_debug_match_field: bool = False,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    renumber_rank_after_shuffle: bool = True,
    keep_original_rank: bool = True
) -> List[Dict[str, Any]]:
    ann_index = build_annotation_index(annotation_records)
    enriched = []
    rng = random.Random(shuffle_seed)

    for rec in retrieved_records:
        qid = rec.get("question_id")
        q_ann = ann_index.get(qid, {}).get("record", {})

        out = dict(rec)

        # Ajout des champs top-level depuis le fichier d'annotations
        if q_ann:
            out["gold_answer"] = q_ann.get("gold_answer")
            out["chunk_relevance_precision_at_k"] = q_ann.get("chunk_relevance_precision_at_k")
            out["chunk_relevance_ndcg_at_k"] = q_ann.get("chunk_relevance_ndcg_at_k")

        retrieved_chunks = rec.get("retrieved_chunks", [])
        enriched_chunks = []

        for chunk in retrieved_chunks:
            new_chunk = dict(chunk)
            match, match_strategy = find_matching_chunk(qid, chunk, ann_index)

            if match:
                new_chunk["chunk_answer"] = match.get("chunk_answer")
                new_chunk["relevance_score"] = match.get("relevance_score")
                new_chunk["justification"] = match.get("justification")
                new_chunk["is_relevant"] = match.get("is_relevant")
            else:
                new_chunk["chunk_answer"] = None
                new_chunk["relevance_score"] = None
                new_chunk["justification"] = None
                new_chunk["is_relevant"] = None

            if include_debug_match_field:
                new_chunk["_match_strategy"] = match_strategy

            enriched_chunks.append(new_chunk)

        # Shuffle seulement après le matching
        if shuffle:
            enriched_chunks = shuffle_chunks(
                chunks=enriched_chunks,
                rng=rng,
                renumber_rank=renumber_rank_after_shuffle,
                keep_original_rank=keep_original_rank
            )
            out["chunks_shuffled"] = True
            out["shuffle_seed"] = shuffle_seed
        else:
            out["chunks_shuffled"] = False

        # On garde l'ordre final choisi
        out["retrieved_chunks"] = enriched_chunks
        enriched.append(out)

    return enriched


# ============================================================
# Main
# ============================================================

def main():
    print(f"Lecture de {RETRIEVED_FILE} ...")
    retrieved_records = load_json_or_jsonl(RETRIEVED_FILE)

    print(f"Lecture de {ANNOTATIONS_FILE} ...")
    annotation_records = load_json_or_jsonl(ANNOTATIONS_FILE)

    print("Enrichissement en cours ...")
    enriched = enrich_records(
        retrieved_records=retrieved_records,
        annotation_records=annotation_records,
        include_debug_match_field=DEBUG_MATCH,
        shuffle=SHUFFLE,
        shuffle_seed=SHUFFLE_SEED,
        renumber_rank_after_shuffle=RENUMBER_RANK_AFTER_SHUFFLE,
        keep_original_rank=KEEP_ORIGINAL_RANK
    )

    save_as_jsonl(enriched, OUTPUT_FILE)

    print(f"OK: {len(enriched)} question(s) enrichie(s)")
    print(f"Shuffle activé: {SHUFFLE}")
    print(f"Fichier JSONL écrit: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()