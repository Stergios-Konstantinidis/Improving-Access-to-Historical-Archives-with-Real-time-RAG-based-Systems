import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("- ", "")
    return t.strip()


def _text_key(text: str) -> str:
    norm = _normalize_text(text)
    return hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()


def _get_score(chunk: Dict[str, Any]) -> Optional[float]:
    md = chunk.get("metadata") or {}
    if md.get("cross_score") is not None:
        try:
            return float(md["cross_score"])
        except (TypeError, ValueError):
            return None
    if chunk.get("rerank_score") is not None:
        try:
            return float(chunk["rerank_score"])
        except (TypeError, ValueError):
            return None
    return None


def _bucket_old_rank(r: int) -> str:
    if r < 10:
        return "00-09"
    if r < 20:
        return "10-19"
    if r < 30:
        return "20-29"
    if r < 40:
        return "30-39"
    if r < 50:
        return "40-49"
    return "50+"


def _build_baseline_text_index(baseline_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    idx: Dict[str, Dict[str, int]] = {}
    for rec in baseline_records:
        qid = str(rec.get("question_id"))
        chunks = rec.get("retrieved_chunks", []) or []
        m: Dict[str, int] = {}
        for c in chunks:
            txt = c.get("text") or ""
            r = c.get("rank")
            if not isinstance(r, int):
                continue
            k = _text_key(txt)
            if k not in m or r < m[k]:
                m[k] = r
        idx[qid] = m
    return idx


def compute_rerank_movement_report(
    *,
    baseline_jsonl: str,
    reranked_jsonl: str,
    top_k: int = 10,
) -> Dict[str, Any]:

    base_path = Path(baseline_jsonl)
    new_path = Path(reranked_jsonl)

    if not base_path.exists():
        raise FileNotFoundError(f"Baseline introuvable: {base_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"Reranked introuvable: {new_path}")

    baseline_records = _load_jsonl(base_path)
    reranked_records = _load_jsonl(new_path)
    base_idx = _build_baseline_text_index(baseline_records)

    global_deltas: List[int] = []
    old_rank_counter = Counter()
    bucket_counter = Counter()
    delta_by_bucket = defaultdict(list)

    # NEW METRICS
    entered_from_outside = 0
    entered_from_outside_old_ranks: List[int] = []
    entered_from_outside_bucket_counter = Counter()

    stayed_in_top_k = 0
    stayed_in_top_k_deltas: List[int] = []
    stayed_in_top_k_bucket_counter = Counter()

    total_top = 0
    not_matched = 0
    total_questions = 0
    per_question = []

    for rec in reranked_records:
        qid = str(rec.get("question_id"))
        question = rec.get("question", "")
        base_map = base_idx.get(qid, {})

        chunks = rec.get("retrieved_chunks", []) or []
        top_chunks = [c for c in chunks if isinstance(c.get("rank"), int) and c["rank"] < top_k]
        top_chunks.sort(key=lambda c: c["rank"])
        if not top_chunks:
            continue

        total_questions += 1
        q_deltas: List[int] = []
        moves = []

        for c in top_chunks:
            new_rank = int(c["rank"])
            txt = c.get("text") or ""
            k = _text_key(txt)

            old_rank = base_map.get(k)
            delta = None

            if old_rank is None:
                not_matched += 1
            else:
                old_rank = int(old_rank)
                delta = old_rank - new_rank

                # stayed in top_k
                if old_rank < top_k:
                    stayed_in_top_k += 1
                    stayed_in_top_k_deltas.append(delta)
                    stayed_in_top_k_bucket_counter[_bucket_old_rank(old_rank)] += 1

                # entered from outside
                else:
                    entered_from_outside += 1
                    entered_from_outside_old_ranks.append(old_rank)
                    entered_from_outside_bucket_counter[_bucket_old_rank(old_rank)] += 1

                q_deltas.append(delta)
                global_deltas.append(delta)

                old_rank_counter[old_rank] += 1
                b = _bucket_old_rank(old_rank)
                bucket_counter[b] += 1
                delta_by_bucket[b].append(delta)

            moves.append({
                "new_rank": new_rank,
                "old_rank": old_rank,
                "delta": delta,
                "score": _get_score(c),
                "id": c.get("id"),
            })

            total_top += 1

        per_question.append({
            "question_id": qid,
            "question": question,
            "n_top_k": len(moves),
            "avg_delta": (sum(q_deltas) / len(q_deltas)) if q_deltas else None,
            "best_gain": max(q_deltas) if q_deltas else None,
            "worst_drop": min(q_deltas) if q_deltas else None,
            "moves": moves,
        })

    total_matched = total_top - not_matched

    def pct(x, total):
        return round((x / total) * 100, 2) if total else None

    report = {
        "inputs": {
            "baseline_jsonl": str(base_path),
            "reranked_jsonl": str(new_path),
            "top_k": top_k,
        },
        "global": {
            "total_questions_processed": total_questions,
            "top_k_comparisons": total_top,
            "matched": total_matched,
            "not_matched": not_matched,
            "match_rate": (total_matched / total_top) if total_top else None,
            "avg_delta_overall": (sum(global_deltas) / len(global_deltas)) if global_deltas else None,
            "max_gain_overall": max(global_deltas) if global_deltas else None,
            "max_drop_overall": min(global_deltas) if global_deltas else None,

            "entered_top_k_from_outside": entered_from_outside,
            "entered_top_k_from_outside_pct": pct(entered_from_outside, total_top),
            "entered_top_k_from_outside_avg_old_rank": (
                sum(entered_from_outside_old_ranks) / len(entered_from_outside_old_ranks)
                if entered_from_outside_old_ranks else None
            ),
            "entered_top_k_from_outside_buckets": {
                b: {
                    "count": c,
                    "pct_of_entered": pct(c, entered_from_outside)
                }
                for b, c in entered_from_outside_bucket_counter.items()
            },

            "stayed_in_top_k": stayed_in_top_k,
            "stayed_in_top_k_avg_delta": (
                sum(stayed_in_top_k_deltas) / len(stayed_in_top_k_deltas)
                if stayed_in_top_k_deltas else None
            ),
            "stayed_in_top_k_buckets": {
                b: {
                    "count": c,
                    "pct_of_stayed": pct(c, stayed_in_top_k)
                }
                for b, c in stayed_in_top_k_bucket_counter.items()
            },
        }
    }

    return report


def write_rerank_movement_report(
    *,
    baseline_jsonl: str,
    reranked_jsonl: str,
    out_path: str,
    top_k: int = 10,
) -> str:

    report = compute_rerank_movement_report(
        baseline_jsonl=baseline_jsonl,
        reranked_jsonl=reranked_jsonl,
        top_k=top_k,
    )

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return str(outp)