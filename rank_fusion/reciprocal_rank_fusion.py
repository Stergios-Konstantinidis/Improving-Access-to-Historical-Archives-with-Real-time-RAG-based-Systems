#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def progress(iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide dans {path} à la ligne {line_no}: {exc}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def coerce_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def coerce_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def merge_latency_ms(existing_latency: Any, add_ms: float) -> Any:
    add_ms = float(add_ms)

    if isinstance(existing_latency, dict):
        out = dict(existing_latency)
        current_total = coerce_float(out.get("real_pipeline_ms"), 0.0) or 0.0
        out["rrf_fusion_ms"] = add_ms
        out["real_pipeline_ms"] = current_total + add_ms
        return out

    current = coerce_float(existing_latency, 0.0) or 0.0
    return current + add_ms


def get_source(chunk: Dict[str, Any]) -> str:
    src = chunk.get("retrieval_source")
    if isinstance(src, str) and src.strip():
        return src.strip().lower()

    metadata = safe_dict(chunk.get("metadata"))
    src = metadata.get("retrieval_source")
    if isinstance(src, str) and src.strip():
        return src.strip().lower()

    raise ValueError("retrieval_source manquant")


def get_local_rank(chunk: Dict[str, Any], fallback_rank: int) -> int:
    metadata = safe_dict(chunk.get("metadata"))

    for value in (
        chunk.get("retrieval_rank"),
        metadata.get("retrieval_rank"),
        chunk.get("rank"),
        metadata.get("rank"),
    ):
        rank = coerce_int(value, None)
        if rank is not None:
            return max(rank, 0)

    return fallback_rank


def get_dedup_key(chunk: Dict[str, Any]) -> str:
    text_key = normalize_text(chunk.get("text"))
    if text_key:
        text_hash = hashlib.sha1(text_key.encode("utf-8")).hexdigest()
        return f"textsha1::{text_hash}"

    metadata = safe_dict(chunk.get("metadata"))
    doc_id = metadata.get("document_id") or metadata.get("doc_id")
    chunk_id = metadata.get("chunk_id")
    if doc_id is not None and chunk_id is not None:
        return f"docchunk::{doc_id}::{chunk_id}"

    _id = chunk.get("id")
    if _id not in (None, ""):
        return f"id::{_id}"

    return "unknown::empty"


def quality_key(
    chunk: Dict[str, Any],
    source_weights: Dict[str, float],
    fallback_rank: int,
) -> Tuple[float, int, int, int]:
    src = get_source(chunk)
    rank = get_local_rank(chunk, fallback_rank)
    text_len = len(normalize_text(chunk.get("text")))
    meta_len = len(safe_dict(chunk.get("metadata")))
    return (
        float(source_weights.get(src, 1.0)),
        -rank,
        text_len,
        meta_len,
    )


def pick_best_representative(
    existing: Dict[str, Any],
    candidate: Dict[str, Any],
    source_weights: Dict[str, float],
    existing_fallback_rank: int,
    candidate_fallback_rank: int,
) -> Dict[str, Any]:
    ex_key = quality_key(existing, source_weights, existing_fallback_rank)
    ca_key = quality_key(candidate, source_weights, candidate_fallback_rank)
    return candidate if ca_key > ex_key else existing


def count_final_sources(final_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_representative_source = {
        "chroma": 0,
        "bm25s": 0,
        "temporal": 0,
        "other": 0,
    }

    by_presence_in_fusion_sources = {
        "chroma": 0,
        "bm25s": 0,
        "temporal": 0,
    }

    overlap_chroma_bm25s = 0
    overlap_chroma_temporal = 0
    overlap_bm25s_temporal = 0
    overlap_all_three = 0

    for chunk in final_chunks:
        try:
            rep_src = get_source(chunk)
        except Exception:
            rep_src = "other"

        if rep_src in by_representative_source:
            by_representative_source[rep_src] += 1
        else:
            by_representative_source["other"] += 1

        metadata = safe_dict(chunk.get("metadata"))
        fusion_sources = metadata.get("fusion_sources", [])

        if not isinstance(fusion_sources, list):
            fusion_sources = []

        fusion_sources_set = {
            str(x).strip().lower()
            for x in fusion_sources
            if str(x).strip()
        }

        if "chroma" in fusion_sources_set:
            by_presence_in_fusion_sources["chroma"] += 1
        if "bm25s" in fusion_sources_set:
            by_presence_in_fusion_sources["bm25s"] += 1
        if "temporal" in fusion_sources_set:
            by_presence_in_fusion_sources["temporal"] += 1

        if "chroma" in fusion_sources_set and "bm25s" in fusion_sources_set:
            overlap_chroma_bm25s += 1
        if "chroma" in fusion_sources_set and "temporal" in fusion_sources_set:
            overlap_chroma_temporal += 1
        if "bm25s" in fusion_sources_set and "temporal" in fusion_sources_set:
            overlap_bm25s_temporal += 1
        if {"chroma", "bm25s", "temporal"}.issubset(fusion_sources_set):
            overlap_all_three += 1

    return {
        "top_k_size": len(final_chunks),
        "by_representative_source": by_representative_source,
        "by_presence_in_fusion_sources": by_presence_in_fusion_sources,
        "overlap_chroma_bm25s": overlap_chroma_bm25s,
        "overlap_chroma_temporal": overlap_chroma_temporal,
        "overlap_bm25s_temporal": overlap_bm25s_temporal,
        "overlap_all_three": overlap_all_three,
    }


def fuse_row_rrf(
    row: Dict[str, Any],
    keep_top_k: int,
    rrf_k: int,
    source_weights: Dict[str, float],
    source_max_ranks: Dict[str, int],
    reserve_bm25s: int,
    max_bm25s_final: int | None,
) -> Dict[str, Any]:
    t0 = now_ms()

    out = dict(row)
    chunks_raw = out.get("retrieved_chunks", [])
    if not isinstance(chunks_raw, list):
        chunks_raw = []

    skipped_invalid = 0
    skipped_by_cap = 0

    source_seen_positions: Dict[str, int] = {}
    merged: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks_raw:
        if not isinstance(chunk, dict):
            skipped_invalid += 1
            continue

        try:
            src = get_source(chunk)
        except Exception:
            skipped_invalid += 1
            continue

        fallback_rank = source_seen_positions.get(src, 0)
        source_seen_positions[src] = fallback_rank + 1

        local_rank = get_local_rank(chunk, fallback_rank)

        max_rank_for_source = source_max_ranks.get(src, 10**9)
        if local_rank >= max_rank_for_source:
            skipped_by_cap += 1
            continue

        key = get_dedup_key(chunk)

        if key not in merged:
            merged[key] = {
                "representative": dict(chunk),
                "source_ranks": {src: local_rank},
                "fallback_ranks": {src: fallback_rank},
                "best_source_rank": local_rank,
                "dedup_key": key,
            }
            continue

        entry = merged[key]
        entry["source_ranks"][src] = min(
            local_rank,
            coerce_int(entry["source_ranks"].get(src), 10**9) or 10**9,
        )
        entry["fallback_ranks"][src] = min(
            fallback_rank,
            coerce_int(entry["fallback_ranks"].get(src), 10**9) or 10**9,
        )
        entry["best_source_rank"] = min(
            int(entry["best_source_rank"]),
            int(local_rank),
        )

        rep = entry["representative"]
        rep_src = get_source(rep)
        rep_fallback_rank = entry["fallback_ranks"].get(rep_src, 10**9)

        entry["representative"] = pick_best_representative(
            existing=rep,
            candidate=chunk,
            source_weights=source_weights,
            existing_fallback_rank=rep_fallback_rank,
            candidate_fallback_rank=fallback_rank,
        )

    fused_chunks: List[Dict[str, Any]] = []

    for entry in merged.values():
        rep = dict(entry["representative"])
        source_ranks = dict(entry["source_ranks"])
        best_source_rank = int(entry["best_source_rank"])
        dedup_key = entry["dedup_key"]

        rrf_score = 0.0
        for src, rank in source_ranks.items():
            weight = float(source_weights.get(src, 1.0))
            rrf_score += weight / (rrf_k + int(rank) + 1)

        metadata = safe_dict(rep.get("metadata"))
        metadata["fusion_rrf_score"] = rrf_score
        metadata["fusion_source_ranks"] = source_ranks
        metadata["fusion_sources"] = sorted(source_ranks.keys())
        metadata["fusion_num_sources"] = len(source_ranks)
        metadata["fusion_best_source_rank"] = best_source_rank
        metadata["fusion_algorithm"] = "weighted_rrf"
        rep["metadata"] = metadata

        rep["_fusion_rrf_score"] = rrf_score
        rep["_best_source_rank"] = best_source_rank
        rep["_fusion_num_sources"] = len(source_ranks)
        rep["_dedup_key"] = dedup_key

        fused_chunks.append(rep)

    def sort_key(ch: Dict[str, Any]) -> Tuple[float, int, int]:
        rrf_score = coerce_float(ch.get("_fusion_rrf_score"), 0.0) or 0.0
        best_rank = coerce_int(ch.get("_best_source_rank"), 10**9) or 10**9
        num_sources = coerce_int(ch.get("_fusion_num_sources"), 1) or 1
        return (-rrf_score, best_rank, -num_sources)

    fused_chunks.sort(key=sort_key)

    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    selected_bm25s = 0

    if reserve_bm25s > 0:
        for ch in fused_chunks:
            metadata = safe_dict(ch.get("metadata"))
            fusion_sources = metadata.get("fusion_sources", [])
            has_bm25s = "bm25s" in fusion_sources

            if not has_bm25s:
                continue

            if max_bm25s_final is not None and selected_bm25s >= max_bm25s_final:
                break

            key = ch.get("_dedup_key")
            if key not in selected_keys:
                selected.append(ch)
                selected_keys.add(key)
                selected_bm25s += 1

            if selected_bm25s >= reserve_bm25s:
                break

    for ch in fused_chunks:
        if len(selected) >= keep_top_k:
            break

        key = ch.get("_dedup_key")
        if key in selected_keys:
            continue

        metadata = safe_dict(ch.get("metadata"))
        fusion_sources = metadata.get("fusion_sources", [])
        has_bm25s = "bm25s" in fusion_sources

        if has_bm25s and max_bm25s_final is not None and selected_bm25s >= max_bm25s_final:
            continue

        selected.append(ch)
        selected_keys.add(key)

        if has_bm25s:
            selected_bm25s += 1

    final_chunks: List[Dict[str, Any]] = []
    for new_rank, chunk in enumerate(selected[:keep_top_k]):
        clean_chunk = dict(chunk)
        clean_chunk.pop("_fusion_rrf_score", None)
        clean_chunk.pop("_best_source_rank", None)
        clean_chunk.pop("_fusion_num_sources", None)
        clean_chunk.pop("_dedup_key", None)
        clean_chunk["rank"] = new_rank

        metadata = safe_dict(clean_chunk.get("metadata"))
        metadata["fusion_final_rank"] = new_rank
        clean_chunk["metadata"] = metadata

        final_chunks.append(clean_chunk)

    final_source_counts = count_final_sources(final_chunks)

    out["retrieved_chunks"] = final_chunks

    method_metadata = safe_dict(out.get("method_metadata"))
    method_metadata["rank_fusion"] = {
        "algorithm": "weighted_rrf",
        "rrf_k": rrf_k,
        "source_weights": source_weights,
        "source_max_ranks": source_max_ranks,
        "reserve_bm25s": reserve_bm25s,
        "input_count": len(chunks_raw),
        "unique_count_after_dedup": len(fused_chunks),
        "output_count": len(final_chunks),
        "dedup_key": "textsha1_first",
        "skipped_invalid": skipped_invalid,
        "skipped_by_cap": skipped_by_cap,
        "final_top_k_source_counts": final_source_counts,
        "max_bm25s_final": max_bm25s_final,
    }
    method_metadata["final_top_k"] = keep_top_k
    out["method_metadata"] = method_metadata

    fusion_ms = now_ms() - t0
    out["latency_ms"] = merge_latency_ms(out.get("latency_ms"), fusion_ms)

    breakdown = safe_dict(out.get("latency_breakdown_ms"))
    breakdown["rrf_fusion_ms"] = fusion_ms
    out["latency_breakdown_ms"] = breakdown

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusion Weighted RRF avec réserve BM25s."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--keep-top-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--chroma-weight", type=float, default=1.0)
    parser.add_argument("--bm25s-weight", type=float, default=0.9)
    parser.add_argument("--chroma-max-rank", type=int, default=25)
    parser.add_argument("--bm25s-max-rank", type=int, default=8)
    parser.add_argument(
        "--reserve-bm25s",
        type=int,
        default=0,
        help="Nombre de chunks BM25s garantis dans le pool final",
    )
    parser.add_argument(
        "--max-bm25s-final",
        type=int,
        default=None,
        help="Nombre maximum de chunks contenant bm25s autorisés dans le top final",
    )
    parser.add_argument("--temporal-weight", type=float, default=1.0)
    parser.add_argument("--temporal-max-rank", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)

    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError("Aucune ligne valide dans le fichier d'entrée.")

    input_source_counts = {"chroma": 0, "bm25s": 0, "temporal": 0}

    for row in rows:
        chunks = row.get("retrieved_chunks", [])
        if not isinstance(chunks, list):
            continue

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            try:
                src = get_source(chunk)
            except Exception:
                continue

            if src in input_source_counts:
                input_source_counts[src] += 1

    print("\nRépartition des chunks avant RRF :")
    print(f"  chroma   : {input_source_counts['chroma']}")
    print(f"  bm25s    : {input_source_counts['bm25s']}")
    print(f"  temporal : {input_source_counts['temporal']}")

    source_weights = {
        "chroma": float(args.chroma_weight),
        "bm25s": float(args.bm25s_weight),
        "temporal": float(args.temporal_weight),
    }
    source_max_ranks = {
        "chroma": int(args.chroma_max_rank),
        "bm25s": int(args.bm25s_max_rank),
        "temporal": int(args.temporal_max_rank),
    }

    out_rows: List[Dict[str, Any]] = []
    global_rep_counts = {"chroma": 0, "bm25s": 0, "temporal": 0, "other": 0}
    global_presence_counts = {"chroma": 0, "bm25s": 0, "temporal": 0}
    global_overlap_chroma_bm25s = 0
    global_overlap_chroma_temporal = 0
    global_overlap_bm25s_temporal = 0
    global_overlap_all_three = 0
    question_count = 0

    for row in progress(rows, desc="Weighted RRF fusion", unit="q"):
        fused_row = fuse_row_rrf(
            row=row,
            keep_top_k=int(args.keep_top_k),
            rrf_k=int(args.rrf_k),
            source_weights=source_weights,
            source_max_ranks=source_max_ranks,
            reserve_bm25s=int(args.reserve_bm25s),
            max_bm25s_final=args.max_bm25s_final,
        )
        out_rows.append(fused_row)

        rank_fusion = safe_dict(safe_dict(fused_row.get("method_metadata")).get("rank_fusion"))
        counts = safe_dict(rank_fusion.get("final_top_k_source_counts"))

        rep = safe_dict(counts.get("by_representative_source"))
        pres = safe_dict(counts.get("by_presence_in_fusion_sources"))

        global_rep_counts["chroma"] += coerce_int(rep.get("chroma"), 0) or 0
        global_rep_counts["bm25s"] += coerce_int(rep.get("bm25s"), 0) or 0
        global_rep_counts["temporal"] += coerce_int(rep.get("temporal"), 0) or 0
        global_rep_counts["other"] += coerce_int(rep.get("other"), 0) or 0

        global_presence_counts["chroma"] += coerce_int(pres.get("chroma"), 0) or 0
        global_presence_counts["bm25s"] += coerce_int(pres.get("bm25s"), 0) or 0
        global_presence_counts["temporal"] += coerce_int(pres.get("temporal"), 0) or 0

        global_overlap_chroma_bm25s += coerce_int(counts.get("overlap_chroma_bm25s"), 0) or 0
        global_overlap_chroma_temporal += coerce_int(counts.get("overlap_chroma_temporal"), 0) or 0
        global_overlap_bm25s_temporal += coerce_int(counts.get("overlap_bm25s_temporal"), 0) or 0
        global_overlap_all_three += coerce_int(counts.get("overlap_all_three"), 0) or 0
        question_count += 1

    write_jsonl(output_path, out_rows)

    print(f"Fichier généré : {output_path}")
    print(f"Questions traitées : {len(out_rows)}")
    print(f"Top-k final fusion : {args.keep_top_k}")
    print(f"RRF k : {args.rrf_k}")
    print(f"Source weights : {source_weights}")
    print(f"Source max ranks : {source_max_ranks}")
    print(f"Reserve BM25s : {args.reserve_bm25s}")

    if question_count > 0:
        print("\nRésumé global sur tous les top-k finaux :")
        print(f"  Représentant final - chroma   : {global_rep_counts['chroma']}")
        print(f"  Représentant final - bm25s    : {global_rep_counts['bm25s']}")
        print(f"  Représentant final - temporal : {global_rep_counts['temporal']}")
        print(f"  Représentant final - other    : {global_rep_counts['other']}")
        print(f"  Présence fusion - chroma      : {global_presence_counts['chroma']}")
        print(f"  Présence fusion - bm25s       : {global_presence_counts['bm25s']}")
        print(f"  Présence fusion - temporal    : {global_presence_counts['temporal']}")
        print(f"  Overlap chroma+bm25s          : {global_overlap_chroma_bm25s}")
        print(f"  Overlap chroma+temporal       : {global_overlap_chroma_temporal}")
        print(f"  Overlap bm25s+temporal        : {global_overlap_bm25s_temporal}")
        print(f"  Overlap all three             : {global_overlap_all_three}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())