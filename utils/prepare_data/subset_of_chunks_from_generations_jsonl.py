# subset_of_chunks_from_generations_jsonl.py
#!/usr/bin/env python3
"""
Trim retrieved_chunks in a generations.jsonl file.

Usage:
    python trim_generations.py
    python trim_generations.py --k 5
    python trim_generations.py --input path/to/generations.jsonl --output path/to/output.jsonl --k 5
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

# =============================
# Paths
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVED_FILE = PROJECT_ROOT / "data" / "reranker" / "baseline" / "generations.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "reranker" / "processed" / "subset"


def ensure_parent_dir(filepath: Path) -> None:
    """Create parent directory if it does not exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    rows: List[Dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number} in {filepath}: {e}") from e
    return rows


def save_jsonl(rows: List[Dict[str, Any]], filepath: Path) -> None:
    """Save a list of dicts to JSONL."""
    ensure_parent_dir(filepath)
    with filepath.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def trim_retrieved_chunks(records: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Keep only the first k retrieved_chunks for each record.
    Preserves the rest of the JSON structure as-is.
    """
    trimmed_records: List[Dict[str, Any]] = []

    for record in records:
        new_record = dict(record)

        retrieved_chunks = new_record.get("retrieved_chunks", [])
        if isinstance(retrieved_chunks, list):
            new_record["retrieved_chunks"] = retrieved_chunks[:k]
        else:
            new_record["retrieved_chunks"] = []

        trimmed_records.append(new_record)

    return trimmed_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim retrieved_chunks in generations.jsonl to the first k chunks."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(RETRIEVED_FILE),
        help="Path to the input generations.jsonl file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSONL path. If omitted, uses generations_kX.jsonl",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of retrieved_chunks to keep (default: 10)",
    )

    args = parser.parse_args()

    if args.k < 0:
        raise ValueError("--k must be >= 0")

    input_file = Path(args.input).resolve()
    k = args.k

    if args.output is not None:
        output_file = Path(args.output).resolve()
    else:
        output_file = OUTPUT_DIR / f"generations_k{k}.jsonl"

    print("=" * 70)
    print("TRIM GENERATIONS")
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")
    print(f"k     : {k}")
    print("=" * 70)

    records = load_jsonl(input_file)
    trimmed_records = trim_retrieved_chunks(records, k)
    save_jsonl(trimmed_records, output_file)

    print(f"Saved {len(trimmed_records)} records to: {output_file}")


if __name__ == "__main__":
    main()