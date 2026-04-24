# clear_generated_answers_from_generations_jsonl.py
#!/usr/bin/env python3
"""
Clear all generated_answer fields in a generations.jsonl file.

Usage:
    python clear_generated_answers.py
    python clear_generated_answers.py --input path/to/generations.jsonl
    python clear_generated_answers.py --output path/to/output.jsonl
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

INPUT_FILE = PROJECT_ROOT / "data" / "reranker" / "baseline" / "generations.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "reranker" / "processed"


def ensure_parent_dir(filepath: Path) -> None:
    """Create parent directory if it does not exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    records: List[Dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {filepath}: {e}"
                ) from e

    return records


def save_jsonl(records: List[Dict[str, Any]], filepath: Path) -> None:
    """Save JSONL file."""
    ensure_parent_dir(filepath)

    with filepath.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_generated_answers(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Set generated_answer to an empty string for all records."""
    updated_records: List[Dict[str, Any]] = []

    for record in records:
        new_record = dict(record)
        new_record["generated_answer"] = ""
        updated_records.append(new_record)

    return updated_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear all generated_answer fields in a generations.jsonl file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_FILE),
        help="Path to input generations.jsonl file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file. If omitted, writes <input_stem>_cleared.jsonl in OUTPUT_DIR",
    )

    args = parser.parse_args()

    input_file = Path(args.input).resolve()

    if args.output is not None:
        output_file = Path(args.output).resolve()
    else:
        output_file = OUTPUT_DIR / f"{input_file.stem}_cleared.jsonl"

    print("=" * 70)
    print("CLEAR GENERATED ANSWERS")
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")
    print("=" * 70)

    records = load_jsonl(input_file)
    updated_records = clear_generated_answers(records)
    save_jsonl(updated_records, output_file)

    print(f"\nSaved file: {output_file}")


if __name__ == "__main__":
    main()