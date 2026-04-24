# extract_questions_from_generations_jsonl.py
#!/usr/bin/env python3
"""
Extract questions from generations.jsonl into a CSV file.

Output CSV columns:
- question
- groundtruth
- golden_doc_id
- knn_search_docs
- category
"""

import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =============================
# Paths
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "baseline" / "generations.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "evaluation_questions_extracted_from_jsonl.csv"

def ensure_parent_dir(filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)


def extract_questions_from_jsonl(input_file: Path, output_file: Path) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    ensure_parent_dir(output_file)

    rows = []

    with input_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping invalid JSON on line {line_number}: {e}")
                continue

            question = record.get("question", "")
            if not isinstance(question, str):
                question = str(question)

            rows.append([
                question,
                "",
                "",
                "",
                "",
            ])

    with output_file.open("w", encoding="utf-8", newline="") as f:
        # Header écrit manuellement, sans guillemets
        f.write("question,groundtruth,golden_doc_id,knn_search_docs,category\n")

        # Données écrites avec guillemets forcés
        writer = csv.writer(
            f,
            quoting=csv.QUOTE_ALL,
        )
        writer.writerows(rows)

    print(f"CSV created: {output_file}")
    print(f"Extracted {len(rows)} questions from: {input_file}")


def main() -> None:
    extract_questions_from_jsonl(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()