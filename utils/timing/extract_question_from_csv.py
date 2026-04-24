import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_csv = PROJECT_ROOT / "data" / "evaluation_questions.csv"
output_jsonl = PROJECT_ROOT / "data" / "processed" / "questions.jsonl"

output_jsonl.parent.mkdir(parents=True, exist_ok=True)

with input_csv.open("r", encoding="utf-8", newline="") as csvfile:
    reader = csv.DictReader(csvfile)

    with output_jsonl.open("w", encoding="utf-8") as outfile:
        for i, row in enumerate(reader):
            question = row.get("question", "").strip()
            if not question:
                continue

            record = {
                "question_id": f"q_{i:04d}",
                "question": question,
                "category": row.get("category", "").strip(),
                "reference": row.get("reference", "").strip(),
                "generated_answer": "",
                "retrieved_chunks": [],
            }

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Questions converties dans {output_jsonl}")