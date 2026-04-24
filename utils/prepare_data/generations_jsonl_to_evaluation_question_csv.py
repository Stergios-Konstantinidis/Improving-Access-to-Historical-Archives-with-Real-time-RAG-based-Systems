import json
import csv
from pathlib import Path


INPUT_FILE = "generations.jsonl"
OUTPUT_FILE = "evaluation_question.csv"


def normalize_value(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return "" if value.strip().lower() == "nan" else value

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def transform_record(record):
    return [
        normalize_value(record.get("question", "")),
        normalize_value(record.get("generated_answer", "")),
        normalize_value(record.get("reference", "")),
        normalize_value(record.get("retrieved_chunks", "")),
        normalize_value(record.get("category", "")),
    ]


def convert_jsonl_to_csv(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_file}")

    header = [
        "question",
        "groundtruth",
        "golden_doc_id",
        "knn_search_docs",
        "category",
    ]

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8", newline="") as fout:

        # Écriture manuelle de l'en-tête sans guillemets
        fout.write(",".join(header) + "\n")

        # Écriture des lignes avec guillemets partout
        writer = csv.writer(fout, quoting=csv.QUOTE_ALL)

        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Ligne {line_number} invalide dans le JSONL : {e}") from e

            writer.writerow(transform_record(record))

    print(f"Conversion terminée : {output_file}")


if __name__ == "__main__":
    convert_jsonl_to_csv(INPUT_FILE, OUTPUT_FILE)