import json
from pathlib import Path

def load_source_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def normalize_row(row: dict) -> dict | None:
    text = (row.get("text") or "").strip()
    if not text:
        return None

    # filtre simple anti-bruit
    if len(text) < 250:
        return None

    metadata = row.get("metadata", {}) or {}

    return {
        "id": str(row.get("id", "")),
        "text": text,
        "metadata": {
            "source_id": str(row.get("id", "")),
            "title": metadata.get("title"),
            "summary": metadata.get("summary"),
            "year": metadata.get("year"),
            "month": metadata.get("month"),
            "day": metadata.get("day"),
            "page_number": metadata.get("page_number"),
            "newspaper_issue_id": metadata.get("newspaper_issue_id"),
            "newspaper": metadata.get("newspaper"),
            "Person": metadata.get("Person"),
            "Location": metadata.get("Location"),
            "Event": metadata.get("Event"),
            "keywords": metadata.get("keywords"),
            "label": metadata.get("label"),
        },
    }

rows = load_source_jsonl("documents_index.jsonl")
chunks = []

seen = set()
for row in rows:
    item = normalize_row(row)
    if item is None:
        continue
    key = item["text"]
    if key in seen:
        continue
    seen.add(key)
    chunks.append(item)

print(f"{len(chunks)} chunks retenus")