import pandas as pd
import json


csv_path = "data/evaluation_questions.csv"
out_gold = "data/gold_answers_garanteed_k50.json"
out_jsonl = "data/evaluation_questions_garanteed_k50.jsonl"
print("RUN")

df = pd.read_csv(csv_path)

# Adapte ces noms si ton CSV change
df["question"] = df["question"].astype(str).str.strip()
df["groundtruth"] = df["groundtruth"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()

df = df.reset_index(drop=True)
df["question_id"] = df.index.map(lambda i: f"q_{i:04d}")

gold_answers = dict(zip(df["question_id"], df["groundtruth"]))

with open(out_gold, "w", encoding="utf-8") as f:
    json.dump(gold_answers, f, ensure_ascii=False, indent=2)

with open(out_jsonl, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        rec = {
            "question_id": row["question_id"],
            "question": row["question"],
            "category": row["category"],
            "reference": row["groundtruth"],   # gold inline (optionnel mais pratique)
            "generated_answer": "",
            "retrieved_chunks": []
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("OK:", out_gold, out_jsonl)