import os
import csv
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

project_root = "/Users/stergios/Documents/GitHub/PKDD_2026"
load_dotenv(os.path.join(project_root, '.env'), override=True)

api_key = os.environ.get("GEMINIAPIKEY") or os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

client_05m = chromadb.PersistentClient(path=os.path.join(project_root, "vector_db", "apple_ocr_gemini_05m"))
col_05m = client_05m.get_collection("apple_ocr_gemini_05m")

client_05m_corr = chromadb.PersistentClient(path=os.path.join(project_root, "vector_db", "apple_ocr_gemini_05m_corrected"))
col_05m_corr = client_05m_corr.get_collection("apple_ocr_gemini_05m_corrected")

input_csv = os.path.join(project_root, "IR_evaluation", "data", "evaluation_questions_garanteed_k50.csv")
output_csv = os.path.join(project_root, "IR_evaluation", "data", "evaluation_questions_neutral.csv")

def embed_questions(questions):
    res = genai.embed_content(
        model="models/gemini-embedding-001",
        content=questions,
        task_type="retrieval_query"
    )
    return res['embedding']

with open(input_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

fieldnames = list(rows[0].keys())

kept_rows = []

batch_size = 50
for i in tqdm(range(0, len(rows), batch_size)):
    batch = rows[i:i+batch_size]
    questions = [row['question'] for row in batch]
    embeddings = embed_questions(questions)
    
    res_05m = col_05m.query(
        query_embeddings=embeddings,
        n_results=50,
        include=["distances"]
    )
    
    res_corr = col_05m_corr.query(
        query_embeddings=embeddings,
        n_results=50,
        include=["distances"]
    )
    
    for j, row in enumerate(batch):
        golden_doc_id = str(row['golden_doc_id']).strip()
        ids_05m = res_05m['ids'][j]
        ids_corr = res_corr['ids'][j]
        
        in_05m = golden_doc_id in ids_05m
        in_corr = golden_doc_id in ids_corr
        
        if in_05m and in_corr:
            kept_rows.append(row)

print(f"Original questions: {len(rows)}")
print(f"Questions kept (neutral): {len(kept_rows)}")

with open(output_csv, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)

print(f"Saved to {output_csv}")
