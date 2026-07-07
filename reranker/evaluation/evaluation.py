# evaluation.py
import json
import time
import re
import csv
import os
import argparse
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI
from chromadb.utils import embedding_functions

import sys
from pathlib import Path

import platform
import importlib.util

def is_torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None

def is_openvino_available() -> bool:
    return importlib.util.find_spec("openvino") is not None

def detect_backend_and_device():
    system = platform.system()  # "Darwin", "Windows", "Linux"

    torch = None
    if is_torch_available():
        import torch

    # macOS -> priorité à MPS
    if system == "Darwin":
        if torch is not None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return {"backend": "torch", "device": "mps"}
        return {"backend": "torch", "device": "cpu"}

    # Windows / Linux -> CUDA, puis OpenVINO, puis CPU
    if system in {"Windows", "Linux"}:
        if torch is not None and torch.cuda.is_available():
            return {"backend": "torch", "device": "cuda"}

        if is_openvino_available():
            return {"backend": "openvino", "device": ""}  # ou "AUTO" selon ton implémentation

        return {"backend": "torch", "device": "cpu"}

    # fallback ultime
    return {"backend": "torch", "device": "cpu"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from reranker.get_relating_document_bis import get_relating_documents_bis
from reranker.reranker_module import RerankConfig
from reranker.evaluation.overall_gain_and_drop_rank import write_rerank_movement_report

ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# ----------------------------
# Env helpers
# ----------------------------
def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


# ----------------------------
# Config
# ----------------------------
DEFAULT_CHROMA_INDEX = os.getenv("DEFAULT_CHROMA_INDEX", "chroma_index")
DEFAULT_BASELINE_JSONL = PROJECT_ROOT / "data" / "reranker" / "baseline" / "generations.jsonl"

CONFIG = {
    # Input / source of baseline retrieval results
    "questions_jsonl": DEFAULT_BASELINE_JSONL,
    "run_file_path": DEFAULT_BASELINE_JSONL,

    # Retrieval / embedding
    "generate_embeddings": False,
    "chroma_persist_mode": "per_index",
    "chroma_base_path": PROJECT_ROOT / "workspace" / "chroma",
    "chroma_index": DEFAULT_CHROMA_INDEX,
    "filters": None,

    # Retrieval / rerank
    "k_value": env_int("EVALUATION_K_VALUE", 10),
    "retrieve_k": env_int("EVALUATION_RETRIEVE_K", 30),

    # Evaluation / judge
    "evaluation_judge_model": env_str("EVALUATION_JUDGE_MODEL", "gpt-4.1-mini"),
    "evaluation_enable_dedup": env_bool("EVALUATION_ENABLE_DEDUP", True),
    "warmup_runs": env_int("EVALUATION_WARMUP_RUNS", 1),

    # Logging
    "verbose": env_bool("EVALUATION_VERBOSE", False),
    "progress_postfix": env_bool("EVALUATION_PROGRESS_POSTFIX", True),

    # Reranker config
    "rerank_config": {
        "model_name": env_str("EVALUATION_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        "batch_size": env_int("EVALUATION_RERANK_BATCH_SIZE", 8),
        # "device": env_str("EVALUATION_RERANK_DEVICE", "") or None,
        # "backend": env_str("EVALUATION_RERANK_BACKEND", "torch"),
        "window_chars": env_int("EVALUATION_RERANK_WINDOW_CHARS", 1200),
        "window_overlap": env_int("EVALUATION_RERANK_WINDOW_OVERLAP", 200),
        "min_window_chars": env_int("EVALUATION_RERANK_MIN_WINDOW_CHARS", 50),
        "agg": env_str("EVALUATION_RERANK_AGG", "softmax_topk"),
        "topk_windows": env_int("EVALUATION_RERANK_TOPK_WINDOWS", 10),
        "softmax_temp": env_float("EVALUATION_RERANK_SOFTMAX_TEMP", 0.25),
        "normalize_whitespace": env_bool("EVALUATION_RERANK_NORMALIZE_WHITESPACE", False),
        "add_scores_to_metadata": env_bool("EVALUATION_RERANK_ADD_SCORES_TO_METADATA", True),
        "use_metadata": env_bool("EVALUATION_RERANK_USE_METADATA", True),
        "verbose": env_bool("EVALUATION_RERANK_VERBOSE", False),
    },

    # Generation
    "llm_provider": env_str("EVALUATION_LLM_PROVIDER", "openai"),  # "openai" or "gemini"
    "temperature": 0,

    # OpenAI
    "openai_model": env_str("EVALUATION_OPENAI_MODEL", "gpt-4.1-nano"),
    "openai_embedding_model": env_str("EVALUATION_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),

    # Gemini
    "gemini_model": env_str("EVALUATION_GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini_api_version": env_str("EVALUATION_GEMINI_API_VERSION", "v1"),

    # Output
    "runs_dir": PROJECT_ROOT / "data" / "reranker" / "runs_reranker_config",
    "export_dir": PROJECT_ROOT / "data" / "reranker" / "runs",
    "export_file": None,

    # System prompt
    "system_prompt": (
        "You are a retrieval-augmented question answering system. "
        "You must answer using only the provided context and no prior or external knowledge. "
        "You may combine information from multiple documents in the context to derive the answer. "
        "Do not introduce new facts. "
        "If the answer cannot be clearly and directly derived from the provided context, "
        "respond exactly with: I don't know (for English questions) or je ne sais pas (for French questions). "
        "Your answers must be concise, factual, and limited to 1–2 sentences."
    ),
}


# ----------------------------
# Utility
# ----------------------------
def sanitize_filename(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r'[<>:"|?*]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name


def build_export_generation_path() -> Path:
    export_file = CONFIG.get("export_file")
    if export_file:
        return Path(export_file)

    model_name = CONFIG["rerank_config"]["model_name"]
    safe_model_name = sanitize_filename(model_name)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(CONFIG["export_dir"])
    return export_dir / f"{safe_model_name}_{date_str}.jsonl"


def _find_latest_no_rerank_run(runs_dir: Path | str) -> Optional[Path]:
    runs_dir = Path(runs_dir)
    candidates = list(runs_dir.glob("*/no_rerank/generations.jsonl"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_generations_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def timed_input(prompt: str, timeout_s: float, default: str = "") -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()

    if os.name == "nt":
        import msvcrt
        start = time.time()
        buf = []
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    return "".join(buf).strip()
                if ch == "\b":
                    if buf:
                        buf.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                else:
                    buf.append(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()

            if (time.time() - start) >= timeout_s:
                sys.stdout.write("\n")
                return default
    else:
        import select
        rlist, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if rlist:
            return sys.stdin.readline().strip()
        return default


def load_questions_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(p: Path | str):
    Path(p).mkdir(parents=True, exist_ok=True)


def dump_json(path: Path | str, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def dump_jsonl(path: Path | str, rows: List[Dict[str, Any]]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Embeddings
# ----------------------------
def make_embedding_fn():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for embeddings (Chroma OpenAIEmbeddingFunction). "
            "Either set OPENAI_API_KEY, or change your embedding function to another provider."
        )

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=CONFIG["openai_embedding_model"],
    )

    def _embedding_fn(texts: List[str]) -> List[List[float]]:
        return openai_ef(texts)

    return _embedding_fn


# ----------------------------
# Generation
# ----------------------------
def build_context(docs: List[str], metas: List[dict], max_chars: int = 12000) -> str:
    parts = []
    total = 0
    for i, d in enumerate(docs):
        if not d:
            continue
        header = f"Document {i+1}:\n"
        chunk = header + d.strip()
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def generate_answer_openai(client: OpenAI, question: str, docs: List[str], metas: List[dict]) -> Tuple[str, float]:
    context = build_context(docs, metas)
    messages = [
        {"role": "system", "content": CONFIG["system_prompt"]},
        {"role": "user", "content": f"Question:\n{question}\n\nDocuments:\n{context}"},
    ]
    t0 = time.time()
    resp = client.chat.completions.create(
        model=CONFIG["openai_model"],
        messages=messages,
        temperature=CONFIG["temperature"],
    )
    latency_ms = (time.time() - t0) * 1000
    answer = resp.choices[0].message.content.strip()
    return answer, latency_ms


def generate_answer_gemini(question: str, docs: List[str], metas: List[dict]) -> Tuple[str, float]:
    if genai is None or types is None:
        raise RuntimeError("Gemini SDK not installed. Run: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment (.env).")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version=CONFIG.get("gemini_api_version", "v1")),
    )

    context = build_context(docs, metas)
    user_text = f"Question:\n{question}\n\nDocuments:\n{context}"

    t0 = time.time()
    resp = client.models.generate_content(
        model=CONFIG["gemini_model"],
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=CONFIG["system_prompt"],
            temperature=CONFIG["temperature"],
        ),
    )
    latency_ms = (time.time() - t0) * 1000
    return (resp.text or "").strip(), latency_ms


def generate_answer(client_openai: Optional[OpenAI], question: str, docs: List[str], metas: List[dict]) -> Tuple[str, float]:
    provider = (CONFIG.get("llm_provider") or "openai").lower()
    if provider == "openai":
        if client_openai is None:
            raise RuntimeError("OpenAI client is required when llm_provider='openai'.")
        return generate_answer_openai(client_openai, question, docs, metas)
    if provider == "gemini":
        return generate_answer_gemini(question, docs, metas)
    raise ValueError(f"Unknown llm_provider={provider!r} (expected 'openai' or 'gemini').")


# ----------------------------
# Metrics
# ----------------------------
_IDK_FR = "je ne sais pas"
_IDK_EN = "i don't know"


def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    return s


def is_idk(ans: str) -> bool:
    a = normalize(ans)
    return a == _IDK_FR or a == _IDK_EN


def exact_match(pred: str, ref: str) -> int:
    return int(normalize(pred) == normalize(ref))


def contains_match(pred: str, ref: str) -> int:
    p = normalize(pred)
    r = normalize(ref)
    return int(r in p or p in r)


def token_f1(pred: str, ref: str) -> float:
    p = normalize(pred).split()
    r = normalize(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    from collections import Counter
    pc = Counter(p)
    rc = Counter(r)
    common = sum((pc & rc).values())
    if common == 0:
        return 0.0
    precision = common / max(1, len(p))
    recall = common / max(1, len(r))
    return 2 * precision * recall / max(1e-12, precision + recall)


# ----------------------------
# Warmup
# ----------------------------
def warmup_pipeline(
    embedding_fn,
    questions: List[Dict[str, Any]],
    *,
    use_reranker: bool,
    warmup_runs: int,
) -> None:
    if warmup_runs <= 0 or not questions:
        return

    rerank_cfg = RerankConfig(**CONFIG["rerank_config"]) if CONFIG["rerank_config"] else None

    warmup_question = questions[0]
    qid = warmup_question["question_id"]
    q = warmup_question["question"]

    print("\n[WARMUP]")
    print(f"warmup_runs={warmup_runs}")
    print(f"question_id={qid}")
    print(f"question={q}")

    for i in range(warmup_runs):
        t0 = time.time()

        _ = get_relating_documents_bis(
            query=q,
            filters=CONFIG["filters"],
            chroma_base_path=CONFIG["chroma_base_path"],
            embedding_fn=embedding_fn,
            k_value=CONFIG["k_value"],
            index=CONFIG["chroma_index"],
            use_reranker=use_reranker,
            rerank_config=rerank_cfg,
            n_candidates=CONFIG["retrieve_k"] if use_reranker else None,
            chroma_persist_mode=CONFIG["chroma_persist_mode"],
            source="run_file",
            run_file_path=CONFIG["run_file_path"],
            question_id=qid,
            generate_embeddings=CONFIG["generate_embeddings"],
            verbose=CONFIG["verbose"],
            return_timing=False,
        )

        dt_ms = (time.time() - t0) * 1000
        print(f"[WARMUP {i + 1}/{warmup_runs}] done in {dt_ms:.2f} ms")


# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(
    client: Optional[OpenAI],
    embedding_fn,
    questions: List[Dict[str, Any]],
    *,
    use_reranker: bool,
    run_dir: Path | str,
) -> List[Dict[str, Any]]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir)

    out = []
    rerank_cfg = RerankConfig(**CONFIG["rerank_config"]) if CONFIG["rerank_config"] else None

    if use_reranker:
        print("\n[RERANK CONFIG - EFFECTIVE]")
        print(json.dumps(asdict(rerank_cfg), ensure_ascii=False, indent=2) if rerank_cfg else "rerank_cfg=None")

    progress = tqdm(
        questions,
        desc=("with_rerank" if use_reranker else "no_rerank"),
        unit="q",
        dynamic_ncols=True,
        leave=True,
    )

    for row in progress:
        qid = row["question_id"]
        q = row["question"]

        res = get_relating_documents_bis(
            query=q,
            filters=CONFIG["filters"],
            chroma_base_path=CONFIG["chroma_base_path"],
            embedding_fn=embedding_fn,
            k_value=CONFIG["k_value"],
            index=CONFIG["chroma_index"],
            use_reranker=use_reranker,
            rerank_config=rerank_cfg,
            n_candidates=CONFIG["retrieve_k"] if use_reranker else None,
            chroma_persist_mode=CONFIG["chroma_persist_mode"],
            source="run_file",
            run_file_path=CONFIG["run_file_path"],
            question_id=qid,
            generate_embeddings=CONFIG["generate_embeddings"],
            verbose=CONFIG["verbose"],
            return_timing=True,
        )

        retrieval_timings = res.get("timings", {}) or {}

        retrieval_ms = 1000.0 * retrieval_timings.get("retrieval_total", 0.0)
        rerank_ms = 1000.0 * retrieval_timings.get("rerank", 0.0)
        run_file_load_ms = 1000.0 * retrieval_timings.get("run_file_load", 0.0)
        filter_document_text_ms = 1000.0 * retrieval_timings.get("filter_document_text", 0.0)
        dedup_ms = 1000.0 * retrieval_timings.get("dedup", 0.0)
        build_final_order_ms = 1000.0 * retrieval_timings.get("build_final_order", 0.0)
        construct_output_ms = 1000.0 * retrieval_timings.get("construct_output", 0.0)
        sort_without_rerank_ms = 1000.0 * retrieval_timings.get("sort_without_rerank", 0.0)

        ids = res.get("ids", [[]])[0] or []
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []

        non_empty = sum(1 for d in docs if (d or "").strip())

        if CONFIG["verbose"]:
            tqdm.write(f"\n[{qid}] Retrieved docs: total={len(docs)} non_empty={non_empty}")

            for i in range(min(3, len(docs))):
                preview = (docs[i] or "").replace("\n", " ")[:200]

                meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
                meta_keys = list(meta.keys())[:10]

                meta_debug = {
                    "id": meta.get("id"),
                    "year": meta.get("year") or meta.get("Year"),
                    "date": meta.get("date") or meta.get("Date"),
                    "topic": meta.get("topic") or meta.get("Topic"),
                    "organization": meta.get("organization") or meta.get("Organization"),
                    "meta_keys": meta_keys,
                }

                tqdm.write(f"  - doc[{i}] meta_debug={meta_debug} preview='{preview}…'")

        if CONFIG["generate_embeddings"]:
            ans, gen_ms = generate_answer(client, q, docs, metas)
        else:
            ans = ""
            gen_ms = 0.0

        total_ms = retrieval_ms + gen_ms

        if CONFIG["verbose"]:
            tqdm.write(
                f"  -> timings: retrieval={retrieval_ms:.2f} ms | "
                f"rerank={rerank_ms:.2f} ms | "
                f"generation={gen_ms:.2f} ms | "
                f"total={total_ms:.2f} ms"
            )
        elif CONFIG["progress_postfix"]:
            progress.set_postfix(
                qid=qid,
                docs=len(docs),
                non_empty=non_empty,
                rerank_ms=f"{rerank_ms:.0f}",
                total_ms=f"{total_ms:.0f}",
            )

        record = {
            "question_id": qid,
            "question": q,
            "generated_answer": ans,
            "retrieved_chunks": [
                {
                    "id": ids[i] if i < len(ids) else None,
                    "text": docs[i] if i < len(docs) else None,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                    "rank": i,
                }
                for i in range(min(len(docs), CONFIG["k_value"]))
            ],
            "method_metadata": {
                "use_reranker": use_reranker,
                "k_value": CONFIG["k_value"],
                "retrieve_k": CONFIG["retrieve_k"] if use_reranker else None,
                "chroma_index": CONFIG["chroma_index"],
                "llm_provider": CONFIG.get("llm_provider"),
                "openai_model": CONFIG.get("openai_model"),
                "gemini_model": CONFIG.get("gemini_model"),
            },
            "latency_ms": retrieval_ms + gen_ms,
            "latency_breakdown_ms": {
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "run_file_load_ms": run_file_load_ms,
                "filter_document_text_ms": filter_document_text_ms,
                "dedup_ms": dedup_ms,
                "build_final_order_ms": build_final_order_ms,
                "construct_output_ms": construct_output_ms,
                "sort_without_rerank_ms": sort_without_rerank_ms,
                "generation_ms": gen_ms,
            },
        }
        out.append(record)

    generations_path = run_dir / "generations.jsonl"
    dump_jsonl(generations_path, out)

    if use_reranker:
        # Export principal :
        # - si --output est donné, écrit dans ce fichier
        # - sinon, écrit dans data/reranker/runs/<model>_<timestamp>.jsonl
        export_path = build_export_generation_path()
        ensure_dir(export_path.parent)
        dump_jsonl(export_path, out)
        print(f"Exported generations to: {export_path}")

        # Copie historique toujours conservée dans data/reranker/runs
        model_name = CONFIG["rerank_config"]["model_name"]
        safe_model_name = sanitize_filename(model_name)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        legacy_export_path = Path(CONFIG["export_dir"]) / f"{safe_model_name}_{date_str}.jsonl"

        # Évite d'écrire deux fois le même fichier si --output n'est pas utilisé
        if legacy_export_path.resolve() != export_path.resolve():
            ensure_dir(legacy_export_path.parent)
            dump_jsonl(legacy_export_path, out)
            print(f"Also exported legacy runs copy to: {legacy_export_path}")
    return out


# ----------------------------
# Evaluate
# ----------------------------
def evaluate_runs(
    questions: List[Dict[str, Any]],
    generations: List[Dict[str, Any]],
    out_csv_path: Path | str,
) -> Dict[str, Any]:
    qmap = {q["question_id"]: q for q in questions}
    rows = []

    for g in generations:
        qid = g["question_id"]
        question_row = qmap[qid]
        ref = (
            question_row.get("reference")
            or question_row.get("gold_answer")
            or question_row.get("answer")
            or question_row.get("expected_answer")
            or ""
        )
        pred = g.get("generated_answer", "")

        rows.append({
            "question_id": qid,
            "exact_match": exact_match(pred, ref),
            "contains": contains_match(pred, ref),
            "token_f1": token_f1(pred, ref),
            "pred_is_idk": int(is_idk(pred)),
            "latency_ms": g.get("latency_ms", None),
            "retrieval_ms": g.get("latency_breakdown_ms", {}).get("retrieval_ms", None),
            "rerank_ms": g.get("latency_breakdown_ms", {}).get("rerank_ms", None),
            "generation_ms": g.get("latency_breakdown_ms", {}).get("generation_ms", None),
        })

    out_csv_path = Path(out_csv_path)
    ensure_dir(out_csv_path.parent)

    if rows:
        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            f.write("")

    def avg(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "n": len(rows),
        "exact_match": avg("exact_match"),
        "contains": avg("contains"),
        "token_f1": avg("token_f1"),
        "pred_is_idk_rate": avg("pred_is_idk"),
        "latency_ms_avg": avg("latency_ms"),
        "retrieval_ms_avg": avg("retrieval_ms"),
        "rerank_ms_avg": avg("rerank_ms"),
        "generation_ms_avg": avg("generation_ms"),
    }


# ----------------------------
# Args
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Lance l'évaluation du reranker avec un chemin d'entrée et un chemin d'export optionnels."
    )

    parser.add_argument(
        "--input",
        "--input-jsonl",
        dest="input",
        type=str,
        default=None,
        help=(
            "Fichier JSONL d'entrée contenant les questions et retrieved_chunks. "
            "Si fourni, il remplace CONFIG['questions_jsonl'] et CONFIG['run_file_path']."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Chemin de sortie. Si le chemin se termine par .jsonl, "
            "le script écrit exactement dans ce fichier. Sinon, il est "
            "traité comme un dossier d'export."
        ),
    )

    return parser.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    if args.input is not None and args.input.strip() != "":
        input_path = Path(args.input)

        if not input_path.exists():
            raise FileNotFoundError(f"Input JSONL introuvable : {input_path}")

        CONFIG["questions_jsonl"] = input_path
        CONFIG["run_file_path"] = input_path

    runtime = detect_backend_and_device()

    CONFIG["rerank_config"]["backend"] = runtime["backend"]
    CONFIG["rerank_config"]["device"] = runtime["device"]

    if args.output is not None and args.output.strip() != "":
        output_path = Path(args.output)

        if output_path.suffix.lower() == ".jsonl":
            CONFIG["export_file"] = output_path
        else:
            CONFIG["export_dir"] = output_path

    print("\n[EVALUATION CONFIG - EFFECTIVE]")
    print(json.dumps({
        "generate_embeddings": CONFIG["generate_embeddings"],
        "questions_jsonl": str(CONFIG["questions_jsonl"]),
        "run_file_path": str(CONFIG["run_file_path"]),
        "chroma_persist_mode": CONFIG["chroma_persist_mode"],
        "chroma_base_path": str(CONFIG["chroma_base_path"]),
        "chroma_index": CONFIG["chroma_index"],
        "filters": CONFIG["filters"],
        "k_value": CONFIG["k_value"],
        "retrieve_k": CONFIG["retrieve_k"],
        "rerank_config": CONFIG["rerank_config"],
        "llm_provider": CONFIG["llm_provider"],
        "temperature": CONFIG["temperature"],
        "openai_model": CONFIG["openai_model"],
        "openai_embedding_model": CONFIG["openai_embedding_model"],
        "gemini_model": CONFIG["gemini_model"],
        "gemini_api_version": CONFIG["gemini_api_version"],
        "runs_dir": str(CONFIG["runs_dir"]),
        "export_dir": str(CONFIG["export_dir"]),
        "export_file": str(CONFIG["export_file"]) if CONFIG["export_file"] else None,
    }, ensure_ascii=False, indent=2))

    # API checks
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment (.env). It is required for embeddings.")

    if CONFIG["generate_embeddings"]:
        provider = (CONFIG.get("llm_provider") or "openai").lower()
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("Set GEMINI_API_KEY in your environment.")

    embedding_fn = make_embedding_fn()

    questions = load_questions_jsonl(CONFIG["questions_jsonl"])
    print(f"Loaded {len(questions)} questions from {CONFIG['questions_jsonl']}")

    run_id = f"rerank_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = Path(CONFIG["runs_dir"]) / run_id
    yes_dir = base_dir / "with_rerank"

    ensure_dir(base_dir)
    dump_json(base_dir / "config.json", CONFIG)

    client = None
    if CONFIG["generate_embeddings"] and CONFIG["llm_provider"].lower() == "openai":
        client = OpenAI()

    warmup_pipeline(
        embedding_fn,
        questions,
        use_reranker=True,
        warmup_runs=CONFIG["warmup_runs"],
    )

    gens_yes = run_pipeline(client, embedding_fn, questions, use_reranker=True, run_dir=yes_dir)
    metrics_yes = evaluate_runs(questions, gens_yes, yes_dir / "per_question_metrics.csv")

    dump_json(yes_dir / "summary_metrics.json", metrics_yes)

    baseline_jsonl = Path(CONFIG["run_file_path"])
    reranked_jsonl = yes_dir / "generations.jsonl"

    movement_out = base_dir / "rerank_movement_report.json"
    write_rerank_movement_report(
        baseline_jsonl=baseline_jsonl,
        reranked_jsonl=reranked_jsonl,
        out_path=movement_out,
        top_k=CONFIG["k_value"],
    )
    print(f"\nSaved rerank movement report to: {movement_out}")
    print(f"Saved summary metrics to: {yes_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()