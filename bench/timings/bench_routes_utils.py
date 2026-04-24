# bench_routes_utils.py
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

import requests
from dotenv import load_dotenv

''' Example command
    python .\plot_benckmark_from_csv.py --csv .\mac_mini_timming\benchmark_results_macmini_miniLm_multilingual_no-stream_macmini.csv --output .\mac_mini_timming\benchmarkoutput_csv_MiniLM_no-stream_597.png --xmax 4.5 --edge-label-side left --alternate-label
'''
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT /".." / ".env"
print(ENV_PATH)
load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_CHROMA_INDEX = os.getenv("DEFAULT_CHROMA_INDEX", "chroma_index")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost")
DEFAULT_ENDPOINT = PUBLIC_BASE_URL + "/benchmark_pipeline"
DEFAULT_CSV = None
DEFAULT_ATTEMPTS = 1
DEFAULT_TIMEOUT = 3600

ENV_PATH = PROJECT_ROOT / ".." / ".env"




EXPECTED_COLUMNS = [
    "batch_id",
    "batch_started_at_utc",
    "label",
    "question_index",
    "run_number",
    "query",
    "rag_query",
    "model",
    "reranker_model",
    "llm_mode",
    "use_reranker",
    "use_agent_prompt",
    "k",
    "n_candidates",
    "index",
    "documents_count",
    "answer",
    "summary_json",
    "raw_response_json",
]


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


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


def env_on_off(name: str, default: str) -> str:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    value = value.strip().lower()
    if value in {"on", "true", "1", "yes"}:
        return "on"
    if value in {"off", "false", "0", "no"}:
        return "off"
    return default


def env_bool_from_on_off(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    value = value.strip().lower()
    if value in {"on", "true", "1", "yes"}:
        return True
    if value in {"off", "false", "0", "no"}:
        return False
    return default


def make_timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_default_csv_path() -> Path:
    return PROJECT_ROOT / "data" / "evaluation" / f"benchmark_timing_results_{make_timestamp_for_filename()}.csv"

def timing_key_order(key: str) -> tuple[int, str]:
    ordered = [
        # Retrieval: étapes internes
        "timings_ms.run_file_load_ms",
        "timings_ms.run_file_doc_embeddings_ms",
        "timings_ms.chroma_prepare_ms",
        "timings_ms.query_embedding_ms",
        "timings_ms.chroma_query_ms",
        "timings_ms.normalize_results_ms",
        "timings_ms.filter_document_text_ms",
        "timings_ms.sort_without_rerank_ms",
        "timings_ms.dedup_ms",
        "timings_ms.dedup_method_ms",
        "timings_ms.rerank_ms",
        "timings_ms.build_final_order_ms",
        "timings_ms.construct_output_ms",

        # Retrieval: agrégats
        "timings_ms.retrieval_total_ms",
        "timings_ms.retrieval_wrapper_ms",

        # LLM: étapes internes / stream
        "timings_ms.llm_create_call_ms",
        "timings_ms.llm_total_ms",
        "timings_ms.llm_first_any_chunk_ms",
        "timings_ms.llm_first_text_chunk_ms",
        "timings_ms.llm_last_chunk_ms",
        "timings_ms.llm_stream_total_ms",

        # Fin de pipeline
        "timings_ms.response_build_ms",
        "timings_ms.total_ms",
    ]

    try:
        return (ordered.index(key), key)
    except ValueError:
        return (9999, key)



def parse_json_arg(value: Optional[str], default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide: {value}\n{e}")


def str_to_bool_like_on_off(value: str) -> bool:
    value = value.strip().lower()
    if value in {"on", "true", "1", "yes"}:
        return True
    if value in {"off", "false", "0", "no"}:
        return False
    raise ValueError(f"Valeur invalide: {value}. Utilise on/off.")


def merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(extra)
    return out


def run_client_side_warmup(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int,
    warmup_question: str,
    warmup_runs: int,
    warmup_question_id: Optional[str] = None,
) -> None:
    if warmup_runs <= 0:
        return

    warmup_payload = dict(payload)
    warmup_payload["query"] = warmup_question

    if warmup_payload.get("source") == "run_file":
        if not warmup_question_id:
            raise ValueError('warmup en source="run_file" requiert un warmup_question_id')
        warmup_payload["question_id"] = warmup_question_id

    print("\n===== WARMUP CLIENT =====")
    print(f"warmup_runs        : {warmup_runs}")
    print(f"warmup_query       : {warmup_question}")
    print(f"warmup_question_id : {warmup_question_id}")

    for i in range(warmup_runs):
        print(f"[WARMUP {i + 1}/{warmup_runs}] START")
        response_json = post_benchmark(endpoint, warmup_payload, timeout=timeout)
        print(f"[WARMUP {i + 1}/{warmup_runs}] DONE")

        runs = response_json.get("runs", []) or []
        if runs:
            answer = runs[0].get("answer")
            if answer:
                print(f"[WARMUP {i + 1}/{warmup_runs}] answer:")
                print(answer)


def load_questions(single_query: Optional[str], questions_file: Optional[str]) -> List[Dict[str, Any]]:
    if single_query and questions_file:
        raise ValueError("Utilise soit --query, soit --questions-file, pas les deux.")

    if single_query:
        q = single_query.strip()
        if not q:
            raise ValueError("--query est vide.")
        return [{"question": q, "question_id": None}]

    if questions_file:
        if not os.path.exists(questions_file):
            raise ValueError(f"Fichier introuvable: {questions_file}")

        items: List[Dict[str, Any]] = []

        with open(questions_file, "r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSON invalide dans {questions_file} à la ligne {lineno}: {e}\n"
                        f"Ligne: {raw_line.rstrip()}"
                    ) from e

                question = (rec.get("question") or "").strip()
                question_id = rec.get("question_id")

                if not question:
                    raise ValueError(
                        f"Ligne {lineno}: champ 'question' manquant ou vide."
                    )

                if not question_id:
                    raise ValueError(
                        f"Ligne {lineno}: champ 'question_id' manquant ou vide."
                    )

                items.append({
                    "question": question,
                    "question_id": question_id,
                })

        if not items:
            raise ValueError("Aucune question valide trouvée dans le fichier.")

        return items

    raise ValueError("Il faut fournir soit --query, soit --questions-file.")


def build_base_payload_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "use_agent_prompt": False,
        "use_reranker": str_to_bool_like_on_off(args.reranker),
        "llm_mode": args.llm_mode,
        "model": args.model,
        "index": args.index,
        "k": args.k,
        "n_candidates": args.n_candidates,
        "include_answer": True,
    }

    if args.filter_document_text is not None:
        payload["filter_document_text"] = args.filter_document_text

    if args.source is not None:
        payload["source"] = args.source

    if args.run_file_path is not None:
        payload["run_file_path"] = os.path.abspath(args.run_file_path)

    if args.question_id is not None:
        payload["question_id"] = args.question_id

    filters = parse_json_arg(args.filters_json, default=None)
    if filters is not None:
        payload["filters"] = filters

    prompt = parse_json_arg(args.prompt_json, default=None)
    if prompt is not None:
        payload["prompt"] = prompt

    reranker_config = parse_json_arg(args.reranker_config_json, default={}) or {}
    if not isinstance(reranker_config, dict):
        raise ValueError("--reranker-config-json doit être un objet JSON")

    # Injecte tout le RerankConfig effectif, pas seulement model_name
    reranker_config["model_name"] = args.reranker_model
    reranker_config["batch_size"] = args.reranker_batch_size
    reranker_config["max_doc_chars"] = args.reranker_max_doc_chars
    reranker_config["window_chars"] = args.reranker_window_chars
    reranker_config["window_overlap"] = args.reranker_window_overlap
    reranker_config["min_window_chars"] = args.reranker_min_window_chars
    reranker_config["agg"] = args.reranker_agg
    reranker_config["topk_windows"] = args.reranker_topk_windows
    reranker_config["softmax_temp"] = args.reranker_softmax_temp
    reranker_config["normalize_whitespace"] = str_to_bool_like_on_off(args.reranker_normalize_whitespace)
    reranker_config["return_spans"] = str_to_bool_like_on_off(args.reranker_return_spans)
    reranker_config["add_scores_to_metadata"] = str_to_bool_like_on_off(args.reranker_add_scores_to_metadata)

    # NOUVEAU
    reranker_config["use_metadata"] = str_to_bool_like_on_off(args.reranker_use_metadata)
    reranker_config["metadata_max_chars"] = args.reranker_metadata_max_chars
    payload["reranker_config"] = reranker_config

    extra_payload = parse_json_arg(args.extra_payload_json, default=None)
    if extra_payload is not None:
        if not isinstance(extra_payload, dict):
            raise ValueError("--extra-payload-json doit être un objet JSON")
        payload = merge_dicts(payload, extra_payload)

    return payload


def build_aggregated_response_for_question(
    responses: List[Dict[str, Any]],
    question: str,
) -> Dict[str, Any]:
    all_runs = []
    configuration = {}

    for idx, response in enumerate(responses, start=1):
        configuration = response.get("configuration", {}) or configuration
        runs = response.get("runs", []) or []
        for run in runs:
            run = dict(run)
            run["run_number"] = idx
            all_runs.append(run)

    keys = sorted({k for run in all_runs for k in (run.get("timings_ms", {}) or {}).keys()})
    summary: Dict[str, Dict[str, float]] = {}

    for key in keys:
        values = []
        for run in all_runs:
            value = (run.get("timings_ms", {}) or {}).get(key)
            if value is None:
                continue
            values.append(float(value))

        if values:
            summary[key] = {
                "avg_ms": round(sum(values) / len(values), 2),
                "min_ms": round(min(values), 2),
                "max_ms": round(max(values), 2),
            }

    configuration = dict(configuration)
    configuration["query"] = question
    configuration["repeat"] = len(all_runs)

    return {
        "configuration": configuration,
        "summary": summary,
        "runs": all_runs,
    }


def post_benchmark(endpoint: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def merge_fieldnames(existing_csv: str | Path, rows: List[Dict[str, Any]]) -> List[str]:
    existing_csv = Path(existing_csv)

    fieldnames = list(EXPECTED_COLUMNS)
    existing_fields: List[str] = []

    if existing_csv.exists() and existing_csv.stat().st_size > 0:
        with existing_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                existing_fields = list(reader.fieldnames)

    seen = set(fieldnames)

    for name in existing_fields:
        if name not in seen:
            fieldnames.append(name)
            seen.add(name)

    for row in rows:
        for name in row.keys():
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)

    return fieldnames


def append_rows_to_csv(csv_path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = merge_fieldnames(csv_path, rows)

    needs_rewrite = False
    existing_rows: List[Dict[str, Any]] = []

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            if list(old_fields) != list(fieldnames):
                needs_rewrite = True
                existing_rows = list(reader)

    if needs_rewrite:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        return

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_rows_from_response(
    response_json: Dict[str, Any],
    *,
    batch_id: str,
    batch_started_at_utc: str,
    label: str,
    question_index: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    configuration = response_json.get("configuration", {}) or {}
    summary = response_json.get("summary", {}) or {}
    runs = response_json.get("runs", []) or []

    for run in runs:
        row: Dict[str, Any] = {
            "batch_id": batch_id,
            "batch_started_at_utc": batch_started_at_utc,
            "label": label,
            "question_index": question_index,
            "run_number": run.get("run_number"),
            "query": run.get("query") or configuration.get("query"),
            "rag_query": run.get("rag_query"),
            "model": configuration.get("model"),
            "reranker_model": configuration.get("reranker_model") or (configuration.get("reranker_config", {}) or {}).get("model_name"),
            "llm_mode": run.get("llm_mode") or configuration.get("llm_mode"),
            "use_reranker": configuration.get("use_reranker"),
            "use_agent_prompt": configuration.get("use_agent_prompt"),
            "k": run.get("k") or configuration.get("k"),
            "n_candidates": configuration.get("n_candidates"),
            "index": configuration.get("index"),
            "documents_count": run.get("documents_count"),
            "answer": run.get("answer"),
            "summary_json": json.dumps(summary, ensure_ascii=False),
            "raw_response_json": json.dumps(run, ensure_ascii=False),
        }

        timings = run.get("timings_ms", {}) or {}
        stream_stats = run.get("stream_stats", {}) or {}
        retrieval_stats = run.get("retrieval_stats", {}) or {}
        top_metadata_preview = run.get("top_metadata_preview", []) or []

        row.update({f"timings_ms.{k}": v for k, v in timings.items()})
        row.update({f"stream_stats.{k}": v for k, v in stream_stats.items()})
        row.update({f"retrieval_stats.{k}": v for k, v in retrieval_stats.items()})
        row["top_metadata_preview"] = json.dumps(top_metadata_preview, ensure_ascii=False)

        rows.append(row)

    return rows


def print_terminal_summary(response_json: Dict[str, Any], title: str) -> None:
    configuration = response_json.get("configuration", {}) or {}
    summary = response_json.get("summary", {}) or {}
    runs = response_json.get("runs", []) or []

    print(f"\n===== {title} =====")
    print(f"query            : {configuration.get('query')}")
    print(f"model            : {configuration.get('model')}")
    print(f"reranker_model   : {configuration.get('reranker_model') or (configuration.get('reranker_config', {}) or {}).get('model_name')}")
    print(f"llm_mode         : {configuration.get('llm_mode')}")
    print(f"use_reranker     : {configuration.get('use_reranker')}")
    print(f"k                : {configuration.get('k')}")
    print(f"n_candidates     : {configuration.get('n_candidates')}")
    print(f"index            : {configuration.get('index')}")
    print(f"repeat           : {configuration.get('repeat')}")
    print(f"warmup           : {configuration.get('warmup')}")

    if summary:
        print("\nTemps moyens par étape (ms):")

        prefixed_summary = {f"timings_ms.{k}": v for k, v in summary.items()}

        for full_key in sorted(prefixed_summary.keys(), key=timing_key_order):
            item = prefixed_summary.get(full_key, {}) or {}
            display_key = full_key.replace("timings_ms.", "", 1)

            print(
                f"{display_key:35s} "
                f"avg={str(item.get('avg_ms')):>10} | "
                f"min={str(item.get('min_ms')):>10} | "
                f"max={str(item.get('max_ms')):>10}"
            )
    else:
        print("\nAucun résumé retourné par l'API.")

    if runs:
        print("\nRéponse(s):")
        for run in runs:
            run_number = run.get("run_number")
            answer = run.get("answer")

            if answer is None:
                print(f"\n[run {run_number}] Réponse absente.")
            else:
                print(f"\n[run {run_number}]")
                print(answer)


def compute_global_summary_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    timing_columns = sorted({
        key for row in rows for key in row.keys()
        if key.startswith("timings_ms.")
    })

    summary: Dict[str, Dict[str, float]] = {}

    for col in timing_columns:
        values: List[float] = []
        for row in rows:
            value = row.get(col)
            if value in (None, "", "None"):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue

        if not values:
            continue

        summary[col] = {
            "avg_ms": round(sum(values) / len(values), 2),
            "min_ms": round(min(values), 2),
            "max_ms": round(max(values), 2),
        }

    return summary


def print_global_summary(rows: List[Dict[str, Any]]) -> None:
    summary = compute_global_summary_from_rows(rows)

    if not summary:
        print("\nAucun timing global à afficher.")
        return

    print("\n===== RÉSUMÉ GLOBAL SUR TOUS LES RUNS =====")
    ordered_keys = [
        "timings_ms.retrieval_wrapper_ms",
        "timings_ms.run_file_load_ms",
        "timings_ms.run_file_doc_embeddings_ms",
        "timings_ms.chroma_prepare_ms",
        "timings_ms.query_embedding_ms",
        "timings_ms.chroma_query_ms",
        "timings_ms.normalize_results_ms",
        "timings_ms.filter_document_text_ms",
        "timings_ms.sort_without_rerank_ms",
        "timings_ms.dedup_ms",
        "timings_ms.dedup_method_ms",
        "timings_ms.rerank_ms",
        "timings_ms.build_final_order_ms",
        "timings_ms.construct_output_ms",
        "timings_ms.retrieval_total_ms",
        "timings_ms.llm_create_call_ms",
        "timings_ms.llm_total_ms",
        "timings_ms.llm_first_any_chunk_ms",
        "timings_ms.llm_first_text_chunk_ms",
        "timings_ms.llm_last_chunk_ms",
        "timings_ms.llm_stream_total_ms",
        "timings_ms.response_build_ms",
        "timings_ms.total_ms",
    ]

    for key in ordered_keys:
        if key not in summary:
            continue
        item = summary[key] or {}
        print(
            f"{key:40s} "
            f"avg={str(item.get('avg_ms')):>10} | "
            f"min={str(item.get('min_ms')):>10} | "
            f"max={str(item.get('max_ms')):>10}"
        )

    total_key = "timings_ms.total_ms"
    if total_key in summary:
        total = summary[total_key]
        print("\nTemps total global:")
        print(
            f"avg={total['avg_ms']} ms | "
            f"min={total['min_ms']} ms | "
            f"max={total['max_ms']} ms"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance une ou plusieurs questions sur /benchmark_pipeline, écrit les runs dans un CSV et affiche les temps moyens."
    )

    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="URL du endpoint benchmark_pipeline")
    parser.add_argument("--csv", default=None, help="Chemin du CSV de sortie")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS, help="Nombre d'essais par question")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout HTTP en secondes")

    parser.add_argument("--label", default=None, help="Label enregistré dans le CSV")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Nombre de warmups côté client")
    parser.add_argument("--warmup-query", default="What is the history of the university of Lausanne", help="Question utilisée pour le warmup")
    parser.add_argument("--warmup-scope", choices=["global", "per-question"], default="global", help="Warmup une seule fois avant tout le batch, ou avant chaque question")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--query", help="Question unique")
    group.add_argument("--questions-file", help="Fichier JSONL de questions", default=PROJECT_ROOT / "data" / "processed" / "questions.jsonl")

    parser.add_argument("--reranker", choices=["on", "off"], default=None)
    parser.add_argument("--reranker-model", default=None, help="Nom du modèle de reranker")
    parser.add_argument("--llm-mode", choices=["stream", "non_stream"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--index", default=DEFAULT_CHROMA_INDEX)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--n-candidates", type=int, default=None)

    parser.add_argument("--no-agent-prompt", action="store_true")
    parser.add_argument("--include-answer", action="store_true")

    parser.add_argument("--parsing", choices=["on", "off"], default=None)

    parser.add_argument("--filter-document-text", default=None)

    parser.add_argument("--source", choices=["chroma", "run_file"], default="chroma")
    parser.add_argument("--run-file-path", default=None)
    parser.add_argument("--question-id", default=None)

    parser.add_argument("--filters-json", default=None, help='Ex: \'{"year":{"$gte":1900}}\'')
    parser.add_argument("--prompt-json", default=None, help="Prompt custom en JSON")
    parser.add_argument("--reranker-config-json", default=None, help='Ex: \'{"window_chars":250,"window_overlap":80,"agg":"max","return_spans":false,"batch_size":16}\'')
    parser.add_argument("--extra-payload-json", default=None, help="Objet JSON fusionné dans le payload final")

    # Defaults bench depuis .env
    parser.add_argument("--reranker-batch-size", type=int, default=None)
    parser.add_argument("--reranker-max-doc-chars", type=int, default=None)
    parser.add_argument("--reranker-window-chars", type=int, default=None)
    parser.add_argument("--reranker-window-overlap", type=int, default=None)
    parser.add_argument("--reranker-min-window-chars", type=int, default=None)
    parser.add_argument("--reranker-agg", choices=["max", "mean_topk", "softmax_topk", "logsumexp_topk"], default=None)
    parser.add_argument("--reranker-topk-windows", type=int, default=None)
    parser.add_argument("--reranker-softmax-temp", type=float, default=None)
    parser.add_argument("--reranker-normalize-whitespace", choices=["on", "off"], default=None)
    parser.add_argument("--reranker-return-spans", choices=["on", "off"], default=None)
    parser.add_argument("--reranker-add-scores-to-metadata", choices=["on", "off"], default=None)

    args = parser.parse_args()

    args.reranker = args.reranker or env_on_off("BENCHMARK_RERANKER", "on")
    args.parsing = args.parsing or env_on_off("BENCHMARK_PARSING", "on")
    args.reranker_model = args.reranker_model or env_str(
        "BENCHMARK_RERANKER_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    args.llm_mode = args.llm_mode or env_str("BENCHMARK_LLM_MODE", "stream")
    args.model = args.model or env_str("BENCHMARK_MODEL", "gpt-4.1-mini")
    args.k = args.k if args.k is not None else env_int("BENCHMARK_K", 10)
    args.n_candidates = args.n_candidates if args.n_candidates is not None else env_int("BENCHMARK_N_CANDIDATES", 30)

    # Defaults reranker bench depuis .env
    args.reranker_batch_size = args.reranker_batch_size if args.reranker_batch_size is not None else env_int("BENCHMARK_RERANK_BATCH_SIZE", 16)
    args.reranker_max_doc_chars = args.reranker_max_doc_chars if args.reranker_max_doc_chars is not None else env_int("BENCHMARK_RERANK_MAX_DOC_CHARS", 20000)
    args.reranker_window_chars = args.reranker_window_chars if args.reranker_window_chars is not None else env_int("BENCHMARK_RERANK_WINDOW_CHARS", 5000)
    args.reranker_window_overlap = args.reranker_window_overlap if args.reranker_window_overlap is not None else env_int("BENCHMARK_RERANK_WINDOW_OVERLAP", 2000)
    args.reranker_min_window_chars = args.reranker_min_window_chars if args.reranker_min_window_chars is not None else env_int("BENCHMARK_RERANK_MIN_WINDOW_CHARS", 50)
    args.reranker_agg = args.reranker_agg or env_str("BENCHMARK_RERANK_AGG", "max")
    args.reranker_topk_windows = args.reranker_topk_windows if args.reranker_topk_windows is not None else env_int("BENCHMARK_RERANK_TOPK_WINDOWS", 1)
    args.reranker_softmax_temp = args.reranker_softmax_temp if args.reranker_softmax_temp is not None else env_float("BENCHMARK_RERANK_SOFTMAX_TEMP", 1.0)
    args.reranker_normalize_whitespace = args.reranker_normalize_whitespace or env_on_off("BENCHMARK_RERANK_NORMALIZE_WHITESPACE", "on")
    args.reranker_return_spans = args.reranker_return_spans or env_on_off("BENCHMARK_RERANK_RETURN_SPANS", "off")
    args.reranker_add_scores_to_metadata = args.reranker_add_scores_to_metadata or env_on_off("BENCHMARK_RERANK_ADD_SCORES_TO_METADATA", "on")

    if args.csv is None:
        args.csv = build_default_csv_path()
    else:
        args.csv = Path(args.csv)

    return args


def main() -> int:
    args = parse_args()

    if not args.query and not args.questions_file and not (args.source == "run_file" and args.run_file_path):
        print(
            "Erreur: il faut fournir soit --query, soit --questions-file, "
            "ou bien --source run_file avec --run-file-path.",
            file=sys.stderr,
        )
        return 1

    if args.attempts < 0:
        print("Erreur: --attempts ne peut pas être négatif.", file=sys.stderr)
        return 1

    if args.attempts == 0:
        print("Aucun essai lancé (attempts=0). Rien à faire.")
        return 0

    try:
        effective_questions_file = args.questions_file

        if effective_questions_file is None and args.source == "run_file" and args.run_file_path:
            effective_questions_file = args.run_file_path

        questions = load_questions(args.query, effective_questions_file)
        base_payload = build_base_payload_from_args(args)
        label = args.label or f"{args.model}-reranker-{args.reranker}"

        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        batch_started_at_utc = datetime.now(timezone.utc).isoformat()

        all_rows: List[Dict[str, Any]] = []

        print("===== ENVOI BENCHMARK =====")
        print(f"endpoint          : {args.endpoint}")
        print(f"label             : {label}")
        print(f"nombre questions  : {len(questions)}")
        print(f"attempts/question : {args.attempts}")
        print(f"warmup_runs       : {args.warmup_runs}")
        print(f"warmup_scope      : {args.warmup_scope}")

        if args.warmup_runs > 0 and args.warmup_scope == "global":
            warmup_item = questions[0]
            warmup_question = args.warmup_query or warmup_item["question"]
            warmup_question_id = warmup_item["question_id"]

            run_client_side_warmup(
                endpoint=args.endpoint,
                payload=base_payload,
                timeout=args.timeout,
                warmup_question=warmup_question,
                warmup_question_id=warmup_question_id,
                warmup_runs=args.warmup_runs,
            )

        for i, item in enumerate(tqdm(questions, desc="Questions", unit="q"), start=1):
            question = item["question"]
            question_id = item.get("question_id")

            print(f"\n--- Question {i}/{len(questions)} ---")
            print(question)
            print(f"question_id       : {question_id}")

            if args.warmup_runs > 0 and args.warmup_scope == "per-question":
                warmup_question = args.warmup_query or question
                warmup_question_id = question_id

                run_client_side_warmup(
                    endpoint=args.endpoint,
                    payload=base_payload,
                    timeout=args.timeout,
                    warmup_question=warmup_question,
                    warmup_question_id=warmup_question_id,
                    warmup_runs=args.warmup_runs,
                )

            question_rows: List[Dict[str, Any]] = []
            question_responses: List[Dict[str, Any]] = []

            for attempt_idx in range(1, args.attempts + 1):
                payload = dict(base_payload)
                payload["query"] = question

                if payload.get("source") == "run_file":
                    if not question_id:
                        raise ValueError('source="run_file" requiert un question_id par question dans le fichier JSONL')
                    payload["question_id"] = question_id

                print(f"[RUN {attempt_idx}/{args.attempts}] START")
                response_json = post_benchmark(args.endpoint, payload, timeout=args.timeout)
                print(f"[RUN {attempt_idx}/{args.attempts}] DONE")

                rows = build_rows_from_response(
                    response_json,
                    batch_id=batch_id,
                    batch_started_at_utc=batch_started_at_utc,
                    label=label,
                    question_index=i,
                )

                for row in rows:
                    row["run_number"] = attempt_idx

                runs = response_json.get("runs", []) or []
                if runs:
                    runs[0]["run_number"] = attempt_idx

                question_rows.extend(rows)
                question_responses.append(response_json)

            all_rows.extend(question_rows)

            if question_responses:
                print_terminal_summary(
                    build_aggregated_response_for_question(question_responses, question),
                    f"RÉSUMÉ QUESTION {i}"
                )

        append_rows_to_csv(args.csv, all_rows)

        print(f"\nCSV mis à jour: {args.csv}")
        print(f"{len(all_rows)} ligne(s) ajoutée(s)")
        print_global_summary(all_rows)

        return 0

    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text
        except Exception:
            pass
        print(f"Erreur HTTP: {e}\n{body}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())