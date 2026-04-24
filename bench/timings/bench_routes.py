# bench_routes.py
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any

from dotenv import load_dotenv
from flask import request, jsonify
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT /".." / ".env"

print(ENV_PATH)

load_dotenv(dotenv_path=ENV_PATH)


def init_benchmark_routes(
    app,
    *,
    get_relating_documents_bis,
    RerankConfig,
    CHROMA_BASE_PATH,
    openai_ef,
    google_ef
):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    env_chroma_base_path = os.getenv("CHROMA_BASE_PATH", CHROMA_BASE_PATH)
    env_default_chroma_index = os.getenv("DEFAULT_CHROMA_INDEX", "chroma_index")
    env_default_llm_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4.1-mini")
    env_openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    env_chroma_collection = os.getenv("CHROMA_COLLECTION", env_default_chroma_index)

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert in historical documents. "
        "Answer only from the provided documents. "
        "If the answer is not explicitly supported by the documents, say so clearly."
    )

    def _normalize_chroma_list(value):
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            return value[0]
        return value

    def _ms(seconds):
        if seconds is None:
            return None
        return round(seconds * 1000, 2)

    def _safe_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _build_rerank_config(data):
        cfg = data.get("reranker_config", {}) or {}
        return RerankConfig(
            model_name=cfg.get("model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
            batch_size=int(cfg.get("batch_size", 8)),
            device=cfg.get("device", None),
            max_doc_chars=int(cfg.get("max_doc_chars", 20000)),
            window_chars=int(cfg.get("window_chars", 1200)),
            window_overlap=int(cfg.get("window_overlap", 200)),
            min_window_chars=int(cfg.get("min_window_chars", 50)),
            agg=cfg.get("agg", "softmax_topk"),
            topk_windows=int(cfg.get("topk_windows", 3)),
            softmax_temp=float(cfg.get("softmax_temp", 0.25)),
            normalize_whitespace=_safe_bool(cfg.get("normalize_whitespace", False), False),
            return_spans=_safe_bool(cfg.get("return_spans", False), False),
            add_scores_to_metadata=_safe_bool(cfg.get("add_scores_to_metadata", True), True),
            use_metadata=_safe_bool(cfg.get("use_metadata", False), False),
            metadata_max_chars=int(cfg.get("metadata_max_chars", 500)),
        )

    def _get_query_embedding_fn_for_index(index_name: str):
        name = (index_name or "").strip().lower()

        if "gemini" in name or "google" in name:
            if google_ef is None:
                raise ValueError(
                    f"L'index '{index_name}' demande Google embeddings, mais google_ef n'est pas configuré."
                )
            return google_ef

        if "openai" in name:
            if openai_ef is None:
                raise ValueError(
                    f"L'index '{index_name}' demande OpenAI embeddings, mais openai_ef n'est pas configuré."
                )
            return openai_ef

        raise ValueError(
            f"Impossible de déterminer le provider d'embedding pour l'index '{index_name}'. "
            f"Le nom doit contenir 'gemini', 'google' ou 'openai'."
        )

    def _build_context(docs: List[str], metas: List[dict]) -> str:
        parts = []
        for i, doc in enumerate(docs):
            md = metas[i] if i < len(metas) else {}
            source_bits = []

            if md.get("id") is not None:
                source_bits.append(f"id={md.get('id')}")
            if md.get("newspaper_issue_id") is not None:
                source_bits.append(f"issue={md.get('newspaper_issue_id')}")
            if md.get("page_number") is not None:
                source_bits.append(f"page={md.get('page_number')}")
            if md.get("year") is not None:
                source_bits.append(f"year={md.get('year')}")

            source_str = ", ".join(source_bits) if source_bits else "source=unknown"
            parts.append(f"[Document {i+1} | {source_str}]\n{doc}")

        return "\n\n".join(parts)

    def _build_messages(question, documents, metadata, prompt_format=None, system_prompt=None):
        documents = _normalize_chroma_list(documents)
        metadata = _normalize_chroma_list(metadata)
        context = _build_context(documents, metadata)

        if not prompt_format:
            return [
                {
                    "role": "system",
                    "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nDocuments:\n{context}",
                },
            ]

        if isinstance(prompt_format, str):
            prompt_format = json.loads(prompt_format)

        return [
            {
                "role": x["role"],
                "content": x["content"]
                .replace("retrieved_docs", context)
                .replace("user_question", question)
            }
            for x in prompt_format
        ]

    def _llm_non_stream_benchmark(question, documents, metadata, model, prompt_format=None, system_prompt=None):
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

        messages = _build_messages(question, documents, metadata, prompt_format, system_prompt)

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            extra_body={},
            model=model,
            messages=messages,
            temperature=0,
            stream=False,
        )
        t1 = time.perf_counter()

        answer = ""
        if response.choices and response.choices[0].message:
            answer = response.choices[0].message.content or ""

        return {
            "answer": answer,
            "timings": {
                "llm_total": (t1 - t0),
            }
        }

    def _llm_stream_benchmark(question, documents, metadata, model, prompt_format=None, system_prompt=None):
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

        messages = _build_messages(question, documents, metadata, prompt_format, system_prompt)

        t0 = time.perf_counter()
        stream = client.chat.completions.create(
            extra_body={},
            model=model,
            messages=messages,
            temperature=0,
            stream=True,
        )
        t_after_create = time.perf_counter()

        first_any_chunk_at = None
        first_text_chunk_at = None
        last_chunk_at = None

        chunks_count = 0
        text_chunks_count = 0
        answer_parts = []

        for chunk in stream:
            now = time.perf_counter()
            chunks_count += 1

            if first_any_chunk_at is None:
                first_any_chunk_at = now

            delta_text = ""
            try:
                if chunk.choices and chunk.choices[0].delta:
                    delta_text = chunk.choices[0].delta.content or ""
            except Exception:
                delta_text = ""

            if delta_text:
                if first_text_chunk_at is None:
                    first_text_chunk_at = now
                text_chunks_count += 1
                answer_parts.append(delta_text)

            last_chunk_at = now

        t_end = time.perf_counter()

        return {
            "answer": "".join(answer_parts),
            "timings": {
                "llm_create_call": (t_after_create - t0),
                "llm_first_any_chunk": (first_any_chunk_at - t0) if first_any_chunk_at else None,
                "llm_first_text_chunk": (first_text_chunk_at - t0) if first_text_chunk_at else None,
                "llm_last_chunk": (last_chunk_at - t0) if last_chunk_at else None,
                "llm_stream_total": (t_end - t0),
            },
            "stream_stats": {
                "chunks_count": chunks_count,
                "text_chunks_count": text_chunks_count,
                "answer_chars": len("".join(answer_parts)),
            }
        }

    def _summarize_runs(runs):
        keys = sorted({k for run in runs for k in run["timings_ms"].keys()})
        out = {}

        for key in keys:
            values = [
                run["timings_ms"][key]
                for run in runs
                if key in run["timings_ms"] and run["timings_ms"][key] is not None
            ]
            if not values:
                continue
            out[key] = {
                "avg_ms": round(mean(values), 2),
                "min_ms": round(min(values), 2),
                "max_ms": round(max(values), 2),
            }

        return out

    def _single_benchmark_run(data):
        question = data.get("query") or data.get("question")
        if not question:
            raise ValueError("query (or question) is required")

        use_reranker = _safe_bool(data.get("use_reranker", True))
        include_answer = _safe_bool(data.get("include_answer", False))

        model = data.get("model") or env_default_llm_model
        index = data.get("index") or env_default_chroma_index
        prompt_format = data.get("prompt", None)
        system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        llm_mode = str(data.get("llm_mode", "stream")).strip().lower()

        filters = data.get("filters", {}) or {}
        k = int(data.get("k", 10))
        n_candidates = int(data.get("n_candidates", 50))
        filter_document_text = data.get("filter_document_text", None)

        source = data.get("source", "chroma")
        run_file_path = data.get("run_file_path", None)
        question_id = data.get("question_id", None)
        verbose = _safe_bool(data.get("verbose", True))

        reranker_config = _build_rerank_config(data)

        print("\n[BENCHMARK RERANK CONFIG - EFFECTIVE]")
        print(json.dumps({
            "model_name": reranker_config.model_name,
            "batch_size": reranker_config.batch_size,
            "device": reranker_config.device,
            "max_doc_chars": reranker_config.max_doc_chars,
            "window_chars": reranker_config.window_chars,
            "window_overlap": reranker_config.window_overlap,
            "min_window_chars": reranker_config.min_window_chars,
            "agg": reranker_config.agg,
            "topk_windows": reranker_config.topk_windows,
            "softmax_temp": reranker_config.softmax_temp,
            "normalize_whitespace": reranker_config.normalize_whitespace,
            "return_spans": reranker_config.return_spans,
            "add_scores_to_metadata": reranker_config.add_scores_to_metadata,
            "k": k,
            "n_candidates": n_candidates,
            "source": source,
            "use_metadata": reranker_config.use_metadata,
            "metadata_max_chars": reranker_config.metadata_max_chars,
            "run_file_path": run_file_path,
            "question_id": question_id,
        }, ensure_ascii=False, indent=2))

        timings_ms = {}
        t_total = time.perf_counter()

        rag_query = question
        rag_filters = filters
        rag_k = k
        embedding_fn = _get_query_embedding_fn_for_index(index)
        t = time.perf_counter()
        results = get_relating_documents_bis(
            rag_query,
            rag_filters,
            chroma_base_path=env_chroma_base_path,
            embedding_fn=embedding_fn,
            k_value=rag_k,
            filter_document_text=filter_document_text,
            index=index,
            use_reranker=use_reranker,
            rerank_config=reranker_config,
            n_candidates=n_candidates,
            chroma_persist_mode="per_index",
            return_timing=True,
            source=source,
            run_file_path=run_file_path,
            question_id=question_id,
            verbose=verbose,
        )
        timings_ms["retrieval_wrapper_ms"] = _ms(time.perf_counter() - t)


        retrieval_timings = results.get("timings", {})
        for key, value in retrieval_timings.items():
            timings_ms[f"{key}_ms"] = _ms(value)

        if llm_mode == "stream":
            llm_result = _llm_stream_benchmark(
                question=question,
                documents=results["documents"],
                metadata=results["metadatas"],
                model=model,
                prompt_format=prompt_format,
                system_prompt=system_prompt,
            )
        elif llm_mode == "non_stream":
            llm_result = _llm_non_stream_benchmark(
                question=question,
                documents=results["documents"],
                metadata=results["metadatas"],
                model=model,
                prompt_format=prompt_format,
                system_prompt=system_prompt,
            )
        else:
            raise ValueError("llm_mode must be 'stream' or 'non_stream'")

        for key, value in llm_result["timings"].items():
            timings_ms[f"{key}_ms"] = _ms(value)

        t = time.perf_counter()
        docs = _normalize_chroma_list(results.get("documents", []))
        metas = _normalize_chroma_list(results.get("metadatas", []))
        dists = _normalize_chroma_list(results.get("distances", []))

        preview = []
        for i in range(min(len(metas), 3)):
            item = dict(metas[i] or {})
            if i < len(dists):
                item["distance"] = dists[i]
            preview.append(item)

        timings_ms["response_build_ms"] = _ms(time.perf_counter() - t)
        timings_ms["total_ms"] = _ms(time.perf_counter() - t_total)

        run = {
            "query": question,
            "rag_query": rag_query,
            "rag_filters": rag_filters,
            "k": rag_k,
            "documents_count": len(docs),
            "timings_ms": timings_ms,
            "top_metadata_preview": preview,
            "llm_mode": llm_mode,
        }

        if "stream_stats" in llm_result:
            run["stream_stats"] = llm_result["stream_stats"]

        if "stats" in results:
            run["retrieval_stats"] = results["stats"]

        if include_answer:
            run["answer"] = llm_result["answer"]

        return run

    @app.route("/benchmark_pipeline", methods=["POST"])
    def benchmark_pipeline():
        data = request.get_json(silent=True) or {}

        try:
            run = _single_benchmark_run(data)
            run["run_number"] = 1
        except Exception as e:
            return jsonify({
                "error": str(e),
            }), 500

        return jsonify({
            "configuration": {
                "query": data.get("query") or data.get("question"),
                "repeat": 1,
                "warmup": 0,
                "use_agent_prompt": False,
                "use_reranker": _safe_bool(data.get("use_reranker", True)),
                "llm_mode": data.get("llm_mode", "stream"),
                "model": data.get("model") or env_default_llm_model,
                "system_prompt": data.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                "index": data.get("index") or env_default_chroma_index,
                "k": int(data.get("k", 10)),
                "n_candidates": int(data.get("n_candidates", 50)),
                "source": data.get("source", "chroma"),
                "reranker_model": (data.get("reranker_config", {}) or {}).get(
                    "model_name",
                    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
                ),
                "reranker_config": data.get("reranker_config", {}),
            },
            "summary": _summarize_runs([run]),
            "runs": [run],
        }), 200