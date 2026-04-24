#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_ragas_synthetic_fr.py
=============================

Génère un jeu de données synthétique Ragas en français à partir d'un fichier JSONL
contenant déjà les chunks de ton index.

Entrée attendue :
    {"id": "...", "text": "...", "metadata": {...}}

Sorties :
- questions.jsonl
- gold_answers.json
- ragas_testset_full.json

Exemple :
    python prepare_ragas_synthetic_fr.py ^
        --input .\data\baseline\documents.jsonl ^
        --output-dir .\data\baseline\synthetic_fr ^
        --testset-size 50 ^
        --max-docs 5000

Variables d'environnement :
    OPENAI_API_KEY=...
    RAGAS_LLM_MODEL=gpt-4.1-mini
    RAGAS_EMBEDDING_MODEL=text-embedding-3-small
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from openai import OpenAI
from langchain_core.documents import Document

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.synthesizers.generate import TestsetGenerator


DEFAULT_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gpt-4.1-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")

DEFAULT_MIN_TEXT_LENGTH = int(os.getenv("RAGAS_MIN_TEXT_LENGTH", "250"))
DEFAULT_MAX_TEXT_LENGTH = int(os.getenv("RAGAS_MAX_TEXT_LENGTH", "12000"))
DEFAULT_TESTSET_SIZE = int(os.getenv("RAGAS_TESTSET_SIZE", "5"))
DEFAULT_MAX_DOCS = int(os.getenv("RAGAS_MAX_DOCS", "200"))
DEFAULT_RANDOM_SEED = int(os.getenv("RAGAS_RANDOM_SEED", "42"))


def is_nan_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


def normalize_text(value: Any) -> str:
    if is_nan_like(value):
        return ""
    return str(value).strip()


def safe_metadata_value(value: Any) -> Any:
    if is_nan_like(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [safe_metadata_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_metadata_value(v) for k, v in value.items()}
    return str(value)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON invalide à la ligne {line_num} dans {path}: {e}") from e

    return rows

def sanity_check_openai(client: OpenAI, llm_model: str) -> None:
    print("[DEBUG] Test OpenAI direct...")
    start = time.time()

    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": "Réponds uniquement par OK"}],
        temperature=0,
        max_tokens=5,
    )

    elapsed = time.time() - start
    content = response.choices[0].message.content
    print(f"[DEBUG] OpenAI répond bien en {elapsed:.2f}s : {content!r}")

def normalize_source_row(
    row: Dict[str, Any],
    min_text_length: int,
    max_text_length: int,
) -> Optional[Dict[str, Any]]:
    source_id = normalize_text(row.get("id"))
    text = normalize_text(row.get("text"))
    metadata = row.get("metadata", {}) or {}

    if not text:
        return None

    if len(text) < min_text_length:
        return None

    if len(text) > max_text_length:
        text = text[:max_text_length]

    cleaned_metadata = {
        "source_id": source_id,
        "title": safe_metadata_value(metadata.get("title")),
        "summary": safe_metadata_value(metadata.get("summary")),
        "year": safe_metadata_value(metadata.get("year")),
        "month": safe_metadata_value(metadata.get("month")),
        "day": safe_metadata_value(metadata.get("day")),
        "page_number": safe_metadata_value(metadata.get("page_number")),
        "newspaper_issue_id": safe_metadata_value(metadata.get("newspaper_issue_id")),
        "newspaper": safe_metadata_value(metadata.get("newspaper")),
        "label": safe_metadata_value(metadata.get("label")),
        "Person": safe_metadata_value(metadata.get("Person")),
        "Location": safe_metadata_value(metadata.get("Location")),
        "Event": safe_metadata_value(metadata.get("Event")),
        "keywords": safe_metadata_value(metadata.get("keywords")),
        "coordinates": safe_metadata_value(metadata.get("coordinates")),
    }

    return {
        "id": source_id,
        "text": text,
        "metadata": cleaned_metadata,
    }


def deduplicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()

    for row in rows:
        key = row["text"].strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def prepare_rows(
    source_rows: List[Dict[str, Any]],
    min_text_length: int,
    max_text_length: int,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []

    for row in tqdm(source_rows, desc="Préparation des chunks", unit="doc"):
        item = normalize_source_row(
            row=row,
            min_text_length=min_text_length,
            max_text_length=max_text_length,
        )
        if item is not None:
            prepared.append(item)

    prepared = deduplicate_rows(prepared)
    return prepared


def sample_rows(
    rows: List[Dict[str, Any]],
    max_docs: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    if max_docs <= 0 or len(rows) <= max_docs:
        return rows

    rng = random.Random(random_seed)
    sampled = rng.sample(rows, max_docs)
    return sampled


def rows_to_langchain_documents(rows: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []

    for row in rows:
        metadata = dict(row.get("metadata", {}) or {})
        metadata["chunk_id"] = row.get("id", "")

        docs.append(
            Document(
                page_content=row["text"],
                metadata=metadata,
            )
        )

    return docs


def build_ragas_generator(
    llm_model: str,
    embedding_model: str,
) -> tuple[TestsetGenerator, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY est manquant dans l'environnement.")

    print("[DEBUG] OPENAI_API_KEY détectée")
    print(f"[DEBUG] LLM model: {llm_model}")
    print(f"[DEBUG] Embedding model: {embedding_model}")

    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=llm_model,
            api_key=api_key,
            temperature=0,
        )
    )
    print("[DEBUG] LangchainLLMWrapper OK")

    openai_client = OpenAI(api_key=api_key)
    generator_embeddings = OpenAIEmbeddings(
        client=openai_client,
        model=embedding_model,
    )
    print("[DEBUG] OpenAIEmbeddings OK")

    personas = [
        Persona(
            name="lecteur curieux",
            role_description="Un lecteur francophone curieux qui cherche des informations précises dans des archives de presse historiques",
        ),
    ]
    print("[DEBUG] Persona list OK")

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas,
    )
    print("[DEBUG] TestsetGenerator OK")

    return generator, generator_llm

import asyncio

from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

async def build_french_query_distribution(generator_llm):
    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
    ]

    for query, _ in distribution:
        prompts = await query.adapt_prompts("french", llm=generator_llm)
        query.set_prompts(**prompts)

    return distribution

def generate_testset(
    generator: TestsetGenerator,
    generator_llm,
    chunks: List[Document],
    testset_size: int,
):
    print("[DEBUG] Construction de la query_distribution FR...")
    query_distribution = asyncio.run(build_french_query_distribution(generator_llm))
    print("[DEBUG] query_distribution FR OK")

    transforms = [NERExtractor()]
    print("[DEBUG] transforms OK")

    print("[DEBUG] Appel à generator.generate_with_langchain_docs(...)")
    print(f"[DEBUG] chunks={len(chunks)}, testset_size={testset_size}")

    try:
        result = generator.generate_with_langchain_docs(
            chunks,
            testset_size=testset_size,
            transforms=transforms,
            query_distribution=query_distribution,
        )
        print("[DEBUG] generate_with_langchain_docs terminé")
        return result
    except Exception as e:
        print(f"[ERROR] generate_with_langchain_docs a échoué: {type(e).__name__}: {e}")
        raise

def coerce_testset_to_list(testset: Any) -> List[Dict[str, Any]]:
    if hasattr(testset, "to_list") and callable(testset.to_list):
        data = testset.to_list()
        if isinstance(data, list):
            return data

    if hasattr(testset, "to_pandas") and callable(testset.to_pandas):
        df = testset.to_pandas()
        return df.to_dict(orient="records")

    if isinstance(testset, list):
        return testset

    raise RuntimeError(
        "Impossible de convertir le testset en liste. "
        "Aucune méthode to_list()/to_pandas() détectée."
    )


def pick_question(sample: Dict[str, Any]) -> str:
    for key in ("user_input", "question", "query"):
        value = normalize_text(sample.get(key))
        if value:
            return value
    return ""


def pick_reference(sample: Dict[str, Any]) -> str:
    for key in ("reference", "ground_truth", "answer"):
        value = normalize_text(sample.get(key))
        if value:
            return value
    return ""


def pick_reference_contexts(sample: Dict[str, Any]) -> List[str]:
    for key in ("reference_contexts", "contexts", "retrieved_contexts"):
        value = sample.get(key)
        if isinstance(value, list):
            out = [normalize_text(v) for v in value if normalize_text(v)]
            if out:
                return out
    return []


def export_outputs(
    testset_records: List[Dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_path = output_dir / "questions.jsonl"
    gold_answers_path = output_dir / "gold_answers.json"
    full_path = output_dir / "ragas_testset_full.json"

    gold_answers_dict: Dict[str, str] = {}
    full_records: List[Dict[str, Any]] = []

    with questions_path.open("w", encoding="utf-8") as f_questions:
        for i, sample in enumerate(testset_records):
            qid = f"q_{i:06d}"

            question = pick_question(sample)
            reference = pick_reference(sample)
            reference_contexts = pick_reference_contexts(sample)

            if not question or not reference:
                continue

            metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}

            question_record = {
                "question_id": qid,
                "question": question,
                "language": "fr",
                "question_type": metadata.get("question_type"),
            }

            f_questions.write(json.dumps(question_record, ensure_ascii=False) + "\n")
            gold_answers_dict[qid] = reference

            full_records.append({
                "question_id": qid,
                "question": question,
                "gold_answer": reference,
                "reference_contexts": reference_contexts,
                "language": "fr",
                "metadata": metadata,
                "raw_sample": sample,
            })

    gold_answers_path.write_text(
        json.dumps(gold_answers_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    full_path.write_text(
        json.dumps(full_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[OK] Fichiers générés")
    print(f" - {questions_path}")
    print(f" - {gold_answers_path}")
    print(f" - {full_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prépare un jeu de données synthétique Ragas en français"
    )
    parser.add_argument("--input", required=True, help="Chemin du corpus source JSONL")
    parser.add_argument("--output-dir", required=True, help="Dossier de sortie")
    parser.add_argument("--testset-size", type=int, default=DEFAULT_TESTSET_SIZE)
    parser.add_argument("--min-text-length", type=int, default=DEFAULT_MIN_TEXT_LENGTH)
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_path}")

    print("=" * 72)
    print("PRÉPARATION DU JEU DE DONNÉES SYNTHÉTIQUE RAGAS (FR)")
    print("=" * 72)
    print(f"Corpus source       : {input_path}")
    print(f"Dossier de sortie   : {args.output_dir}")
    print(f"Taille du testset   : {args.testset_size}")
    print(f"LLM                 : {args.llm_model}")
    print(f"Embeddings model    : {args.embedding_model}")
    print(f"Longueur min texte  : {args.min_text_length}")
    print(f"Longueur max texte  : {args.max_text_length}")
    print(f"Max docs            : {args.max_docs}")
    print(f"Random seed         : {args.random_seed}")
    print("=" * 72)

    source_rows = load_jsonl(input_path)
    print(f"[INFO] Documents lus: {len(source_rows)}")

    prepared_rows = prepare_rows(
        source_rows=source_rows,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
    )
    print(f"[INFO] Chunks retenus après filtrage/dédup: {len(prepared_rows)}")

    if not prepared_rows:
        raise RuntimeError("Aucun chunk exploitable après filtrage.")

    sampled_rows = sample_rows(
        rows=prepared_rows,
        max_docs=args.max_docs,
        random_seed=args.random_seed,
    )
    print(f"[INFO] Chunks utilisés pour la génération: {len(sampled_rows)}")

    langchain_docs = rows_to_langchain_documents(sampled_rows)

    generator, generator_llm = build_ragas_generator(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
    )

    print("[INFO] Génération du testset...")
    testset = generate_testset(
        generator=generator,
        generator_llm=generator_llm,
        chunks=langchain_docs,
        testset_size=args.testset_size,
    )

    testset_records = coerce_testset_to_list(testset)
    print(f"[INFO] Samples générés: {len(testset_records)}")

    export_outputs(
        testset_records=testset_records,
        output_dir=args.output_dir,
    )

    print("\n[OK] Terminé.")


if __name__ == "__main__":
    main()