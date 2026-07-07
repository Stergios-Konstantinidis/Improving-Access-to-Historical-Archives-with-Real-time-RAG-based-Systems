#!/usr/bin/env python3
"""
Pipeline runner for RAG / reranker evaluation.

Logical flow:
1. Chroma baseline generations
2. Temporal search enrichment
3. BM25 enrichment
4. Reciprocal Rank Fusion
5. Optional add_relevancy on the final fused retrieved_chunks
6. Optional legacy reranker
7. Subsets
8. Fill generated answers
9. RAGAS / eRAG

Important:
- No run_id.
- No state.json.
- Every step writes to a separate deterministic file.
- Existing outputs are reused unless the user explicitly reruns the step.
- When rerunning a step, the previous output is archived before overwrite.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


# =========================
# Environment
# =========================

def load_env(env_path: str | Path | None = None) -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path, override=False)


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = (PROJECT_ROOT / ".." / ".env").resolve()
load_env(os.getenv("ENV_FILE", str(ENV_PATH)))


# =========================
# Editable defaults
# =========================

DEFAULT_K_VALUES = [5, 10,25]

PIPELINE_ROOT = PROJECT_ROOT / "data" / "evaluation" / "pipeline"

CHROMA_GENERATIONS_FILE = PIPELINE_ROOT / "01_chroma" / "generations.chroma.jsonl"
TEMPORAL_GENERATIONS_FILE = PIPELINE_ROOT / "02_temporal" / "generations.chroma_temporal.jsonl"
BM25_GENERATIONS_FILE = PIPELINE_ROOT / "03_bm25" / "generations.chroma_temporal_bm25.jsonl"

FUSED_GENERATIONS_FILE = PIPELINE_ROOT / "04_fusion" / "generations.rrf.jsonl"

RELEVANCE_ANNOTATIONS_FILE = PIPELINE_ROOT / "05_relevance" / "annotations.metrics.json"
RELEVANCE_GENERATIONS_FILE = PIPELINE_ROOT / "05_relevance" / "generations.rrf.relevance.jsonl"

CHUNK_NDCG_GENERATIONS_FILE = (
    PIPELINE_ROOT
    / "06_metrics"
    / "generations.rrf.relevance.chunk_ndcg.jsonl"
)

GENERATION_NDCG_GENERATIONS_FILE = (
    PIPELINE_ROOT
    / "06_metrics"
    / "generations.rrf.relevance.chunk_ndcg.generation_ndcg.jsonl"
)

LEGACY_RERANKER_GENERATIONS_FILE = (
    PIPELINE_ROOT
    / "07_legacy_reranker"
    / "generations.legacy_reranker.jsonl"
)

PROCESSED_SUBSET_DIR = PIPELINE_ROOT / "08_subsets"
ENRICHED_SUBSET_DIR = PIPELINE_ROOT / "09_answered"

ARCHIVE_DIR = PIPELINE_ROOT / "archive" / "overwritten"

CORPUS_JSON = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"

# Garde ton gold ici si c'est son chemin actuel.
GOLD_FILE = PROJECT_ROOT / "data" / "baseline" / "gold_answers.json"

RAGAS_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "processed" / "ragas_erag"


# =========================
# Child scripts
# =========================

EXTRACT_CHROMA_SCRIPT = (
    PROJECT_ROOT
    / "utils"
    / "prepare_data"
    / "extract_chunk_from_chroma_from_evaluation_question_csv.py"
)

TEMPORAL_SCRIPT = (
    PROJECT_ROOT
    / "query_parsing"
    / "add_temporal_search_in_chroma_manually_in_a_generations_jsonl.py"
)

BM25_SCRIPT = (
    PROJECT_ROOT
    / "bm25s"
    / "add_bm25s_to_retrived_chunk_with_chroma.py"
)

RRF_SCRIPT = (
    PROJECT_ROOT
    / "rank_fusion"
    / "reciprocal_rank_fusion.py"
)

ADD_RELEVANCY_SCRIPT = (
    PROJECT_ROOT
    / "utils"
    / "reranker"
    / "add_relevancy.py"
)

CHUNK_NDCG_SCRIPT = (
    PROJECT_ROOT
    / "utils"
    / "reranker"
    / "chunk_revelance_ndcg_at_k.py"
)

GENERATION_NDCG_SCRIPT = (
    PROJECT_ROOT
    / "utils"
    / "reranker"
    / "generations_ndcg.py"
)

LEGACY_RERANKER_SCRIPT = (
    PROJECT_ROOT
    / "reranker"
    / "evaluation"
    / "evaluation.py"
)

SUBSET_SCRIPT = (
    PROJECT_ROOT
    / "utils"
    / "prepare_data"
    / "subset_of_chunks_from_generations_jsonl.py"
)

FILL_ANSWERS_SCRIPT = (
    PROJECT_ROOT
    / "utils"
    / "prepare_data"
    / "fill_generated_answers_if_empty_with_LLM.py"
)

COMPUTE_RAGAS_SCRIPT = (
    PROJECT_ROOT
    / "utils"
    / "reranker"
    / "compute_ragas.py"
)


STATIC_OUTPUT_PATHS = {
    "chroma": CHROMA_GENERATIONS_FILE,
    "temporal": TEMPORAL_GENERATIONS_FILE,
    "bm25": BM25_GENERATIONS_FILE,
    "rrf": FUSED_GENERATIONS_FILE,
    "relevance_annotations": RELEVANCE_ANNOTATIONS_FILE,
    "relevance_generations": RELEVANCE_GENERATIONS_FILE,
    "chunk_ndcg": CHUNK_NDCG_GENERATIONS_FILE,
    "generation_ndcg": GENERATION_NDCG_GENERATIONS_FILE,
    "legacy_reranker": LEGACY_RERANKER_GENERATIONS_FILE,
}


# =========================
# Helpers
# =========================

def run_command(cmd: list[str], description: str) -> None:
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print("Command:", " ".join(str(x) for x in cmd))
    print()

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(str(x) for x in cmd)}"
        )


def ask_yes_no(question: str, default: bool | None = None) -> bool:
    if default is True:
        suffix = " [Y/n] "
    elif default is False:
        suffix = " [y/N] "
    else:
        suffix = " [y/n] "

    while True:
        answer = input(question + suffix).strip().lower()

        if not answer and default is not None:
            return default

        if answer in {"y", "yes", "o", "oui"}:
            return True
        if answer in {"n", "no", "non"}:
            return False

        print("Réponse invalide. Tape 'y'/'yes'/'oui' ou 'n'/'no'/'non'.")


def as_path_list(paths: Path | list[Path] | tuple[Path, ...]) -> list[Path]:
    if isinstance(paths, Path):
        return [paths]
    return list(paths)


def script_arg(path: Path) -> str:
    """
    Return a path usable by subprocess from PROJECT_ROOT.
    Prefer relative paths for readable logs.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def assert_script_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Script introuvable : {path}")


def assert_input_output_different(
    input_paths: list[Path] | tuple[Path, ...] | None,
    output_paths: list[Path] | tuple[Path, ...],
) -> None:
    if not input_paths:
        return

    resolved_inputs = {Path(p).resolve() for p in input_paths}
    resolved_outputs = {Path(p).resolve() for p in output_paths}

    collisions = resolved_inputs & resolved_outputs
    if collisions:
        collision_text = "\n".join(str(p) for p in sorted(collisions))
        raise ValueError(
            "Refus d'exécuter : au moins un fichier est à la fois input et output.\n"
            f"{collision_text}"
        )


def validate_static_output_paths() -> None:
    seen: dict[Path, str] = {}

    for name, path in STATIC_OUTPUT_PATHS.items():
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(
                "Deux étapes écrivent dans le même fichier :\n"
                f"- {seen[resolved]}\n"
                f"- {name}\n"
                f"Chemin : {resolved}"
            )
        seen[resolved] = name


def validate_required_inputs() -> None:
    if not CORPUS_JSON.exists():
        raise FileNotFoundError(f"Corpus BM25 introuvable : {CORPUS_JSON}")

    if not GOLD_FILE.exists():
        raise FileNotFoundError(f"Gold file introuvable : {GOLD_FILE}")


def archive_existing_outputs(output_paths: list[Path]) -> None:
    existing = [p for p in output_paths if p.exists()]
    if not existing:
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_run_dir = ARCHIVE_DIR / stamp
    archive_run_dir.mkdir(parents=True, exist_ok=True)

    for path in existing:
        try:
            relative = path.relative_to(PIPELINE_ROOT)
            destination = archive_run_dir / relative
        except ValueError:
            destination = archive_run_dir / path.name

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

        print(f"[archive] {path} -> {destination}")


def should_run_step(
    output_paths: list[Path],
    description: str,
    force: bool = False,
    assume_no: bool = False,
) -> bool:
    if force:
        return True

    missing = [p for p in output_paths if not p.exists()]

    if missing:
        print(f"\n[INFO] Étape nécessaire : {description}")
        for path in missing:
            print(f"[INFO] Fichier absent : {path}")
        return True

    print(f"\n[INFO] Sortie(s) existante(s) pour : {description}")
    for path in output_paths:
        print(f"[INFO] Existe déjà : {path}")

    if assume_no:
        print("[INFO] --assume-no actif : étape ignorée.")
        return False

    return ask_yes_no(
        f"{description} existe déjà. Veux-tu relancer cette étape ?",
        default=False,
    )


def run_step_with_output(
    cmd: list[str],
    description: str,
    output_paths: Path | list[Path] | tuple[Path, ...],
    input_paths: list[Path] | tuple[Path, ...] | None = None,
    force: bool = False,
    assume_no: bool = False,
    archive_on_overwrite: bool = True,
) -> None:
    outputs = as_path_list(output_paths)

    assert_input_output_different(input_paths=input_paths, output_paths=outputs)

    run_it = should_run_step(
        output_paths=outputs,
        description=description,
        force=force,
        assume_no=assume_no,
    )

    if run_it:
        for output_path in outputs:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if archive_on_overwrite:
            archive_existing_outputs(outputs)

        run_command(cmd, description)

    missing_after = [p for p in outputs if not p.exists()]
    if missing_after:
        missing_text = "\n".join(str(p) for p in missing_after)
        raise FileNotFoundError(
            "L'étape est terminée ou ignorée, mais des fichiers attendus sont introuvables :\n"
            f"{missing_text}"
        )


def jsonl_has_relevance_scores(path: Path, max_records: int = 5) -> bool:
    if not path.exists():
        return False

    checked = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            checked += 1
            chunks = record.get("retrieved_chunks", [])

            for chunk in chunks[:5]:
                if not isinstance(chunk, dict):
                    continue

                metadata = chunk.get("metadata", {}) or {}

                for key in ("relevance_score", "is_relevant", "justification"):
                    if key in chunk or key in metadata:
                        return True

            if checked >= max_records:
                break

    return False


def build_output_subdir_name(generations_file: Path) -> str:
    """
    Same naming logic as compute_ragas.py.

    examples:
    - generations_k5.jsonl -> generations_5k
    - generations_5k.jsonl -> generations_5k
    - anything_else.jsonl -> anything_else
    """
    stem = generations_file.stem

    match_k_prefix = re.match(r"^generations_k(\d+)$", stem)
    if match_k_prefix:
        return f"generations_{match_k_prefix.group(1)}k"

    match_k_suffix = re.match(r"^generations_(\d+)k$", stem)
    if match_k_suffix:
        return stem

    return stem


def build_ragas_summary_path(generations_file: Path) -> Path:
    return RAGAS_OUTPUT_DIR / build_output_subdir_name(generations_file) / "ragas_summary.json"


# =========================
# Pipeline steps
# =========================

def ensure_chroma_generations(args: argparse.Namespace) -> None:
    assert_script_exists(EXTRACT_CHROMA_SCRIPT)

    run_step_with_output(
        cmd=[
            sys.executable,
            script_arg(EXTRACT_CHROMA_SCRIPT),
            "--output-jsonl",
            str(CHROMA_GENERATIONS_FILE),
        ],
        description="Step 1 - Extract Chroma chunks from evaluation questions",
        output_paths=CHROMA_GENERATIONS_FILE,
        force=args.force,
        assume_no=args.assume_no,
    )


def run_temporal_search(args: argparse.Namespace) -> None:
    assert_script_exists(TEMPORAL_SCRIPT)

    run_step_with_output(
        cmd=[
            sys.executable,
            script_arg(TEMPORAL_SCRIPT),
            "--input-jsonl",
            str(CHROMA_GENERATIONS_FILE),
            "--output-jsonl",
            str(TEMPORAL_GENERATIONS_FILE),
            "--top-k-temporal",
            str(args.top_k_temporal),
        ],
        description="Step 2 - Add temporal search results",
        input_paths=[CHROMA_GENERATIONS_FILE],
        output_paths=TEMPORAL_GENERATIONS_FILE,
        force=args.force,
        assume_no=args.assume_no,
    )


def run_bm25(args: argparse.Namespace) -> None:
    assert_script_exists(BM25_SCRIPT)

    run_step_with_output(
        cmd=[
            sys.executable,
            script_arg(BM25_SCRIPT),
            "--corpus-json",
            str(CORPUS_JSON),
            "--input-generations-jsonl",
            str(TEMPORAL_GENERATIONS_FILE),
            "--output-jsonl",
            str(BM25_GENERATIONS_FILE),
            "--bm25-stopwords",
            args.bm25_stopwords,
            "--bm25-stem-language",
            args.bm25_stem_language,
        ],
        description="Step 3 - Add BM25 results",
        input_paths=[TEMPORAL_GENERATIONS_FILE, CORPUS_JSON],
        output_paths=BM25_GENERATIONS_FILE,
        force=args.force,
        assume_no=args.assume_no,
    )


def run_rrf(args: argparse.Namespace) -> None:
    assert_script_exists(RRF_SCRIPT)

    run_step_with_output(
        cmd=[
            sys.executable,
            script_arg(RRF_SCRIPT),
            "--input-jsonl",
            str(BM25_GENERATIONS_FILE),
            "--output-jsonl",
            str(FUSED_GENERATIONS_FILE),
            "--keep-top-k",
            str(args.keep_top_k),
            "--rrf-k",
            str(args.rrf_k),
            "--chroma-weight",
            str(args.chroma_weight),
            "--bm25s-weight",
            str(args.bm25s_weight),
            "--temporal-weight",
            str(args.temporal_weight),
            "--chroma-max-rank",
            str(args.chroma_max_rank),
            "--bm25s-max-rank",
            str(args.bm25s_max_rank),
            "--temporal-max-rank",
            str(args.temporal_max_rank),
            "--reserve-bm25s",
            str(args.reserve_bm25s),
        ],
        description="Step 4 - Reciprocal Rank Fusion",
        input_paths=[BM25_GENERATIONS_FILE],
        output_paths=FUSED_GENERATIONS_FILE,
        force=args.force,
        assume_no=args.assume_no,
    )


def should_run_add_relevancy(args: argparse.Namespace, relevance_ready: bool) -> bool:
    if args.add_relevancy == "yes":
        return True

    if args.add_relevancy == "no":
        return False

    if args.assume_no:
        if relevance_ready:
            print("[INFO] --assume-no actif : relevance déjà présente, add_relevancy ignoré.")
            return False

        print("[INFO] --assume-no actif : relevance absente, add_relevancy exécuté car l'étape manque.")
        return True

    if relevance_ready:
        return ask_yes_no(
            "Le fichier de relevance existe déjà. Veux-tu recalculer add_relevancy ?",
            default=False,
        )

    return ask_yes_no(
        "Le fichier de relevance n'existe pas encore. Veux-tu exécuter add_relevancy maintenant ?",
        default=True,
    )


def run_final_relevancy_if_needed(args: argparse.Namespace) -> Path:
    """
    add_relevancy must NOT modify FUSED_GENERATIONS_FILE in place.

    Expected child-script contract:

      utils/reranker/add_relevancy.py
        --input-jsonl
        --output-jsonl
        --annotations-output

      utils/reranker/chunk_ndcg.py
        --input-jsonl
        --output-jsonl
        --annotations-json

      utils/reranker/generation_ndcg.py
        --input-jsonl
        --output-jsonl
    """
    if not FUSED_GENERATIONS_FILE.exists():
        raise FileNotFoundError(f"Fichier fusionné introuvable : {FUSED_GENERATIONS_FILE}")

    relevance_ready = (
        RELEVANCE_GENERATIONS_FILE.exists()
        and RELEVANCE_ANNOTATIONS_FILE.exists()
        and jsonl_has_relevance_scores(RELEVANCE_GENERATIONS_FILE)
    )

    run_add_relevancy = should_run_add_relevancy(args, relevance_ready=relevance_ready)

    if run_add_relevancy:
        assert_script_exists(ADD_RELEVANCY_SCRIPT)
        assert_script_exists(CHUNK_NDCG_SCRIPT)
        assert_script_exists(GENERATION_NDCG_SCRIPT)

        run_step_with_output(
            cmd=[
                sys.executable,
                script_arg(ADD_RELEVANCY_SCRIPT),
                "--input-jsonl",
                str(FUSED_GENERATIONS_FILE),
                "--output-jsonl",
                str(RELEVANCE_GENERATIONS_FILE),
                "--annotations-output",
                str(RELEVANCE_ANNOTATIONS_FILE),
                "--gold",
                str(GOLD_FILE),
            ],
            description="Step 5 - Add relevancy on final fused generations",
            input_paths=[FUSED_GENERATIONS_FILE],
            output_paths=[RELEVANCE_GENERATIONS_FILE, RELEVANCE_ANNOTATIONS_FILE],
            force=True,
            assume_no=False,
        )

        run_step_with_output(
            cmd=[
                sys.executable,
                script_arg(CHUNK_NDCG_SCRIPT),
                "--input-jsonl",
                str(RELEVANCE_GENERATIONS_FILE),
                "--output-jsonl",
                str(CHUNK_NDCG_GENERATIONS_FILE),
                "--annotations-json",
                str(RELEVANCE_ANNOTATIONS_FILE),
            ],
            description="Step 6 - Compute chunk relevance NDCG@k",
            input_paths=[RELEVANCE_GENERATIONS_FILE, RELEVANCE_ANNOTATIONS_FILE],
            output_paths=CHUNK_NDCG_GENERATIONS_FILE,
            force=True,
            assume_no=False,
        )

        run_step_with_output(
            cmd=[
                sys.executable,
                script_arg(GENERATION_NDCG_SCRIPT),
                "--input-jsonl",
                str(CHUNK_NDCG_GENERATIONS_FILE),
                "--output-jsonl",
                str(GENERATION_NDCG_GENERATIONS_FILE),
            ],
            description="Step 7 - Compute generations NDCG",
            input_paths=[CHUNK_NDCG_GENERATIONS_FILE],
            output_paths=GENERATION_NDCG_GENERATIONS_FILE,
            force=True,
            assume_no=False,
        )

        return GENERATION_NDCG_GENERATIONS_FILE

    if relevance_ready:
        print("[INFO] add_relevancy ignoré, mais relevance existante détectée.")

        assert_script_exists(CHUNK_NDCG_SCRIPT)
        assert_script_exists(GENERATION_NDCG_SCRIPT)

        run_step_with_output(
            cmd=[
                sys.executable,
                script_arg(CHUNK_NDCG_SCRIPT),
                "--input-jsonl",
                str(RELEVANCE_GENERATIONS_FILE),
                "--output-jsonl",
                str(CHUNK_NDCG_GENERATIONS_FILE),
                "--annotations-json",
                str(RELEVANCE_ANNOTATIONS_FILE),
            ],
            description="Step 6 - Compute chunk relevance NDCG@k",
            input_paths=[RELEVANCE_GENERATIONS_FILE, RELEVANCE_ANNOTATIONS_FILE],
            output_paths=CHUNK_NDCG_GENERATIONS_FILE,
            force=args.force,
            assume_no=args.assume_no,
        )

        run_step_with_output(
            cmd=[
                sys.executable,
                script_arg(GENERATION_NDCG_SCRIPT),
                "--input-jsonl",
                str(CHUNK_NDCG_GENERATIONS_FILE),
                "--output-jsonl",
                str(GENERATION_NDCG_GENERATIONS_FILE),
            ],
            description="Step 7 - Compute generations NDCG",
            input_paths=[CHUNK_NDCG_GENERATIONS_FILE],
            output_paths=GENERATION_NDCG_GENERATIONS_FILE,
            force=args.force,
            assume_no=args.assume_no,
        )

        return GENERATION_NDCG_GENERATIONS_FILE

    print("[INFO] add_relevancy ignoré.")
    print(
        "[WARN] Aucun fichier de relevance valide disponible. "
        "Les subsets/RAGAS continueront depuis le fichier RRF brut."
    )

    return FUSED_GENERATIONS_FILE


def run_optional_legacy_reranker(args: argparse.Namespace, input_file: Path) -> Path:
    """
    Optional legacy reranker.

    Expected child-script contract:
      reranker/evaluation/evaluation.py
        --input
        --output
    """
    if args.reranker == "yes":
        use_reranker = True
    elif args.reranker == "no":
        use_reranker = False
    else:
        use_reranker = ask_yes_no(
            "Veux-tu exécuter aussi le reranker legacy après RRF / relevance ?",
            default=False,
        )

    if not use_reranker:
        print("\n[INFO] Reranker legacy ignoré.")
        return input_file

    assert_script_exists(LEGACY_RERANKER_SCRIPT)

    run_step_with_output(
        cmd=[
            sys.executable,
            script_arg(LEGACY_RERANKER_SCRIPT),
            "--input",
            str(input_file),
            "--output",
            str(LEGACY_RERANKER_GENERATIONS_FILE),
        ],
        description="Optional - Legacy reranker evaluation",
        input_paths=[input_file],
        output_paths=LEGACY_RERANKER_GENERATIONS_FILE,
        force=args.force,
        assume_no=args.assume_no,
    )

    return LEGACY_RERANKER_GENERATIONS_FILE


def build_subset_output_path(active_prefix: str, k: int) -> Path:
    return PROCESSED_SUBSET_DIR / f"{active_prefix}_k{k}.jsonl"


def build_answered_output_path(active_prefix: str, k: int) -> Path:
    return ENRICHED_SUBSET_DIR / f"{active_prefix}_k{k}_answered.jsonl"


def create_subsets(args: argparse.Namespace, active_generations_file: Path, active_prefix: str) -> None:
    assert_script_exists(SUBSET_SCRIPT)
    PROCESSED_SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    for k in args.k_values:
        subset_output = build_subset_output_path(active_prefix, k)

        run_step_with_output(
            cmd=[
                sys.executable,
                script_arg(SUBSET_SCRIPT),
                "--k",
                str(k),
                "--input",
                str(active_generations_file),
                "--output",
                str(subset_output),
            ],
            description=f"Create subset for k={k}",
            input_paths=[active_generations_file],
            output_paths=subset_output,
            force=args.force,
            assume_no=args.assume_no,
        )


def fill_answers(args: argparse.Namespace, active_prefix: str) -> None:
    """
    Expected child-script contract:
      utils/prepare_data/fill_generated_answers_if_empty_with_LLM.py
        --input
        --output
    """
    assert_script_exists(FILL_ANSWERS_SCRIPT)
    ENRICHED_SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    for k in args.k_values:
        subset_input = build_subset_output_path(active_prefix, k)
        answered_output = build_answered_output_path(active_prefix, k)

        if not subset_input.exists():
            raise FileNotFoundError(f"Subset file introuvable : {subset_input}")

        run_step_with_output(
            cmd=[
                sys.executable,
                script_arg(FILL_ANSWERS_SCRIPT),
                "--input",
                str(subset_input),
                "--output",
                str(answered_output),
            ],
            description=f"Fill generated answers for k={k}",
            input_paths=[subset_input],
            output_paths=answered_output,
            force=args.force,
            assume_no=args.assume_no,
        )


def compute_ragas(args: argparse.Namespace, active_prefix: str) -> None:
    """
    Expected child-script contract:
      utils/reranker/compute_ragas.py
        --run
        --gold

    The summary path is expected to follow compute_ragas.py naming logic.
    """
    assert_script_exists(COMPUTE_RAGAS_SCRIPT)

    for k in args.k_values:
        answered_input = build_answered_output_path(active_prefix, k)

        if not answered_input.exists():
            raise FileNotFoundError(f"Fichier enrichi attendu introuvable : {answered_input}")

        ragas_summary = build_ragas_summary_path(answered_input)

        run_step_with_output(
            cmd=[
                sys.executable,
                script_arg(COMPUTE_RAGAS_SCRIPT),
                "--run",
                str(answered_input),
                "--gold",
                str(GOLD_FILE),
            ],
            description=f"Compute RAGAS/eRAG for k={k}",
            input_paths=[answered_input, GOLD_FILE],
            output_paths=ragas_summary,
            force=args.force,
            assume_no=args.assume_no,
        )


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete Chroma + temporal + BM25 + RRF + RAGAS/eRAG pipeline."
    )

    parser.add_argument(
        "--reranker",
        choices=["yes", "no", "ask"],
        default="ask",
        help="Also run legacy reranker/evaluation/evaluation.py after RRF. Default: ask",
    )

    parser.add_argument(
        "--add-relevancy",
        choices=["yes", "no", "ask"],
        default="ask",
        help="Run add_relevancy on final fused generations. Default: ask",
    )

    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="List of k values for subset generation. Example: --k-values 5 10 25 50",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun every step even if output files already exist.",
    )

    parser.add_argument(
        "--assume-no",
        action="store_true",
        help="Non-interactive mode: reuse existing files and only run missing steps.",
    )

    parser.add_argument("--top-k-temporal", type=int, default=10)

    parser.add_argument("--keep-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=1)

    parser.add_argument("--chroma-weight", type=float, default=0.5)
    parser.add_argument("--bm25s-weight", type=float, default=0.6)
    parser.add_argument("--temporal-weight", type=float, default=0.5)

    parser.add_argument("--chroma-max-rank", type=int, default=1000000)
    parser.add_argument("--bm25s-max-rank", type=int, default=50)
    parser.add_argument("--temporal-max-rank", type=int, default=10)

    parser.add_argument("--reserve-bm25s", type=int, default=0)

    parser.add_argument("--bm25-stopwords", type=str, default="fr")
    parser.add_argument("--bm25-stem-language", type=str, default="french")

    args = parser.parse_args()

    if args.force and args.assume_no:
        raise ValueError("--force et --assume-no ne doivent pas être utilisés ensemble.")

    validate_static_output_paths()
    validate_required_inputs()

    print("\n[INFO] PROJECT_ROOT:", PROJECT_ROOT)
    print("[INFO] ENV_PATH:", ENV_PATH)
    print("[INFO] PIPELINE_ROOT:", PIPELINE_ROOT)
    print("[INFO] K values:", args.k_values)
    print("[INFO] add_relevancy mode:", args.add_relevancy)
    print("[INFO] reranker mode:", args.reranker)

    ensure_chroma_generations(args)
    run_temporal_search(args)
    run_bm25(args)
    run_rrf(args)

    active_generations_file = run_final_relevancy_if_needed(args)
    active_generations_file = run_optional_legacy_reranker(args, active_generations_file)

    active_prefix = active_generations_file.stem

    print("\n[INFO] Fichier actif pour subsets/RAGAS :", active_generations_file)
    print("[INFO] Préfixe actif :", active_prefix)

    create_subsets(args, active_generations_file, active_prefix)
    fill_answers(args, active_prefix)
    compute_ragas(args, active_prefix)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()