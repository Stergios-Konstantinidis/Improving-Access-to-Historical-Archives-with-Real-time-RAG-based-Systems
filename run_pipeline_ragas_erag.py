# run_pipeline_ragas_erag.py
#!/usr/bin/env python3
"""
Pipeline runner for RAG / reranker evaluation.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def load_env(env_path: str | Path | None = None) -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path, override=False)

# Configuration

# =========================
# Editable parameters
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = (PROJECT_ROOT / ".." / ".env").resolve()

print(ENV_PATH)

load_env(os.getenv("ENV_FILE", str(ENV_PATH)))
K_VALUES = [10]

BASELINE_DIR = PROJECT_ROOT / "data" / "reranker" / "baseline"
PROCESSED_SUBSET_DIR = PROJECT_ROOT / "data" / "reranker" / "processed" / "subset"
ENRICHED_SUBSET_DIR = PROCESSED_SUBSET_DIR / "enriched"

BASE_GENERATIONS_FILE = BASELINE_DIR / "generations.jsonl"
RERANKER_GENERATIONS_FILE = BASELINE_DIR / "generations_reranker.jsonl"
GOLD_FILE = PROJECT_ROOT / "data" / "baseline" / "gold_answers.json"


def run_command(cmd: list[str], description: str) -> None:
    """Run a shell command and stop on failure."""
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print("Command:", " ".join(str(x) for x in cmd))
    print()

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(str(x) for x in cmd)}"
        )


def ask_yes_no(question: str, default: bool | None = None) -> bool:
    """Interactive yes/no prompt."""
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


def run_initial_pipeline() -> None:
    """Run the initial preprocessing pipeline."""
    initial_commands = [
        (
            [sys.executable, "utils/prepare_data/extract_chunk_from_chroma_from_evaluation_question_csv.py"],
            "Step 1/4 - Extract chunks from evaluation_questions.csv",
        ),
        (
            [sys.executable, "utils/reranker/add_relevancy.py"],
            "Step 2/4 - Add relevancy",
        ),
        (
            [sys.executable, "utils/reranker/chunk_revelance_ndcg_at_k.py"],
            "Step 3/4 - Compute chunk relevance NDCG@k",
        ),
        (
            [sys.executable, "utils/reranker/generations_ndcg.py"],
            "Step 4/4 - Compute generations NDCG",
        ),
    ]

    for cmd, desc in initial_commands:
        run_command(cmd, desc)

    if not BASE_GENERATIONS_FILE.exists():
        raise FileNotFoundError(
            f"La phase initiale a terminé, mais le fichier attendu n'existe toujours pas : "
            f"{BASE_GENERATIONS_FILE}"
        )


def ensure_initial_pipeline() -> None:
    """
    If generations.jsonl exists, ask whether to rerun the initial phase.
    If it does not exist, run it automatically.
    """
    if BASE_GENERATIONS_FILE.exists():
        print(f"\n[INFO] Fichier existant détecté : {BASE_GENERATIONS_FILE}")
        rerun = ask_yes_no(
            "La phase initiale a déjà été exécutée. Veux-tu la relancer ?",
            default=False,
        )
        if rerun:
            run_initial_pipeline()
        else:
            print("[INFO] Phase initiale ignorée.")
    else:
        print(f"\n[INFO] Fichier absent : {BASE_GENERATIONS_FILE}")
        print("[INFO] Exécution automatique de la phase initiale...")
        run_initial_pipeline()


def run_reranker_if_requested(use_reranker: bool) -> Path:
    """
    If reranker is requested, reuse existing reranker output unless user asks
    to regenerate it. Otherwise return the baseline generations file.
    """
    if not use_reranker:
        print("\n[INFO] Mode sans reranker sélectionné.")
        if not BASE_GENERATIONS_FILE.exists():
            raise FileNotFoundError(f"Fichier introuvable : {BASE_GENERATIONS_FILE}")
        return BASE_GENERATIONS_FILE

    print("\n[INFO] Mode avec reranker sélectionné.")

    if RERANKER_GENERATIONS_FILE.exists():
        print(f"\n[INFO] Fichier existant détecté : {RERANKER_GENERATIONS_FILE}")
        rerun = ask_yes_no(
            "Le reranking a déjà été exécuté. Veux-tu le relancer ?",
            default=False,
        )
        if not rerun:
            print("[INFO] Phase reranker ignorée, fichier existant réutilisé.")
            return RERANKER_GENERATIONS_FILE

    run_command(
        [
            sys.executable,
            "reranker/evaluation/evaluation.py",
            "--output",
            str(RERANKER_GENERATIONS_FILE),
        ],
        "Running reranker evaluation",
    )

    if not RERANKER_GENERATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Le reranker a été exécuté, mais le fichier de sortie est introuvable : "
            f"{RERANKER_GENERATIONS_FILE}"
        )

    return RERANKER_GENERATIONS_FILE


def build_subset_output_path(active_prefix: str, k: int) -> Path:
    return PROCESSED_SUBSET_DIR / f"{active_prefix}_k{k}.jsonl"


def build_answered_output_path(active_prefix: str, k: int) -> Path:
    return ENRICHED_SUBSET_DIR / f"{active_prefix}_k{k}_answered.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete baseline/reranker evaluation pipeline."
    )
    parser.add_argument(
        "--reranker",
        choices=["yes", "no", "ask"],
        default="ask",
        help="Use reranker or not. Default: ask",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=K_VALUES,
        help="List of k values for subset generation. Example: --k-values 5 10 25 50",
    )
    args = parser.parse_args()

    # Step 1: initial phase
    ensure_initial_pipeline()

    # Step 2: reranker
    if args.reranker == "yes":
        use_reranker = True
    elif args.reranker == "no":
        use_reranker = False
    else:
        use_reranker = ask_yes_no("Veux-tu exécuter le pipeline avec reranker ?", default=False)

    active_generations_file = run_reranker_if_requested(use_reranker)
    active_prefix = active_generations_file.stem

    print("\n[INFO] Fichier actif pour la suite du pipeline :", active_generations_file)
    print("[INFO] Préfixe actif :", active_prefix)
    print("[INFO] Valeurs de k :", args.k_values)

    PROCESSED_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    # Step 3: create subsets
    for k in args.k_values:
        subset_output = build_subset_output_path(active_prefix, k)

        run_command(
            [
                sys.executable,
                "utils/prepare_data/subset_of_chunks_from_generations_jsonl.py",
                "--k",
                str(k),
                "--input",
                str(active_generations_file),
                "--output",
                str(subset_output),
            ],
            f"Creating subset for k={k}",
        )

    # Step 4: fill generated answers
    for k in args.k_values:
        subset_input = build_subset_output_path(active_prefix, k)

        if not subset_input.exists():
            raise FileNotFoundError(
                f"Subset file introuvable avant fill_generated_answers_if_empty_with_LLM.py : "
                f"{subset_input}"
            )

        run_command(
            [
                sys.executable,
                "utils/prepare_data/fill_generated_answers_if_empty_with_LLM.py",
                "--input",
                str(subset_input),
            ],
            f"Filling generated answers for k={k}",
        )

    # Step 5: compute RAGAS
    for k in args.k_values:
        answered_input = build_answered_output_path(active_prefix, k)

        if not answered_input.exists():
            raise FileNotFoundError(
                f"Fichier enrichi attendu introuvable : {answered_input}\n"
                f"Vérifie le nom de sortie généré par fill_generated_answers_if_empty_with_LLM.py."
            )

        run_command(
            [
                sys.executable,
                "utils/reranker/compute_ragas.py",
                "--run",
                str(answered_input),
                "--gold",
                str(GOLD_FILE),
            ],
            f"Computing RAGAS for k={k}",
        )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()