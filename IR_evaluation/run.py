#!/usr/bin/env python3
"""
RAG Evaluation Pipeline
=======================
Main entry point for running the evaluation pipeline.

Usage:
    # Run generation
    python run.py generate --config configs/rag_k1.yaml
    
    # Run metrics on a run directory
    python run.py metrics --run runs/baseline_k1
    
    # Run metrics on a single .jsonl file (for testing)
    python run.py metrics --run temp_generation.jsonl
"""

import sys
import argparse
from pathlib import Path

# Add this directory to path
sys.path.insert(0, str(Path(__file__).parent))


def cmd_generate(args):
    """Run the generation pipeline."""
    from generation.generate import main as generate_main
    
    # Rebuild sys.argv for the module
    new_argv = ['generate', '--config', args.config]
    if args.quiet:
        new_argv.append('--quiet')
    
    sys.argv = new_argv
    generate_main()


def cmd_metrics(args):
    """Run the metrics pipeline."""
    from metrics import run_metrics
    
    run_metrics(args.run)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Run generation pipeline")
    gen_parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    gen_parser.add_argument("--quiet", action="store_true", help="Suppress output")
    
    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Run metrics computation")
    metrics_parser.add_argument("--run", type=str, required=True, help="Path to run directory or single .jsonl file")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
