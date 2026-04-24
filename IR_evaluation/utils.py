"""
Shared Utilities
================
Common functions used across all modules.
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def get_git_commit_hash() -> Optional[str]:
    """
    Get current git commit hash if available.
    
    Returns:
        Commit hash string or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
    except Exception:
        pass
    return None


def generate_run_id(prefix: str = "run") -> str:
    """
    Generate a unique run ID based on timestamp.
    
    Args:
        prefix: Prefix for the run ID
        
    Returns:
        Unique run ID string (e.g., "run_20260111_143052")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file into list of dictionaries.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str):
    """
    Save list of dictionaries to JSONL file.
    
    Args:
        records: List of dictionaries
        path: Output path
    """
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output path
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text (lowercase, no punctuation, collapsed whitespace)
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def setup_logging(verbose: bool = True):
    """
    Setup basic logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)
