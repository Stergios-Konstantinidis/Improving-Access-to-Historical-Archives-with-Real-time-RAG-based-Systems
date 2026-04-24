#!/usr/bin/env python3
# NOTE : please check readme for this --> easier to run on COLAB notebook (CF README.md)
"""
Reranker Module
===============
Supports:
- "default" / "none": No reranking (passthrough)
- "bge-reranker": BAAI/bge-reranker-v2-m3 (multilingual)
"""
from typing import List, Dict, Any, Optional, Callable

_cross_encoder_cache = {}

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

def _load_model(model_name: str = DEFAULT_MODEL):
    """Load or get cached cross-encoder model."""
    global _cross_encoder_cache
    if model_name not in _cross_encoder_cache:
        try:
            from sentence_transformers import CrossEncoder
            print(f"📦 Loading reranker model: {model_name}")
            _cross_encoder_cache[model_name] = CrossEncoder(model_name)
            print(f"   ✓ Model loaded successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: pip install sentence-transformers"
            )
    return _cross_encoder_cache[model_name]

def _rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank chunks using BGE reranker model.
    Returns:Reranked chunks with updated scores and ranks
    """
    if not chunks:
        return []
    model = _load_model(DEFAULT_MODEL)
    # Prepare query-document pairs
    texts = [chunk.get("text", "") for chunk in chunks]
    pairs = [[query, text] for text in texts]
    # Get scores
    scores = model.predict(pairs)
    # Combine chunks with scores
    scored_chunks = []
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        scored_chunks.append({
            **chunk,
            "original_rank": chunk.get("rank", i),
            "original_score": chunk.get("score"),
            "rerank_score": float(score),
        })
    # Sort by rerank score (descending)
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    # Update ranks
    for i, chunk in enumerate(scored_chunks):
        chunk["rank"] = i
        chunk["score"] = chunk["rerank_score"]
    # Limit to top_k if specified
    if top_k is not None:
        scored_chunks = scored_chunks[:top_k]
    return scored_chunks

def get_reranker(name: str, model_name: Optional[str] = None) -> Optional[Callable]:
    if name in ("none", "default") or name.upper() == "HNSW":
        return None
    elif name == "bge-reranker":
        return lambda q, c, k=None: _rerank(q, c, k)
    else:
        raise ValueError(f"Unknown reranker: {name}. Options: default, bge-reranker, HNSW")

def rerank_chunks(
    reranker: Optional[Callable],
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if reranker is None:
        return chunks[:top_k] if top_k else chunks
    try:
        return reranker(query, chunks, top_k)
    except Exception as e:
        print(f"Error during reranking: {e}")
        return chunks[:top_k] if top_k else chunks
