#!/usr/bin/env python3
"""
eRAG Metrics Module
===================
Computes eRAG-based metrics:
- Precision@k: Fraction of relevant chunks
- NDCG@k: Ranking quality of relevant chunks

Two approaches implemented:

1. LLM-Judged Chunk Contribution (LJCC):
   - For each chunk, generate a chunk-isolated answer
   - Judge the answer against gold answer using LLM
   - Returns relevance_score (0-1) for NDCG, binary label for Precision

2. Embedding-Based eRAG (Original Paper Approach):
   - For each chunk, generate a chunk-isolated answer
   - Compute cosine similarity between chunk answer and gold answer embeddings
   - Binary relevance = 1 if similarity >= threshold_tau

Usage:
    from metrics.erag_metrics import compute_erag_metrics, compute_erag_embedding_metrics
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


# =============================================================================
# Utility Functions
# =============================================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_ndcg_at_k(relevance_scores: List[float]) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k).
    Uses continuous relevance scores as gains.
    
    Note: For k=1, NDCG would always be 1.0 if the score is non-zero,
    which is not informative. In this case, we return the raw relevance
    score directly to provide a meaningful metric.
    """
    if not relevance_scores or sum(relevance_scores) == 0:
        return 0.0

    k = len(relevance_scores)
    
    # Special case: k=1 - return raw relevance score instead of 1.0
    # This is more informative as NDCG@1 with a single item would always be 1.0
    if k == 1:
        return relevance_scores[0]

    # DCG with continuous relevance scores
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))

    # Ideal DCG (sort by relevance descending)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))

    return dcg / idcg if idcg > 0 else 0.0


def chunk_relevance(
    client: OpenAI,
    question: str,
    chunk: str,
    gold_answer: str,
    llm_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    LLM-Judged Chunk Contribution (LJCC) for a single chunk.
    
    Process:
    1. Generate a chunk-isolated answer using ONLY the given chunk
    2. Judge the generated answer against the gold answer using LLM
    3. Return relevance_score (0-1) and justification
    
    Args:
        client: OpenAI client
        question: The question being asked
        chunk: The chunk text to evaluate
        gold_answer: The gold standard answer
        llm_model: LLM model to use
        
    Returns:
        {
            "chunk_text": str,
            "chunk_answer": str,
            "relevance_score": float (0-1),
            "justification": str (max 8 words)
        }
    """
    # Handle empty chunk
    if not chunk or not chunk.strip():
        return {
            "chunk_text": chunk,
            "chunk_answer": "INSUFFICIENT",
            "relevance_score": 0.0,
            "justification": "Empty chunk provided"
        }
    
    # Step 1: Generate chunk-isolated answer
    generation_prompt = f"""You are a precise document analyst. Your task is to extract an answer from the given context.

Context:
{chunk}

Question: {question}

INSTRUCTIONS:
1. Answer the question using ONLY information explicitly stated in the context above.
2. Provide a direct, factual answer in 1-2 sentences.
3. Match the language of the question (French question = French answer, English = English).
4. Include specific details: names, dates, numbers, locations when present.
5. If the context contains NO information to answer the question, respond with exactly: "INSUFFICIENT"

Answer:"""

    try:
        gen_response = client.chat.completions.create(
            model=llm_model,
            temperature=0,
            messages=[{"role": "user", "content": generation_prompt}],
        )
        chunk_answer = gen_response.choices[0].message.content.strip()
    except Exception as e:
        return {
            "chunk_text": chunk,
            "chunk_answer": f"ERROR: {str(e)}",
            "relevance_score": 0.0,
            "justification": "Generation failed"
        }
    
    # If chunk was insufficient, score is 0
    insufficient_markers = ["insufficient", "no relevant information", "cannot answer", "no information"]
    chunk_answer_lower = chunk_answer.lower()
    if any(marker in chunk_answer_lower for marker in insufficient_markers):
        return {
            "chunk_text": chunk,
            "chunk_answer": chunk_answer,
            "relevance_score": 0.0,
            "justification": "Chunk insufficient for answer"
        }
    
    # Step 2: Judge the chunk answer against the gold answer
    judge_prompt = f"""You are a strict but fair evaluator comparing two answers to the same question.

Question: {question}

Reference Answer (Ground Truth): {gold_answer}

Candidate Answer (from document chunk): {chunk_answer}

EVALUATION CRITERIA:
- 1.0 = Candidate contains the SAME key facts as reference (dates, names, numbers match)
- 0.8 = Candidate is factually correct but with minor omissions or different wording
- 0.6 = Candidate is partially correct, contains SOME relevant facts from reference
- 0.4 = Candidate mentions the topic but has wrong or missing key information
- 0.2 = Candidate is loosely related but mostly incorrect
- 0.0 = Candidate is completely wrong, unrelated, or contradicts the reference

Focus on FACTUAL ACCURACY, not writing style or completeness.

Return ONLY valid JSON:
{{"relevance_score": <float 0-1>, "justification": "<max 8 words>"}}"""

    try:
        judge_response = client.chat.completions.create(
            model=llm_model,
            temperature=0,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        judge_output = judge_response.choices[0].message.content.strip()
        
        # Parse JSON response - handle potential markdown wrapping
        json_match = re.search(r'\{[^}]+\}', judge_output)
        if json_match:
            judge_result = json.loads(json_match.group())
        else:
            judge_result = json.loads(judge_output)
        
        relevance_score = float(judge_result.get("relevance_score", 0.0))
        justification = str(judge_result.get("justification", ""))[:50]  # Limit length
        
        # Clamp score to [0, 1]
        relevance_score = max(0.0, min(1.0, relevance_score))
        
    except Exception as e:
        return {
            "chunk_text": chunk,
            "chunk_answer": chunk_answer,
            "relevance_score": 0.0,
            "justification": f"Judge parse error"
        }
    
    return {
        "chunk_text": chunk,
        "chunk_answer": chunk_answer,
        "relevance_score": round(relevance_score, 4),
        "justification": justification
    }


def compute_single_erag(
    client: OpenAI,
    question: str,
    gold_answer: str,
    chunks: List[Dict],
    threshold_tau: float = 0.5,
    llm_model: str = "gpt-4o-mini",
) -> Dict:
    """
    Compute eRAG Precision@k and NDCG@k for a single sample using LJCC.
    
    LLM-Judged Chunk Contribution (LJCC) Algorithm:
    1. For each chunk c_i, generate answer a_i using ONLY that chunk
    2. Judge a_i against gold_answer using LLM (returns relevance_score 0-1)
    3. Binary relevance label = 1 if relevance_score >= threshold_tau
    4. Precision@k = sum(binary_labels) / k
    5. NDCG@k = DCG / iDCG using continuous relevance_scores
    
    Returns:
        Dict with precision_at_k, ndcg_at_k, and chunk_details for each chunk
    """
    k = len(chunks)
    
    # Handle edge cases
    if k == 0:
        return {
            "chunk_relevance_precision_at_k": 0.0,
            "chunk_relevance_ndcg_at_k": 0.0,
            "chunk_details": []
        }
    
    if not gold_answer or not gold_answer.strip():
        return {
            "chunk_relevance_precision_at_k": 0.0,
            "chunk_relevance_ndcg_at_k": 0.0,
            "chunk_details": []
        }

    # Extract chunk text robustly
    def get_chunk_text(chunk) -> str:
        if isinstance(chunk, dict):
            return chunk.get("text", chunk.get("content", ""))
        elif isinstance(chunk, str):
            return chunk
        else:
            return str(chunk)
    
    def get_chunk_rank(chunk, idx) -> int:
        if isinstance(chunk, dict):
            return chunk.get("rank", idx)
        return idx

    # Process each chunk with LJCC
    def evaluate_chunk(idx: int, chunk: Any) -> tuple:
        chunk_text = get_chunk_text(chunk)
        chunk_rank = get_chunk_rank(chunk, idx)
        
        result = chunk_relevance(
            client=client,
            question=question,
            chunk=chunk_text,
            gold_answer=gold_answer,
            llm_model=llm_model,
        )
        
        return idx, {
            "rank": chunk_rank,
            "chunk_text": result["chunk_text"],
            "chunk_answer": result["chunk_answer"],
            "relevance_score": result["relevance_score"],
            "justification": result["justification"],
            "is_relevant": result["relevance_score"] >= threshold_tau,
        }

    # Process chunks in parallel
    chunk_details = [None] * k
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(evaluate_chunk, i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            idx, detail = future.result()
            chunk_details[idx] = detail

    # Compute metrics
    relevance_scores = [d["relevance_score"] for d in chunk_details]
    binary_labels = [1 if d["is_relevant"] else 0 for d in chunk_details]
    
    precision_at_k = sum(binary_labels) / k
    ndcg_at_k = compute_ndcg_at_k(relevance_scores)

    return {
        "chunk_relevance_precision_at_k": round(precision_at_k, 4),
        "chunk_relevance_ndcg_at_k": round(ndcg_at_k, 4),
        "chunk_details": chunk_details,
    }


def compute_erag_metrics(
    generations: List[Dict],
    gold_answers: Dict[str, str],
    threshold_tau: float = 0.5,
    max_concurrent_samples: int = 5,
) -> List[Dict]:
    """
    Compute eRAG metrics for all samples using LLM-Judged Chunk Contribution (LJCC).
    
    Args:
        generations: List of generation records
        gold_answers: Dict mapping question_id to gold answer
        threshold_tau: Relevance threshold for binary label (default: 0.5)
        max_concurrent_samples: Max number of samples to process concurrently (default: 5)
        
    Returns:
        List of metric results per sample with:
        - question_id
        - chunk_relevance_precision_at_k
        - chunk_relevance_ndcg_at_k  
        - chunk_details: list of per-chunk evaluation results
    """
    client = OpenAI()
    total = len(generations)
    
    # Counter for progress tracking
    progress = {"count": 0}
    
    def process_sample(idx: int, sample: Dict) -> tuple:
        """Process a single sample and return (index, result) to preserve order."""
        question_id = sample["question_id"]
        metrics = compute_single_erag(
            client=client,
            question=sample["question"],
            gold_answer=gold_answers.get(question_id, ""),
            chunks=sample["retrieved_chunks"],
            threshold_tau=threshold_tau,
        )
        progress["count"] += 1
        print(f"   [eRAG-LJCC] {progress['count']}/{total}: {question_id}")
        return idx, {
            "question_id": question_id,
            **metrics,
        }
    
    # Process samples in parallel while preserving order
    results = [None] * total
    with ThreadPoolExecutor(max_workers=max_concurrent_samples) as executor:
        futures = [
            executor.submit(process_sample, i, sample)
            for i, sample in enumerate(generations)
        ]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
    
    return results


# =============================================================================
# Embedding-Based eRAG (Original Paper Approach)
# =============================================================================

def compute_ndcg_at_k_binary(chunk_labels: List[int]) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k) with binary labels.
    
    Note: For k=1, returns the raw binary label (0 or 1) instead of always 1.0
    """
    if not chunk_labels or sum(chunk_labels) == 0:
        return 0.0

    k = len(chunk_labels)
    
    # Special case: k=1 - return raw label
    if k == 1:
        return float(chunk_labels[0])

    # DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(chunk_labels))

    # Ideal DCG
    ideal_labels = sorted(chunk_labels, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_labels))

    return dcg / idcg if idcg > 0 else 0.0


def compute_single_erag_embedding(
    client: OpenAI,
    question: str,
    gold_answer: str,
    chunks: List[Dict],
    threshold_tau: float = 0.75,
    embedding_model: str = "text-embedding-3-large",
    llm_model: str = "gpt-4o-mini",
) -> Dict:
    """
    Compute eRAG Precision@k and NDCG@k using embedding-based similarity (original paper approach).
    
    eRAG Algorithm (per the paper):
    1. For each chunk c_i, generate answer a_i using ONLY that chunk
    2. Compute semantic similarity: sim(embedding(a_i), embedding(gold_answer))
    3. Label chunk as relevant if sim >= threshold_tau
    4. Precision@k = relevant_chunks / k
    5. NDCG@k = DCG / iDCG (ranking quality)
    
    Args:
        client: OpenAI client
        question: The question being asked
        gold_answer: The gold standard answer
        chunks: List of retrieved chunks
        threshold_tau: Similarity threshold for relevance (default: 0.75)
        embedding_model: Model for computing embeddings
        llm_model: Model for generating chunk answers
        
    Returns:
        Dict with precision_at_k, ndcg_at_k metrics
    """
    k = len(chunks)
    
    # Handle edge cases
    if k == 0:
        return {"erag_precision_at_k": 0.0, "erag_ndcg_at_k": 0.0}
    
    if not gold_answer or not gold_answer.strip():
        return {"erag_precision_at_k": 0.0, "erag_ndcg_at_k": 0.0}

    # Extract chunk text robustly
    def get_chunk_text(chunk) -> str:
        if isinstance(chunk, dict):
            return chunk.get("text", chunk.get("content", ""))
        elif isinstance(chunk, str):
            return chunk
        else:
            return str(chunk)

    # 1. Generate per-chunk answers in parallel
    def generate_chunk_answer(idx: int, chunk_text: str) -> tuple:
        if not chunk_text or not chunk_text.strip():
            return idx, ""
        prompt = f"""You are a precise document analyst. Extract an answer from the context.

Context: {chunk_text}

Question: {question}

INSTRUCTIONS:
- Answer using ONLY information from the context above.
- Be direct and factual (1-2 sentences).
- Match the question's language (French/English).
- Include specific details: names, dates, numbers, locations.
- If no relevant information exists, say "No relevant information found."

Answer:"""
        try:
            response = client.chat.completions.create(
                model=llm_model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return idx, response.choices[0].message.content.strip()
        except Exception as e:
            print(f"      Warning: Failed to generate chunk answer: {e}")
            return idx, ""

    chunk_answers = [None] * k
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(generate_chunk_answer, i, get_chunk_text(chunk))
            for i, chunk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            idx, answer = future.result()
            chunk_answers[idx] = answer

    # 2. Batch embed gold answer + chunk answers
    texts = [gold_answer] + [a if a else " " for a in chunk_answers]
    try:
        response = client.embeddings.create(model=embedding_model, input=texts)
        embeddings = [item.embedding for item in response.data]
    except Exception as e:
        print(f"      Warning: Failed to compute embeddings: {e}")
        return {"erag_precision_at_k": 0.0, "erag_ndcg_at_k": 0.0}

    emb_gold = embeddings[0]
    chunk_embeddings = embeddings[1:]

    # 3. Compute labels based on similarity threshold
    chunk_labels = []
    similarities = []
    for chunk_answer, emb_chunk in zip(chunk_answers, chunk_embeddings):
        if not chunk_answer:
            chunk_labels.append(0)
            similarities.append(0.0)
        else:
            sim = cosine_similarity(emb_chunk, emb_gold)
            similarities.append(sim)
            chunk_labels.append(1 if sim >= threshold_tau else 0)

    # 4. Compute metrics
    precision_at_k = sum(chunk_labels) / k
    ndcg_at_k = compute_ndcg_at_k_binary(chunk_labels)

    return {
        "erag_precision_at_k": round(precision_at_k, 4),
        "erag_ndcg_at_k": round(ndcg_at_k, 4),
    }


def compute_erag_embedding_metrics(
    generations: List[Dict],
    gold_answers: Dict[str, str],
    threshold_tau: float = 0.75,
    max_concurrent_samples: int = 5,
) -> List[Dict]:
    """
    Compute eRAG metrics using embedding-based similarity (original paper approach).
    
    Args:
        generations: List of generation records
        gold_answers: Dict mapping question_id to gold answer
        threshold_tau: Similarity threshold for relevance (default: 0.75)
        max_concurrent_samples: Max number of samples to process concurrently (default: 5)
        
    Returns:
        List of metric results per sample (in original order) with:
        - question_id
        - erag_precision_at_k
        - erag_ndcg_at_k
    """
    client = OpenAI()
    total = len(generations)
    
    # Counter for progress tracking (thread-safe via GIL for simple increments)
    progress = {"count": 0}
    
    def process_sample(idx: int, sample: Dict) -> tuple:
        """Process a single sample and return (index, result) to preserve order."""
        question_id = sample["question_id"]
        metrics = compute_single_erag_embedding(
            client=client,
            question=sample["question"],
            gold_answer=gold_answers.get(question_id, ""),
            chunks=sample["retrieved_chunks"],
            threshold_tau=threshold_tau,
        )
        progress["count"] += 1
        print(f"   [eRAG-Embed] {progress['count']}/{total}: {question_id}")
        return idx, {
            "question_id": question_id,
            **metrics,
        }
    
    # Process samples in parallel while preserving order
    results = [None] * total
    with ThreadPoolExecutor(max_workers=max_concurrent_samples) as executor:
        futures = [
            executor.submit(process_sample, i, sample)
            for i, sample in enumerate(generations)
        ]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
    
    return results
