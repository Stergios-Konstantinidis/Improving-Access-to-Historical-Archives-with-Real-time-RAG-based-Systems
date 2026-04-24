#!/usr/bin/env python3
"""
RAGAS Metrics Module
====================
Computes RAGAS-based metrics:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Relevance: Are the retrieved contexts relevant?
- Answer Correctness: Is the answer correct compared to reference?

Usage:
    from metrics.ragas_metrics import compute_ragas_metrics
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    Faithfulness,
    ContextRelevance,
    AnswerCorrectness,
)


def is_no_answer(text: str) -> bool:
    """Check if the answer is a non-answer like 'I don't know' or 'Je ne sais pas'."""
    no_answer_phrases = [
        "i don't know",
        "i do not know",
        "je ne sais pas",
        "i'm not sure",
        "i am not sure",
        "cannot answer",
        "unable to answer",
    ]
    text_lower = text.lower().strip()
    return any(phrase in text_lower for phrase in no_answer_phrases)


# Load generations from a JSONL file
def load_generations(filepath: str) -> List[Dict]:
    """Load generations from a JSONL file."""
    generations = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line))
    return generations


async def evaluate_llm_correctness(
    client: AsyncOpenAI,
    question: str,
    generated_answer: str,
    gold_answer: str,
) -> float:
    """Evaluate if generated answer is correct compared to gold answer using LLM.
    
    Returns:
        1.0 if factually correct
        0.5 if partially correct
        0.0 if incorrect
        -1.0 if evaluation fails
    """
    prompt = f"""You are an evaluation assistant. Compare the generated answer with the ground truth answer for the given question.

Question: {question}

Generated Answer: {generated_answer}

Ground Truth Answer: {gold_answer}

Evaluate if the generated answer is factually correct compared to the ground truth.
Respond with ONLY a single number:
- 1 if the generated answer is factually correct
- 0.5 if the generated answer is partially correct
- 0 if the generated answer is incorrect

Your response (only the number):"""
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an evaluation assistant that compares answers and returns only a numeric score."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10,
        )
        
        result_text = response.choices[0].message.content.strip()
        score = float(result_text)
        
        # Validate score is one of the expected values
        if score in [0.0, 0.5, 1.0]:
            return score
        else:
            print(f"      Warning: Unexpected LLM correctness score: {score}, defaulting to 0.0")
            return 0.0
            
    except Exception as e:
        print(f"      Warning: LLM correctness evaluation failed: {e}")
        return -1.0


async def evaluate_single_sample(
    sample: Dict,
    relevancy_scorer: AnswerRelevancy,
    faithfulness_scorer: Faithfulness,
    context_relevance_scorer: ContextRelevance,
    correctness_scorer: AnswerCorrectness,
    gold_answer: str,
    client: AsyncOpenAI,
) -> Dict:
    """Evaluate a single sample with RAGAS metrics."""
    question = sample["question"]
    response = sample["generated_answer"]
    
    # Robust chunk text extraction
    retrieved_contexts = []
    for chunk in sample.get("retrieved_chunks", []):
        if isinstance(chunk, dict):
            text = chunk.get("text", chunk.get("content", ""))
        elif isinstance(chunk, str):
            text = chunk
        else:
            text = str(chunk)
        if text and text.strip():
            retrieved_contexts.append(text)
    
    # Handle empty contexts
    if not retrieved_contexts:
        retrieved_contexts = ["No context available."]

    try:
        # Check if answer is a non-answer for faithfulness
        is_non_answer = is_no_answer(response)
        
        # Compute metrics concurrently
        relevancy_task = relevancy_scorer.ascore(
            user_input=question,
            response=response,
        )

        # Only compute faithfulness if it's not a non-answer
        if is_non_answer:
            faithfulness_task = None
        else:
            faithfulness_task = faithfulness_scorer.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )

        context_relevance_task = context_relevance_scorer.ascore(
            user_input=question,
            retrieved_contexts=retrieved_contexts,
        )
        
        correctness_task = correctness_scorer.ascore(
            user_input=question,
            response=response,
            reference=gold_answer if gold_answer else "No reference available.",
        )
        
        # LLM correctness evaluation
        llm_correctness_task = evaluate_llm_correctness(
            client=client,
            question=question,
            generated_answer=response,
            gold_answer=gold_answer if gold_answer else "No reference available.",
        )

        # Await all tasks concurrently
        tasks_to_gather = [relevancy_task, context_relevance_task, correctness_task, llm_correctness_task]
        if faithfulness_task is not None:
            tasks_to_gather.insert(1, faithfulness_task)
        
        results = await asyncio.gather(*tasks_to_gather)
        
        if faithfulness_task is not None:
            relevancy_result, faithfulness_result, context_relevance_result, correctness_result, llm_correctness_result = results
            faithfulness_value = faithfulness_result.value if faithfulness_result.value is not None else 0.0
        else:
            relevancy_result, context_relevance_result, correctness_result, llm_correctness_result = results
            faithfulness_value = -1.0  # Mark as non-answer

        return {
            "question_id": sample["question_id"],
            "answer_relevancy": relevancy_result.value if relevancy_result.value is not None else 0.0,
            "faithfulness": faithfulness_value,
            "context_relevance": context_relevance_result.value if context_relevance_result.value is not None else 0.0,
            "answer_correctness": correctness_result.value if correctness_result.value is not None else 0.0,
            "llm_correct_or_not": llm_correctness_result,
        }
    except Exception as e:
        print(f"      Warning: RAGAS evaluation failed for {sample['question_id']}: {e}")
        return {
            "question_id": sample["question_id"],
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "context_relevance": 0.0,
            "answer_correctness": 0.0,
            "llm_correct_or_not": 0.0,
        }


async def run_ragas_evaluation(
    generations: List[Dict],
    gold_answers: Dict[str, str],
    max_concurrent_samples: int = 5,
) -> List[Dict]:
    """
    Run RAGAS evaluation on all samples with concurrent processing.

    Args:
        generations: List of generation records
        gold_answers: Dictionary mapping question_id to gold answer
        max_concurrent_samples: Max number of samples to evaluate concurrently (default: 5)

    Returns:
        List of evaluation results for each sample (in original order)
    """
    # Initialize OpenAI client
    client = AsyncOpenAI()

    # Initialize LLM
    llm = llm_factory(
        "gpt-4o-mini",
        client=client,
    )
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=client,
    )

    # Initialize scorers
    relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    faithfulness_scorer = Faithfulness(llm=llm)
    context_relevance_scorer = ContextRelevance(llm=llm)
    correctness_scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)

    # Use semaphore to limit concurrency and avoid rate limiting
    semaphore = asyncio.Semaphore(max_concurrent_samples)
    total = len(generations)
    progress = {"count": 0}
    
    async def process_sample_with_semaphore(idx: int, sample: Dict) -> tuple:
        """Process a single sample with semaphore control, return (index, result)."""
        async with semaphore:
            gold_answer = gold_answers.get(sample["question_id"], "")
            result = await evaluate_single_sample(
                sample=sample,
                relevancy_scorer=relevancy_scorer,
                faithfulness_scorer=faithfulness_scorer,
                context_relevance_scorer=context_relevance_scorer,
                correctness_scorer=correctness_scorer,
                gold_answer=gold_answer,
                client=client,
            )
            progress["count"] += 1
            print(f"   [RAGAS] {progress['count']}/{total}: {sample['question_id']}")
            return idx, result

    # Create tasks for all samples
    tasks = [
        process_sample_with_semaphore(i, sample)
        for i, sample in enumerate(generations)
    ]
    
    # Run all tasks concurrently (semaphore limits actual concurrency)
    indexed_results = await asyncio.gather(*tasks)
    
    # Sort by index to preserve original order
    indexed_results.sort(key=lambda x: x[0])
    results = [result for _, result in indexed_results]

    return results


def compute_ragas_metrics(generations: List[Dict], gold_answers: Dict[str, str]) -> List[Dict]:
    """
    Compute RAGAS metrics for all samples.
    
    Args:
        generations: List of generation records
        gold_answers: Dictionary mapping question_id to gold answer
        
    Returns:
        List of metric results per sample
    """
    return asyncio.run(run_ragas_evaluation(generations, gold_answers))
