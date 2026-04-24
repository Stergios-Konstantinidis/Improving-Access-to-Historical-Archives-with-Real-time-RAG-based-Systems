"""
Data Schemas
============
Explicit data schemas for all evaluation artifacts.
Ensures consistent data structures throughout the pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class RetrievedChunk:
    """Schema for a single retrieved chunk."""
    text: str
    rank: int
    score: Optional[float] = None
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class GenerationRecord:
    """
    Schema for a single generation record.
    One record per question in generations.jsonl.
    """
    question_id: str
    question: str
    generated_answer: str
    retrieved_chunks: List[Dict[str, Any]]
    method_metadata: Dict[str, Any]
    latency_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "generated_answer": self.generated_answer,
            "retrieved_chunks": self.retrieved_chunks,
            "method_metadata": self.method_metadata,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict) -> "GenerationRecord":
        return cls(**data)


@dataclass
class MetricRecord:
    """
    Schema for a single metric record.
    One record per question in metrics.json.
    """
    question_id: str
    
    # eRAG metrics (reference-based)
    erag_similarity: Optional[float] = None
    erag_useful: Optional[bool] = None
    
    # RAGAS metrics (reference-free)
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_relevance: Optional[float] = None
    
    # Additional metrics
    qa_f1: Optional[float] = None
    semantic_similarity: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RunMetadata:
    """
    Schema for run metadata (meta.json).
    Contains all information needed to reproduce a run.
    """
    run_id: str
    date: str
    model_name: str
    prompt_version: str
    reranker_name: str
    top_k: int
    git_commit_hash: Optional[str] = None
    rag_endpoint: Optional[str] = None
    erag_threshold_tau: Optional[float] = None
    embedding_model: Optional[str] = None
    llm_judge_model: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        result = {
            "run_id": self.run_id,
            "date": self.date,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "reranker_name": self.reranker_name,
            "top_k": self.top_k,
        }
        if self.git_commit_hash:
            result["git_commit_hash"] = self.git_commit_hash
        if self.rag_endpoint:
            result["rag_endpoint"] = self.rag_endpoint
        if self.erag_threshold_tau is not None:
            result["erag_threshold_tau"] = self.erag_threshold_tau
        if self.embedding_model:
            result["embedding_model"] = self.embedding_model
        if self.llm_judge_model:
            result["llm_judge_model"] = self.llm_judge_model
        if self.extra:
            result["extra"] = self.extra
        return result
    
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "RunMetadata":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class GenerationConfig:
    """Configuration for generation run."""
    # Data paths
    questions_csv_path: str = "data/evaluation_questions.csv"
    
    # RAG endpoint
    rag_endpoint: str = "https://stergios.hopto.org/get_answer"
    
    # Retrieval parameters
    top_k: int = 5
    fetch_k: Optional[int] = None  # Number of chunks to fetch before reranking
    reranker_name: str = "default"  # Options: "default" or "bge-reranker"
    chroma_index: str = "apple_ocr_openai"  # Options: "apple_ocr_openai" or "apple_ocr_gpt_correction_openai_embeddings"
    
    # Model metadata
    model_name: str = "unknown"
    prompt_version: str = "v1"
    
    # Output
    output_dir: str = "runs"
    run_name: Optional[str] = None  # Custom name for the run folder (e.g., "baseline_k5")
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_yaml(cls, path: str) -> "GenerationConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "GenerationConfig":
        return cls(**data)
