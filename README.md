# Grounding Generative Answers in Verifiable Evidence with Retrieval on Historical Newspaper Archives

## Overview

This repository accompanies the paper *"Grounding Generative Answers in Verifiable Evidence with Retrieval on Historical Newspaper Archives"*, which presents an end-to-end framework for source-grounded archival question answering. The system combines LLM-based OCR correction with hybrid lexical-semantic retrieval and retrieval-augmented generation (RAG), so that answers are generated from archival evidence rather than model memory alone.

The system introduces three core components:

1. **LLM-based OCR Correction**: A post-OCR refinement operator that uses instruction-tuned LLMs to improve transcription quality of noisy historical text.
2. **Hybrid Retrieval with Rank Fusion**: BM25s lexical retrieval, dense retrieval, and date-filtered retrieval combined through weighted reciprocal rank fusion (RRF), followed by cross-encoder reranking.
3. **Grounded Answer Generation**: Retrieval-augmented generation with streaming output, exposing the retrieved sources behind each answer.

---

## Motivation

Generative AI produces fluent answers, but trust depends on whether answers can be traced to verifiable evidence. This is especially important for primary sources such as historical newspapers. Current access methods for digitized archives rely heavily on:

- **Noisy OCR output**: Degraded scans, old-style fonts, and varying document conditions lead to significant transcription errors.
- **Rigid keyword-based retrieval**: Traditional search interfaces fail to capture user intent and semantic meaning.

This work reframes archival access as a **data systems problem** involving ingestion, representation, indexing, retrieval, and interactive serving, rather than a simple search problem.

---

## Architecture

The pipeline spans the full lifecycle from raw scan to conversational query:

```
Scanned PDFs → Layout Segmentation → OCR → LLM Post-Correction → Embedding & Indexing
            → Hybrid Retrieval (BM25s + Dense + Temporal) → Weighted RRF Fusion
            → Cross-Encoder Reranking → Streaming LLM Answer Generation
```

### Ingestion Pipeline

1. **Layout Segmentation**: Digitized issues (PDF) are segmented into individual articles using the [LayoutParser](https://github.com/Layout-Parser/layout-parser) framework.
2. **OCR**: Each article segment is processed by an OCR engine to produce machine-readable text.
3. **LLM Post-Correction**: Instruction-tuned LLMs refine the raw OCR output using engineered prompts and augmented context, preserving structure and semantic faithfulness while reducing transcription errors.
4. **Embedding & Indexing**: Both raw and corrected corpus views are embedded and indexed in Chroma for dense retrieval; a separate BM25s index is built on each corpus view.

### Query Pipeline

1. **Hybrid Retrieval**: The query is served by three retrievers: BM25s lexical retrieval, dense retrieval over Chroma, and a date-filtered dense retrieval when the query contains explicit temporal constraints.
2. **Weighted Reciprocal Rank Fusion**: Candidate lists are merged with weighted RRF (K = 60, weights 0.6 for BM25s, 0.5 for dense, 0.5 for temporal, calibrated on a 50-query subset).
3. **Cross-Encoder Reranking**: A cross-encoder model jointly evaluates each query-document pair from the fused candidate pool for refined relevance scoring.
4. **Streaming LLM Generation**: A grounded answer is generated conditioned on the query and retrieved context, with tokens streamed incrementally to reduce perceived latency.

### Latency-Aware Design

The system is designed for interactive use with response times of about **1.3 seconds** end-to-end:

- **ANN indexing** for scalable, sublinear dense retrieval
- **Lightweight reranking** (MiniLM-based cross-encoders) balancing quality vs. latency
- **Streaming generation** for immediate partial output delivery

---

## Dataset

Experiments were conducted on a **proprietary curated subset of historical newspaper issues** from the [Cantonal and University Library of Lausanne (BCUL)](https://www.scriptorium.ch), obtained via Scriptorium. This dataset is novel and highly diverse.

| Property | Details |
|---|---|
| **Source** | 9 periodicals from the BCUL Scriptorium collection |
| **Temporal coverage** | 1762 through 2001 |
| **Segments** | 500,000 curated article segments with structured metadata (dates, bounding boxes, NER, summaries) |
| **Queries** | 384 manually formulated natural-language queries |
| **OCR Ground Truth** | ~700 manually transcribed document snippets |

---

## Experiments & Key Results

### Research Questions

1. **Do LLM sanitizations improve the quality of raw OCR text?**
2. **Do improvements in OCR quality translate into downstream information retrieval benefits?**
3. **How does the system balance retrieval quality and response time?**

### OCR Correction (Post-OCR LLM Refinement)

Five OCR engines were benchmarked: **Tesseract**, **EasyOCR**, **Apple Vision OCR**, **Azure Document Intelligence**, and **Google Document AI**.

Ten LLMs were evaluated for post-OCR correction: **Gemma 4**, **DeepSeek V4**, **Qwen3 32B**, **Qwen3.7 Max**, **GPT-4o / GPT-5 / GPT-5.2**, and **Gemini 3 Flash / 3.1 Pro / 3.1 Flash Lite**.

**Key findings:**

- The best model-engine combination (Google Document AI + Qwen3.7 Max) reaches a **CER of about 1.5%** and a **WER of about 5%** on challenging historical documents.
- LLM refinement reduces error rates on average across engines by up to **44.5% (CER, Gemini 3.1 Pro)** and **61.0% (WER, Qwen3.7 Max)**.
- Larger models deliver the most stable improvements, while smaller models (e.g., Qwen3 32B) can *degrade* character-level fidelity due to poor instruction-following.

### Information Retrieval (RAG Evaluation)

Evaluated with [RAGAS](https://docs.ragas.io/) metrics (**Answer Correctness**, **Context Relevance**) and **NDCG@k** across 384 queries, at retrieval depths k in {5, 10, 25}. Results at k = 5:

| Method | Context Relevance | Answer Correctness | NDCG@5 |
|---|---|---|---|
| BM25s (keyword) | 69.27% | 52.22% | 65.14% |
| Semantic (dense) | 67.32% | 51.69% | 63.10% |
| Semantic + BM25s (RRF fusion) | 73.83% | 60.22% | 75.42% |
| Semantic + BM25s + Reranker | 83.92% | 64.71% | 85.73% |
| **Ours (LLM correction + Fusion + Reranker)** | **86.72%** | **67.40%** | **87.79%** |

**Key findings:**

- Hybrid rank fusion and cross-encoder reranking are the primary drivers of retrieval quality: the full pipeline improves **NDCG@5 by 34.8% relative to BM25s** (65.14% → 87.79%) and Answer Correctness from 52.22% to 67.40%.
- OCR correction alone yields marginal and mixed changes, but once reranking is applied, the corrected corpus view wins consistently on all metrics at all depths.

### System Response Time

| Configuration | Mean Response Time |
|---|---|
| BGE Reranker (high quality) | > 2.5 s (reranking alone) |
| MiniLM Reranker + Streaming | **~1.3 s** (end-to-end) |

---

## Technologies & Models Used

### OCR Engines
- Tesseract
- EasyOCR
- Apple Vision OCR
- Azure Document Intelligence
- Google Document AI

### Large Language Models
- Gemma 4, DeepSeek V4, Qwen3 32B, Qwen3.7 Max
- GPT-4o, GPT-5, GPT-5.2
- Gemini 3 Flash, Gemini 3.1 Pro, Gemini 3.1 Flash Lite
- Deployment: Gemini 3 Flash (OCR correction), GPT-4.1 mini (answer generation), GPT-4o mini (LLM judge for NDCG grading)

### Retrieval & Indexing
- Dense embeddings (text-embedding-2-preview) indexed in Chroma
- BM25s lexical retrieval
- Weighted reciprocal rank fusion (BM25s + dense + date-filtered retrieval)
- Cross-encoder reranking (MiniLM-based: `mmarco-mMiniLMv2-L12-H384-v1`)
- RAGAS for evaluation

### Document Processing
- LayoutParser for layout segmentation
- DocLayNet for document layout analysis

---

## Citation

```bibtex
@article{grounding_generative_answers_2026,
  title = {Grounding Generative Answers in Verifiable Evidence with Retrieval on Historical Newspaper Archives},
  year  = {2026},
  note  = {Under review}
}
```

---

## License

Please refer to the paper and dataset agreements for usage terms. The BCUL dataset is proprietary; partial release is planned upon clearance with the data owner.
