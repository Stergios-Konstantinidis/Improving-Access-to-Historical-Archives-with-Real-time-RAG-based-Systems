# Improving Access to Historical Archives with Real-time RAG-based Systems

## Overview

This repository accompanies the paper *"Improving Access to Historical Archives with Real-time RAG-based Systems"*, which presents an end-to-end archival processing and retrieval framework that integrates large language models (LLMs) into the data ingestion and query pipeline of digitized historical archives.

The system introduces two core components:

1. **LLM-based OCR Correction** — A post-OCR refinement operator that uses instruction-tuned LLMs to improve transcription quality of noisy historical text.
2. **Semantic Retrieval & RAG Pipeline** — A dense retrieval + cross-encoder reranking pipeline supporting natural-language question answering via retrieval-augmented generation (RAG).

---

## Motivation

Digitized historical archives (newspapers, magazines, printed material spanning several centuries) are among the most important publicly accessible cultural heritage collections. However, current access methods rely heavily on:

- **Noisy OCR output** — Degraded scans, old-style fonts, and varying document conditions lead to significant transcription errors.
- **Rigid keyword-based retrieval** — Traditional search interfaces fail to capture user intent and semantic meaning.

This work reframes archival access as a **data systems problem** involving ingestion, representation, indexing, retrieval, and interactive serving — rather than a simple search problem.

---

## Architecture

The pipeline spans the full lifecycle from raw scan to conversational query:

```
Scanned PDFs → Layout Segmentation → OCR → LLM Post-Correction → Embedding & Indexing → Dense Retrieval → Cross-Encoder Reranking → Streaming LLM Answer Generation
```

### Ingestion Pipeline

1. **Layout Segmentation** — Digitized issues (PDF) are segmented into individual articles using the [LayoutParser](https://github.com/Layout-Parser/layout-parser) framework.
2. **OCR** — Each article segment is processed by an OCR engine to produce machine-readable text.
3. **LLM Post-Correction** — Instruction-tuned LLMs refine the raw OCR output using engineered prompts and augmented context, preserving structure and semantic faithfulness while reducing transcription errors.
4. **Embedding & Indexing** — Both raw and corrected texts are embedded using dense vector representations and indexed via a hierarchical graph-based approximate nearest-neighbor (ANN) structure for sublinear query time.

### Query Pipeline

1. **Dense Retrieval** — User queries are embedded and matched against the ANN index to retrieve the top-k candidate documents (optimized for recall and speed).
2. **Cross-Encoder Reranking** — A cross-encoder model jointly evaluates each query–document pair for refined relevance scoring.
3. **Streaming LLM Generation** — A grounded answer is generated conditioned on the query and retrieved context, with tokens streamed incrementally to reduce perceived latency.

### Latency-Aware Design

The system is designed for interactive use with response times on the order of **~1.3 seconds**:

- **ANN indexing** for scalable, sublinear retrieval
- **Lightweight reranking** (MiniLM-based cross-encoders) balancing quality vs. latency
- **Streaming generation** for immediate partial output delivery

---

## Dataset

Experiments were conducted on a **proprietary curated subset of historical newspaper issues** from the [Cantonal and University Library of Lausanne (BCUL)](https://www.scriptorium.ch), obtained via Scriptorium. This dataset is novel, highly diverse, and used in a research publication for the first time.

| Property | Details |
|---|---|
| **Source** | 9 periodicals from the BCUL Scriptorium collection |
| **Temporal coverage** | Late 18th century (1762) through 2001 |
| **Segments** | 500,000 curated article segments |
| **Queries** | 384 natural-language queries (including 5 unanswerable questions for hallucination testing) |
| **OCR Ground Truth** | 94 manually transcribed document snippets |

---

## Experiments & Key Results

### Research Questions

1. **Do LLM sanitizations improve the quality of raw OCR text?**
2. **Do improvements in OCR quality translate into downstream information retrieval benefits?**
3. **How does the system balance retrieval quality and response time?**

### OCR Correction (Post-OCR LLM Refinement)

Six OCR engines were benchmarked: **Tesseract**, **Docling**, **EasyOCR**, **Apple Vision OCR**, **Azure Document AI**, and **Google Document AI**.

Multiple LLMs were evaluated for post-OCR correction: **Gemma 3**, **Qwen3 (235B)**, **DeepSeek-r1**, **GPT-4o / GPT-5 / GPT-5.2**, and **Gemini 2.5 Pro / 3 Pro / 3 Flash**.

**Key finding:** OCR + LLM post-processing achieves **CER < 6%** and **WER < 14%** on challenging historical documents. Larger LLMs (GPT-4o, Gemini 2.5 Pro) deliver the most stable improvements, while smaller models (e.g., Gemma 3) can *degrade* performance due to poor instruction-following.

### Information Retrieval (RAG Evaluation)

Evaluated using [RAGAS](https://docs.ragas.io/) metrics: **Answer Correctness**, **Context Relevancy**, **Answer Relevancy**, and **NDCG@k** across 384 queries.

| Method | NDCG@10 | Answer Correctness | Context Relevancy | Answer Relevancy |
|---|---|---|---|---|
| BM25 (keyword) | 36.39% | 35.17% | 43.62% | 48.07% |
| Semantic (dense) | 63.11% | 50.20% | 69.86% | 71.79% |
| Semantic + LLM correction | 62.88% | 49.43% | 69.86% | 72.49% |
| **Ours (Semantic + LLM + Reranker)** | **73.06%** | **54.10%** | **73.70%** | **74.57%** |

**Key findings:**
- The full reranking pipeline increases **NDCG@10 by ~100%** over BM25 (36.39% → 73.06%) and **Context Relevancy by ~69%** (43.62% → 73.70%).
- LLM-corrected OCR yields a **9.45% relative improvement in Answer Correctness** over the uncorrected semantic baseline.
- The **reranker is the definitive driver** of overall retrieval quality; LLM OCR correction stabilizes factual grounding.

### System Response Time

| Configuration | Mean Response Time |
|---|---|
| BGE Reranker (high quality) | > 2.5 s (reranking alone) |
| MiniLM Reranker + Streaming | **~1.3 s** (end-to-end) |

---

## Application Demo

An immersive archive exploration interface was deployed on an **Apple Vision Pro** device, enabling:

- Natural-language queries over historical archives
- Generated responses with supporting archival evidence
- Provenance inspection of original source documents
- Geographical mapping of locations mentioned in retrieved documents

---

## Technologies & Models Used

### OCR Engines
- [Tesseract](https://github.com/tesseract-ocr/tesseract)
- [Docling](https://github.com/DS4SD/docling)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Apple Vision OCR](https://developer.apple.com/documentation/vision/vnrecognizetextrequest)
- Azure Document AI
- Google Document AI

### Large Language Models
- Gemma 3, Qwen3 (235B), DeepSeek-r1
- GPT-4o, GPT-5, GPT-5.2
- Gemini 2.5 Pro, Gemini 3 Pro, Gemini 3 Flash

### Retrieval & Indexing
- Dense embeddings (Google embeddings-001)
- ANN indexing (hierarchical graph-based)
- Cross-encoder reranking (MiniLM-based: `mmarco-mMiniLMv2-L12-H384-v1`)
- BM25 (keyword baseline)
- [RAGAS](https://docs.ragas.io/) for evaluation

### Document Processing
- [LayoutParser](https://layout-parser.github.io/) for layout segmentation
- [DocLayNet](https://github.com/DS4SD/DocLayNet) for document layout analysis

---

## Citation

```bibtex
@article{improving_access_historical_archives_2026,
  title={Improving Access to Historical Archives with Real-time RAG-based Systems},
  journal={Journal of the Association for Information Science and Technology},
  year={2026},
  note={Under review}
}
```

---

## License

Please refer to the paper and dataset agreements for usage terms. The BCUL dataset is proprietary; partial release is planned upon clearance with the data owner.
