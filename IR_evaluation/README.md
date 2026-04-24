# RAG Evaluation Setup

This repository contains a comprehensive evaluation framework for Retrieval-Augmented Generation (RAG) systems. It includes tools for generation, metrics computation, and experimental workflows.

## Prerequisites

### Environment Setup
Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Installation
```bash
pip install -r requirements.txt
```

---

## Project Structure

### 1. **Generation Module** (`generation/`)

The generation module handles querying the RAG system and generating answers.

**What it does:**
- Retrieves relevant chunks from the vector database (Pinecone)
- Queries the RAG system with configurable parameters
- Generates answers using LLM with retrieved context
- Supports optional OCR correction for improved text quality
- Outputs generation results in JSONL format

**Key Parameters:**
- `top_k`: Number of chunks to retrieve (e.g., 1, 3, 5, 10, 20, 50)
- `prompt`: Custom prompt template for the LLM
- `ocr_correction`: Boolean flag to enable/disable OCR correction
- `reranker`: Optional reranking model for improved chunk selection

**Files:**
- `generate.py`: Main generation logic and RAG querying
- `reranking.py`: Can be used, but recommend Google Colab for GPU acceleration (see below for details)

---

### 2. **Metrics Module** (`metrics/`)

The metrics module computes evaluation metrics to assess RAG system performance.

**Available Metrics:**

#### eRAG Metrics (`erag_metrics.py`)
- **Precision@k**: Fraction of relevant chunks in the top-k retrieved results
- **NDCG@k**: Normalized Discounted Cumulative Gain - measures ranking quality of relevant chunks
- Two approaches: LLM-Judged Chunk Contribution (LJCC) and Embedding-Based eRAG
- **Note** In the results we report only LLM-Judged Chunk Contribution (LJCC) scores

#### RAGAS Metrics (`ragas_metrics.py`)
- **Faithfulness**: Measures if the generated answer is grounded in the retrieved context
- **Answer Relevancy**: Evaluates if the answer is relevant to the input question
- **Context Relevance**: Assesses if the retrieved contexts are relevant to the question
- **Answer Correctness**: Compares the generated answer against the ground truth reference
- **LLM correctness**: Uses LLM to judge the correctness of the generated answer

---

## Running Experiments

### **Method 1: Standard Workflow (VS Code / Local)**

#### Step 1: Create a Configuration File
Create or modify a YAML config file in `configs/` directory:
```yaml
# Example: configs/rag_k1.yaml
run_name: "run_baseline_k1"

# Data paths
questions_csv_path: "data/evaluation_questions.csv"

# RAG endpoint configuration
rag_endpoint: "https://stergios.hopto.org/get_answer"

# Retrieval parameters
top_k: 1
reranker_name: "default"

# Model metadata
model_name: "rag_v1"
prompt_version: "v1"

# Output configuration
output_dir: "runs"
```

#### Step 2: Generate Answers
Run the generation script:
```bash
python run.py generate --config configs/rag_k1.yaml
```

**Output:**
- Creates a folder in `runs/` (e.g., `runs/baseline_k1/`)
- Contains:
  - `generations.jsonl`: Generated answers for each question
  - `gold_answers.json`: Ground truth answers

#### Step 3: Compute Metrics
Run the metrics computation:
```bash
python run.py metrics --run runs/baseline_k1
```

**Output:**
- Updates the run folder with:
  - `metrics.json`: Detailed metric scores
  - `summary.json`: Summary statistics

---

### **Method 2: Reranker Workflow (Google Colab)**

**Note:** The reranker requires GPU for optimal performance. Use Google Colab for execution.

#### Step 1: Generate Base Results (Local)
First, generate initial results locally:
```bash
python run.py generate --config configs/rag_k10.yaml
```

#### Step 2: Upload and Run Colab Notebook
1. Open `generate_colab.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Upload the notebook and run all cells

#### Step 3: Import Required Files
The notebook will create a folder structure. Import:
- `generations.jsonl` (from your local `runs/` folder)
- `gold_answers.json` (from your local `runs/` folder)

#### Step 4: Run Reranking
Execute the reranking cells in the notebook:
- Reranks the retrieved chunks using the GPU-accelerated model
- Generates new answers with reranked context

#### Step 5: Export and Compute Metrics (Local)
1. Download the reranked results from Colab
2. Create a new folder in `runs/` (e.g., `runs/reranker_k10/`)
3. Place the downloaded files in this folder
4. Compute metrics:
```bash
python run.py metrics --run runs/reranker_k10
```

---

### **Method 3: BM25 Index Workflow (Colab)**

**For BM25 retrieval experiments:** Use `Prompt_Setup_and_Index_download.ipynb`

#### Step 1: Create BM25 Index
1. Open `Prompt_Setup_and_Index_download.ipynb` in Google Colab
2. Import `evaluation_questions.csv` from `data/` folder
3. Run the notebook to:
   - Create the BM25 index
   - Generate answers using the same prompt template

#### Step 2: Import Results to VS Code
1. Download the generated results from Colab
2. Create a new folder in `runs/` (e.g., `runs/bm25_k5/`)
3. Place the results in this folder

#### Step 3: Compute Metrics (Local)
```bash
python run.py metrics --run runs/bm25_k5
```

---

## Experiment Visualization

Use `Experiment_visualization_plots.ipynb` to:
- Visualize and compare results across different experiments
- Generate plots for metrics (Precision@k, NDCG@k, Faithfulness, etc.)
- Analyze performance trends across different configurations

---

## Output Structure

Each experiment run creates a folder in `runs/` with the following structure:
```
runs/
├── baseline_k5/
│   ├── generations.jsonl      # Generated answers
│   ├── gold_answers.json      # Ground truth answers
│   ├── metrics.json           # Detailed metrics
│   └── summary.json           # Summary statistics
├── reranker_k10/
│   └── ...
└── bm25_k5/
    └── ...
```
