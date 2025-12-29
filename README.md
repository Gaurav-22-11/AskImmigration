# AskImmigration  
**Hybrid Retrieval-Augmented Question Answering for U.S. Immigration Policy**

AskImmigration is a Retrieval-Augmented Generation (RAG) system that answers U.S. immigration-related questions **strictly using official U.S. government sources** (USCIS, DHS, ICE, Travel.State.gov).  
The system combines **hybrid retrieval (BM25 + dense embeddings)**, **cross-encoder reranking**, and **NLI-based verification** to ensure grounded and reliable answers.

This repository contains **end-to-end code** for:
- data collection
- preprocessing & chunking
- index construction
- retrieval evaluation
- question answering via FastAPI or CLI

---

## 1. Repository Structure

IMMI/
├── api/
│ └── main.py # FastAPI backend (Uvicorn)
├── artifacts/
│ ├── bm25_nodes.jsonl # Serialized BM25 nodes
│ └── faiss_llamaindex.index # FAISS dense index
├── data/
│ ├── raw/ # Raw scraped text (Playwright output)
│ ├── processed/ # Cleaned + chunked corpus (JSONL)
│ └── eval/
│ └── eval.jsonl # Evaluation questions + gold chunk IDs
├── frontend/
│ └── index.html # Simple browser UI
├── images/
│ └── askimmi.drawio (2).png # System architecture diagram
├── rag_llamaindex/
│ ├── build_index.py # Build BM25 + FAISS indices
│ ├── query.py # Main RAG pipeline
│ ├── reranker.py # Cross-encoder reranking
│ ├── postprocess.py # Filtering + cleanup
│ └── settings.py # Model configuration
├── scripts/
│ ├── ask.py # CLI querying
│ ├── eval_retrievers.py # Retriever evaluation (Recall/MRR/nDCG)
│ └── make_eval_synthetic.py # (Optional) synthetic eval generation
├── src/
│ ├── ingest_playwright.py # Data collection using Playwright
│ └── chunk.py # Chunking + preprocessing
├── .env # Gemini API key (provided)
├── requirements.txt
├── README.md
└── runtime.txt

## 2. Environment Setup (From Scratch)

### 2.1 Create and activate a virtual environment

python -m venv .venv

macOS / Linux
source .venv/bin/activate

Windows (PowerShell)
.venv\Scripts\Activate.ps1

### 2.2 Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

### 2.3 Install Playwright browsers (required)
python -m playwright install

## 3. Gemini API Key (Provided)

This project uses Gemini for answer generation.

A Gemini API key is provided in the .env to facilitate evaluation.

## 4. Data Collection (Playwright Scraping)

The system collects data from official sources using Playwright.

Run: python src/ingest_playwright.py --url-file data/raw/seed_urls.txt --out data/processed/corpus.jsonl

## 5. Preprocessing and Chunking

This step:
cleans text
removes duplicates
chunks documents (with overlap)
attaches metadata (URL, agency, title)

Run: python chunk.py --in data/processed/corpus.jsonl --out data/processed/chunks.jsonl

## 6. Build Retrieval Indices

This step builds:

BM25 index (lexical)
FAISS dense index (embeddings)

Run: python rag_llamaindex/build_index.py

## 7. Retriever Evaluation

Evaluation compares BM25, Dense, and Hybrid retrieval using:
Recall@k
MRR@k
nDCG@k

Run: python scripts/eval_retrievers.py --eval-file data/eval/eval.jsonl --k 10

## 8. Querying the System (CLI)

Run a question directly from the terminal:

python scripts/ask.py "YOUR QUESTION HERE"

## 9. Web Application (FastAPI)

Start the backend:

uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

Then open:
http://127.0.0.1:8000

The frontend UI (frontend/index.html) sends queries to this API.
