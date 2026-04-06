# FinSight RAG

FinSight is a financial-document RAG application that can ingest PDFs, retrieve relevant chunks, rerank with Cohere, and answer with OpenAI models.

It includes:
- A FastAPI backend for ingestion, chat, and metrics
- A Streamlit frontend
- A local persistent Chroma vector database
- RAGAs evaluation scripts

## Tech Stack
- Python 3.9+
- FastAPI + Uvicorn
- Streamlit
- LangChain
- OpenAI (chat + embeddings)
- Cohere reranker
- Chroma vector store
- RAGAs + Hugging Face Datasets

## Project Structure
- `api/` API server
- `ingestion/` PDF loading, chunking, embedding, storage
- `retrieval/` query rewrite + vector retrieval + reranking
- `generation/` answer generation prompt chain
- `evaluation/` offline quality evaluation with RAGAs
- `app.py` Streamlit UI
- `requirements.txt` Python dependencies

## Prerequisites
1. Python 3.9 or newer
2. OpenAI API key
3. Cohere API key

## Setup
1. Clone the repo and move into it.
2. Create a virtual environment.
3. Install dependencies.
4. Create your `.env` file from `.env.example`.

Example commands:

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Then edit `.env` and set your real keys:

```env
OPENAI_API_KEY=...
COHERE_API_KEY=...
```

## Run The App (End-to-End)

Open two terminals.

Terminal 1: start the API

```bash
python3 -m uvicorn api.main:app --reload --port 8000
```

Terminal 2: start Streamlit UI

```bash
python3 -m streamlit run app.py
```

Why `python3 -m streamlit` instead of `streamlit run`:
- It avoids PATH issues where the `streamlit` command is not found even though the package is installed.

## Ingest A PDF

You have two options.

Option A: Use the UI upload button in Streamlit (recommended).

Option B: Use CLI ingestion:

```bash
python3 -m ingestion.run_ingest --file /path/to/your/report.pdf
```

Embedded chunks are stored in `./chroma_db`.

## API Endpoints
- `GET /` health/info
- `POST /ingest` upload and index a PDF
- `POST /chat` NDJSON streaming answer endpoint
- `GET /metrics` returns current evaluation/system metadata

## Run Evaluation

```bash
python3 -m evaluation.eval
```

This runs the golden-set evaluation and prints:
- faithfulness
- answer_relevancy
- context_precision
- context_recall
- overall score

## Quick Sanity Checks

```bash
python3 test_setup.py
python3 test_retrieval.py
python3 test_generation.py
```

## Common Issues

1. Error: Form data requires python-multipart

Install dependencies again in the same Python interpreter used to run Uvicorn:

```bash
python3 -m pip install -r requirements.txt
```

2. Error: streamlit command not found (exit code 127)

Use module invocation:

```bash
python3 -m streamlit run app.py
```

3. API offline warning in Streamlit
- Ensure Uvicorn is running on port 8000.
- Default API URL in `app.py` is `http://localhost:8000`.

4. Missing keys / auth errors
- Confirm `.env` exists at repo root.
- Confirm both `OPENAI_API_KEY` and `COHERE_API_KEY` are set.

## Notes For Developers
- Retrieval pipeline in `retrieval/retriever.py`:
  - question rewrite
  - Chroma similarity search
  - Cohere rerank
- Generation pipeline in `generation/chain.py` uses `gpt-4o-mini`.
- Streaming API path in `api/main.py` currently uses `gpt-4o`.

If you want one model everywhere, update those files to use the same model value.
