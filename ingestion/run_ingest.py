"""
Run this once per new document you want to add to FinSight.
Usage: python -m ingestion.run_ingest --file reports/apple_10k_2024.pdf
"""
import argparse
from ingestion.loader import load_financial_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store

def ingest(file_path: str):
    print(f"\n--- FinSight Ingestion Pipeline ---")
    print(f"File: {file_path}\n")

    # Step 1: Load
    pages = load_financial_pdf(file_path)

    # Step 2: Chunk
    chunks = chunk_documents(pages)

    # Step 3: Embed + Store
    vectorstore = embed_and_store(chunks)

    print(f"\nDone. {len(chunks)} chunks indexed and ready to query.")
    return vectorstore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    ingest(args.file)