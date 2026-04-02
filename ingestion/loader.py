# ingestion/loader.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List
import re

def load_financial_pdf(file_path: str) -> List[Document]:
    """
    Load a financial PDF and clean it up.
    Financial PDFs often have noisy headers/footers — we strip them.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    cleaned = []
    for page in pages:
        text = page.page_content

        text = re.sub(r'\n{3,}', '\n\n', text)         # collapse blank lines
        text = re.sub(r'(?m)^\s*\d+\s*$', '', text)    # remove lone page numbers
        text = text.strip()

        if len(text) < 50:      # skip near-empty pages
            continue

        page.metadata.update({
            "source": Path(file_path).name,
            "page": page.metadata.get("page", 0) + 1,
            "doc_type": "financial_report",
        })
        page.page_content = text
        cleaned.append(page)

    print(f"Loaded {len(cleaned)} pages from {Path(file_path).name}")
    return cleaned