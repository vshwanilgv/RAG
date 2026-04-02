# ingestion/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into retrievable chunks.

    Why these settings:
    - chunk_size=600: large enough to hold a financial paragraph + numbers
    - chunk_overlap=80: ensures context isn't lost at boundaries
      e.g. "Revenue grew 12%..." at end of chunk A bleeds into chunk B
    - separators: respect document structure first (sections > paragraphs > sentences)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # adding chunk-level metadata for filtering and citations
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}_chunk_{i}"
        chunk.metadata["char_count"] = len(chunk.page_content)
        # Tag chunks that likely contain tables or financial figures
        chunk.metadata["has_numbers"] = any(
            c.isdigit() for c in chunk.page_content
        )

    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks