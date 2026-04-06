import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "finsight_docs"

def get_embeddings():
    """
    text-embedding-3-small: cheaper than large, still excellent for
    financial text. At $0.02/million tokens vs $0.13, it's a smart default.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def embed_and_store(chunks: List[Document]) -> Chroma:
    """
    Embed chunks and store in persistent Chroma vector DB.
    Returns the vectorstore for immediate querying.
    """
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    print(f"Stored {len(chunks)} chunks in Chroma at '{CHROMA_PATH}'")
    return vectorstore

def load_vectorstore() -> Chroma:
    """
    Load an existing Chroma DB without re-embedding.
    Call this on app startup instead of re-ingesting every time.
    """
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )