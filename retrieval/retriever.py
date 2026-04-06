import os
import cohere
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from ingestion.embedder import load_vectorstore, get_embeddings

load_dotenv()

REWRITE_PROMPT = PromptTemplate.from_template("""
You are a financial document search assistant.
Rewrite the user's question as a short, keyword-rich search query 
optimized for retrieving passages from financial reports.
Remove conversational filler. Keep numbers, company names, and financial terms.

User question: {question}
Search query (max 15 words):""")

def rewrite_query(question: str) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",       
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    chain = REWRITE_PROMPT | llm
    result = chain.invoke({"question": question})
    rewritten = result.content.strip()
    print(f"  Original : {question}")
    print(f"  Rewritten: {rewritten}")
    return rewritten

def vector_retrieve(query: str, vectorstore: Chroma, k: int = 15):
    results = vectorstore.similarity_search_with_score(query, k=k)
    # Filter out very low similarity scores (below 0.4 = likely irrelevant)
    filtered = [(doc, score) for doc, score in results if score < 1.2]
    return [doc for doc, _ in filtered]

def rerank(question: str, documents, top_n: int = 5):
    co = cohere.Client(os.getenv("COHERE_API_KEY"))

    passages = [doc.page_content for doc in documents]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=question,
        documents=passages,
        top_n=top_n,
    )

    # Return documents in reranked order
    reranked_docs = []
    for hit in response.results:
        doc = documents[hit.index]
        doc.metadata["rerank_score"] = round(hit.relevance_score, 4)
        reranked_docs.append(doc)

    return reranked_docs

def retrieve(question: str, vectorstore: Chroma, top_n: int = 5):
    print("\n[Retrieval Pipeline]")
    search_query = rewrite_query(question)
    candidates = vector_retrieve(search_query, vectorstore, k=15)
    print(f"  Retrieved {len(candidates)} candidate chunks")
    reranked = rerank(question, candidates, top_n=top_n)

    filtered = [d for d in reranked if d.metadata["rerank_score"] >= 0.5]
    final_docs = filtered if len(filtered) >= 2 else reranked[:2]

    print(f"  Reranked to top {len(final_docs)} chunks")

    top_score = final_docs[0].metadata["rerank_score"] if final_docs else 0
    confidence = "high" if top_score > 0.7 else "low"
    print(f"  Confidence: {confidence} (top score: {top_score})")

    return final_docs, confidence