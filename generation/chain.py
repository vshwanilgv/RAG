import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List, Tuple
from retrieval.retriever import retrieve
from ingestion.embedder import load_vectorstore

load_dotenv()

FINANCIAL_QA_PROMPT = PromptTemplate.from_template("""
You are FinSight, an expert financial analyst assistant.
You answer questions strictly based on the provided context from financial documents.

RULES:
# - Only use information present in the context below
# - Always cite the page number(s) you used, like: [Page 37]
# - If the context does not contain enough information to answer, say:
#   "The provided document sections don't contain enough information to answer this confidently."
# - For numerical data, quote the exact figures from the document
# - Be concise but complete
- Every sentence in your answer must be directly supported by the context
- Do NOT infer, extrapolate, or add background knowledge
- Quote exact figures as they appear in the document
- Cite page numbers inline like [Page 37] after each factual claim
- If a specific fact is not in the context, do not include it
- Keep your answer to 3-5 sentences maximum                                                  

                                                   

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")

def format_context(docs: List[Document]) -> str:
    sections = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("rerank_score", 0)
        sections.append(
            f"[Page {page} | relevance: {score}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(sections)

def answer(question: str, vectorstore=None) -> dict:
    if vectorstore is None:
        vectorstore = load_vectorstore()

    docs, confidence = retrieve(question, vectorstore, top_n=5)

    context = format_context(docs)

    if confidence == "low":
        print("  Warning: low retrieval confidence — answer may be incomplete")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,              # temperature=0 for factual financial Q&A
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    chain = FINANCIAL_QA_PROMPT | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "question": question,
        "answer": response.content,
        "sources": [
            {
                "page": d.metadata.get("page"),
                "source": d.metadata.get("source"),
                "rerank_score": d.metadata.get("rerank_score"),
                "preview": d.page_content[:120],
                "full_text": d.page_content,
            }
            for d in docs
        ],
        "confidence": confidence,
        "context_used": context
    }