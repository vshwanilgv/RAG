from ingestion.embedder import load_vectorstore
from retrieval.retriever import retrieve

db = load_vectorstore()

questions = [
    "How did AWS perform in 2024?",
    "What was Amazon's operating income?",
    "What risks does Amazon face in international markets?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Question: {q}")
    docs = retrieve(q, db, top_n=3)
    print(f"\nTop chunks returned:")
    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] Page {doc.metadata['page']} "
              f"| rerank score: {doc.metadata['rerank_score']}")
        print(f"  {doc.page_content[:250]}...")