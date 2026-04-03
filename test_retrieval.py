from ingestion.embedder import load_vectorstore

db = load_vectorstore()

questions = [
    "What was Amazon's total net sales in 2024?",
    "How did AWS perform in 2024?",
    "What are the main risk factors Amazon faces?",
]

for q in questions:
    print(f"\nQuery: {q}")
    results = db.similarity_search(q, k=2)
    for r in results:
        print(f"  Page {r.metadata['page']} | {r.metadata['source']}")
        print(f"  {r.page_content[:180]}...")
    print("-" * 60)