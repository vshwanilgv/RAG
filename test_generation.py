from generation.chain import answer

questions = [
    "What was Amazon's total revenue in 2024 and how does it compare to 2023?",
    "How did AWS operating income change in 2024?",
    "What risks does Amazon face in international markets?",
]

for q in questions:
    print(f"\n{'='*60}")
    result = answer(q)
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used:")
    for s in result['sources']:
        print(f"  Page {s['page']} | score: {s['rerank_score']} | {s['preview'][:80]}...")
    print(f"\nConfidence: {result['confidence']}")