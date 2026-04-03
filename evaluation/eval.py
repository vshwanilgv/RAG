import asyncio
import nest_asyncio
nest_asyncio.apply()
import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from generation.chain import answer
from ingestion.embedder import load_vectorstore

load_dotenv()


GOLDEN_SET = [
    {
        "question": "What was Amazon's total revenue in 2024?",
        "ground_truth": "Amazon's total net sales were $638 billion in 2024, representing 11% growth year-over-year from $575 billion in 2023."
    },
    {
        "question": "How much did AWS operating income grow in 2024?",
        "ground_truth": "AWS operating income grew from $24,631 million in 2023 to $39,834 million in 2024."
    },
    {
        "question": "What was Amazon's North America revenue in 2024?",
        "ground_truth": "North America revenue was $387 billion in 2024, up 10% from $353 billion in 2023."
    },
    {
        "question": "What was Amazon's consolidated operating income in 2024?",
        "ground_truth": "Amazon's consolidated operating income was $68,593 million ($68.6 billion) in 2024, up from $36,852 million in 2023."
    },
    {
        "question": "By how much did AWS sales increase in 2024?",
        "ground_truth": "AWS sales increased 19% in 2024 compared to the prior year."
    },
]


def run_evaluation():
    print("\n--- FinSight RAGAs Evaluation ---\n")
    vectorstore = load_vectorstore()

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in GOLDEN_SET:
        print(f"Evaluating: {item['question']}")
        result = answer(item["question"], vectorstore)

        questions.append(item["question"])
        answers.append(result["answer"])
        ground_truths.append(item["ground_truth"])

        contexts.append([s["full_text"] for s in result["sources"]])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run evaluation — uses GPT-4o as the judge
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n--- Results ---")
    print(f"Faithfulness:      {scores['faithfulness']:.3f}  (is answer grounded in context?)")
    print(f"Answer relevancy:  {scores['answer_relevancy']:.3f}  (does answer address the question?)")
    print(f"Context precision: {scores['context_precision']:.3f}  (are retrieved chunks relevant?)")
    print(f"Context recall:    {scores['context_recall']:.3f}  (did we retrieve all needed info?)")
    print(f"\nOverall RAG score: {sum([scores['faithfulness'], scores['answer_relevancy'], scores['context_precision'], scores['context_recall']]) / 4:.3f}")

    return scores


if __name__ == "__main__":
    run_evaluation()