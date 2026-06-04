import os
import django
import json

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from rag.views import run_rag_pipeline


SOFTWARE_ENGINEER_ID = 12
ML_ENGINEER_ID = 7
DATA_ANALYST_ID = 9
DEVOPS_ID = 8
PRODUCT_MANAGER_ID = 10

TEST_CASES = [
    {
        "question": "Does this candidate know Python?",
        "ground_truth": "Yes. The resume lists Python.",
        "document_id": SOFTWARE_ENGINEER_ID,
    },
    {
        "question": "Does this candidate have backend experience?",
        "ground_truth": "Yes. The candidate has backend experience with Django, PostgreSQL, and REST APIs.",
        "document_id": SOFTWARE_ENGINEER_ID,
    },
    {
        "question": "Does this candidate have machine learning experience?",
        "ground_truth": "Yes. The candidate has machine learning and NLP experience.",
        "document_id": ML_ENGINEER_ID,
    },
    {
        "question": "Does this candidate know SQL?",
        "ground_truth": "Yes. The candidate lists SQL or database experience.",
        "document_id": DATA_ANALYST_ID,
    },
    {
        "question": "Does this candidate have cloud experience?",
        "ground_truth": "Yes. The candidate has AWS or cloud infrastructure experience.",
        "document_id": DEVOPS_ID,
    },
    {
        "question": "Does this candidate have RAG experience?",
        "ground_truth": "I cannot determine this from the resume.",
        "document_id": PRODUCT_MANAGER_ID,
    },
]

evaluator_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

evaluator_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

def main():
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for test in TEST_CASES:
        result = run_rag_pipeline(
            query=test["question"],
            document_id=test["document_id"]
        )

        questions.append(test["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(test["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    
    df = result.to_pandas()

    print("\n=== OVERALL METRICS ===")

    print(
        f"Faithfulness: {df['faithfulness'].mean():.3f}"
    )

    print(
        f"Answer Relevancy: {df['answer_relevancy'].mean():.3f}"
    )

    print(
        f"Context Precision: {df['context_precision'].mean():.3f}"
    )

    print(
        f"Context Recall: {df['context_recall'].mean():.3f}"
    )

    results_dict = result.to_pandas().to_dict(orient="records")

    with open("ragas_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(results_dict)
    


if __name__ == "__main__":
    main()