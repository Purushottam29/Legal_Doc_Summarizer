from fastapi import APIRouter
import json

from app.services.vector_store import VectorStore
from app.evaluation.metrics import precision_recall_f1

router = APIRouter()

vs = VectorStore(
    collection_name="legal_docs",
    persist_directory="./chroma_db",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

with open("app/evaluation/gold_data.json") as f:
    GOLD_DATA = json.load(f)


@router.get("/evaluate")
def evaluate_rag():
    results = []

    for item in GOLD_DATA:
        query = item["query"]
        relevant_ids = item["relevant_clause_ids"]

        retrieved = vs.query(
            query_text=query,
            top_k=5,
            include=["metadatas"]
        )

        retrieved_ids = [
            meta["clause_id"]
            for meta in retrieved.get("metadatas", [])
            if "clause_id" in meta
        ]

        metrics = precision_recall_f1(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids
        )

        results.append({
            "query": query,
            "metrics": metrics,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids
        })

    avg_precision = sum(r["metrics"]["precision"] for r in results) / len(results)
    avg_recall = sum(r["metrics"]["recall"] for r in results) / len(results)
    avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results)

    return {
        "average_metrics": {
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "f1": round(avg_f1, 4)
        },
        "per_query_results": results
    }

