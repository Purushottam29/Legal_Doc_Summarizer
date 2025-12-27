"""
app/routers/query.py
Query endpoint for retrieving the most relevant legal document chunks
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

from app.services.vector_store import VectorStore

router = APIRouter()

# shared vector store (same defaults as ingestion)
vs = VectorStore(
    collection_name="legal_docs",
    persist_directory="./chroma_db",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post("/query")
def query_document(req: QueryRequest) -> Dict[str, Any]:
    if not req.question or len(req.question.strip()) < 3:
        return {"status": "error", "message": "Invalid question"}

    results = vs.query(query_text=req.question, top_k=req.top_k, include=["documents", "metadatas", "distances"])

    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    dists = results.get("distances") or []

    # ensure nested-list normalization already done by vector_store; handle empty
    if not docs:
        return {"status": "success", "query": req.question, "results": [], "message": "No relevant chunks found"}

    formatted = []
    # length of docs should equal length of ids/metas/dists (best effort)
    n = len(docs)
    for i in range(n):
        formatted.append({
            "chunk_id": ids[i] if i < len(ids) else None,
            "chunk_text": docs[i],
            "metadata": metas[i] if i < len(metas) else {},
            "distance_score": dists[i] if i < len(dists) else None
        })

    return {"status": "success", "query": req.question, "results": formatted}

