from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from app.services.vector_store import VectorStore
from app.services.summarizer import generate_summary

router = APIRouter()

# Shared vector DB
vs = VectorStore(
    collection_name="legal_docs",
    persist_directory="./chroma_db",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 3

#BUILD RAG PROMPT
def build_rag_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Builds a safe, citation-friendly, hallucination-resistant RAG prompt.
    """
    context_text = "\n\n".join([
        f"[CLAUSE {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])
    prompt = f"""
You are a legal assistant specialized in analyzing contracts.

Use ONLY the information provided in the clauses below.

If the answer is not contained in the clauses, respond exactly with:
"The document does not contain this information."

---------------------
CLAUSES:
{context_text}
---------------------

QUESTION:
{question}

INSTRUCTIONS:
- Answer in full sentences.
- Be legally precise.
- Do NOT make up facts.
- If relevant, cite clause numbers like: "According to Clause 2..."
- If unsure or context does not contain the answer, say:
  "The document does not contain this information."

FINAL ANSWER:
"""

    return prompt.strip()

#RAG ANSWER GENERATION ENDPOINT
@router.post("/ask")
def ask_question(req: AskRequest) -> Dict[str, Any]:
    """Answer user questions using retrieved legal clauses + Gemma RAG."""

    if not req.question or len(req.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Invalid question")

    # Step 1 — Retrieve relevant chunks from vector store
    retrieved = vs.query(
        query_text=req.question,
        top_k=req.top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = retrieved.get("documents") or []

    # If nothing relevant found → respond safely
    if not chunks:
        return {
            "status": "success",
            "answer": "The document does not contain this information.",
            "used_clauses": [],
            "raw_chunks": []
        }

    # Step 2 — Build the RAG prompt
    rag_prompt = build_rag_prompt(req.question, chunks)

    # Step 3 — Generate answer using Gemma
    try:
        answer = generate_summary(rag_prompt)
    except Exception:
        answer = "The answer could not be generated due to an internal error."

    # Step 4 — Return everything cleanly
    raw_chunk_info = []
    metas = retrieved.get("metadatas") or []
    dists = retrieved.get("distances") or []

    for i in range(len(chunks)):
        raw_chunk_info.append({
            "clause_number": i + 1,
            "chunk_text": chunks[i],
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
        })

    return {
        "status": "success",
        "question": req.question,
        "answer": answer,
        "used_clauses": chunks,
        "raw_chunks": raw_chunk_info,
    }

