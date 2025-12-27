import re
import uuid
from typing import Dict, Any, List

from app.util.chunker import smart_chunker
from app.services.vector_store import VectorStore
from transformers import AutoTokenizer

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

vs = VectorStore(
    collection_name="legal_docs",
    persist_directory="./chroma_db",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x0c", " ").strip()
    return text


def ingest_document(text: str, doc_id: str = None) -> Dict[str, Any]:
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    cleaned = clean_text(text)
    if not cleaned or len(cleaned) < 20:
        return {"status": "error", "message": "Document empty or too short"}

    chunks = smart_chunker(
        cleaned,
        tokenizer=tokenizer,
        max_tokens=256,
        overlap=20,
    )

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
             "doc_id": doc_id,
             "chunk_index": i,
             "clause_id": f"{doc_id}_clause_{i}"
        }
        for i in range(len(chunks))
    ]

    vs.add_documents(ids=ids, documents=chunks, metadatas=metadatas, batch_size=50)

    return {"status": "success", "doc_id": doc_id, "chunks_created": len(chunks)}

