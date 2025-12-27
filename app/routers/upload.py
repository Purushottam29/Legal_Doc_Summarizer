"""
app/routers/upload.py
Upload endpoint: receives file, extracts text, ingests to vector DB, returns summary + details
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
from app.services.extract_text import extract_text_from_file
from app.services.summarizer import generate_summary
from app.services.extract_details import extract_important_details
from app.services.ingest_document import ingest_document

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    # 1. extract text
    try:
        text = extract_text_from_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {e}")

    # 2. generate summary
    try:
        summary = generate_summary(text)
    except Exception as e:
        # don't fail ingest just because summarizer had an issue; warn instead
        summary = ""
        # optional: log here

    # 3. extract details
    try:
        details = extract_important_details(text)
    except Exception:
        details = {}

    # 4. ingest to vector DB (create doc_id from filename or uuid)
    doc_id = file.filename or None
    ingest_result = ingest_document(text=text, doc_id=doc_id)

    if ingest_result.get("status") != "success":
        raise HTTPException(status_code=500, detail=f"Ingest failed: {ingest_result}")

    return {
        "status": "success",
        "doc_id": ingest_result.get("doc_id"),
        "chunks_created": ingest_result.get("chunks_created"),
        "summary": summary,
        "details": details,
    }

