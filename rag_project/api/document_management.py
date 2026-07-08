"""
rag_project/api/document_management.py

Document management endpoints — list, upload, delete, search documents.

Routes:
    GET  /documents          — list all indexed documents
    POST /documents/upload   — upload + index new document
    DELETE /documents/{id}   — remove document from index
    GET  /documents/search   — search documents by query
"""
from __future__ import annotations
import os
import hashlib
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory document store (replace with ChromaDB/FAISS in production)
_DOCUMENTS: dict[str, dict] = {}


class DocumentMeta(BaseModel):
    id: str
    filename: str
    size_bytes: int
    chunks: int
    indexed_at: str
    hash: str


class DocumentList(BaseModel):
    total: int
    documents: list[DocumentMeta]


@router.get("", response_model=DocumentList, summary="List all indexed documents")
async def list_documents():
    docs = list(_DOCUMENTS.values())
    return DocumentList(total=len(docs), documents=docs)


@router.post("/upload", response_model=DocumentMeta, summary="Upload and index a document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, MD) and add it to the RAG index.
    Returns the document metadata including chunk count.
    """
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(413, "File too large — max 10MB")

    allowed = {".pdf", ".txt", ".md", ".docx"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed}")

    doc_hash = hashlib.sha256(content).hexdigest()[:16]
    doc_id = f"doc_{doc_hash}"

    if doc_id in _DOCUMENTS:
        raise HTTPException(409, f"Document already indexed (id={doc_id})")

    # Simulate chunking (replace with real text splitter)
    text = content.decode("utf-8", errors="replace") if ext in {".txt", ".md"} else "[PDF content]"
    chunks = max(1, len(text) // 500)

    meta = DocumentMeta(
        id=doc_id,
        filename=file.filename or "unknown",
        size_bytes=len(content),
        chunks=chunks,
        indexed_at=datetime.now().isoformat(),
        hash=doc_hash,
    )
    _DOCUMENTS[doc_id] = meta.model_dump()
    return meta


@router.delete("/{doc_id}", summary="Remove document from index")
async def delete_document(doc_id: str):
    """Remove a document from the RAG index by its ID."""
    if doc_id not in _DOCUMENTS:
        raise HTTPException(404, f"Document {doc_id!r} not found")
    del _DOCUMENTS[doc_id]
    return {"message": f"Document {doc_id} deleted", "remaining": len(_DOCUMENTS)}


@router.get("/search", summary="Search indexed documents by query")
async def search_documents(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(5, ge=1, le=20),
):
    """Search document metadata by filename or content preview."""
    results = [
        doc for doc in _DOCUMENTS.values()
        if q.lower() in doc["filename"].lower()
    ][:limit]
    return {"query": q, "results": results, "total": len(results)}
