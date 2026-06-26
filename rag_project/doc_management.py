"""
Issue #11: Document management API — list, delete, re-index.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
DOC_REGISTRY = os.getenv("DOC_REGISTRY", "./data/doc_registry.json")


def _load_registry() -> dict[str, Any]:
    try:
        with open(DOC_REGISTRY) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_registry(data: dict) -> None:
    Path(DOC_REGISTRY).parent.mkdir(parents=True, exist_ok=True)
    with open(DOC_REGISTRY, "w") as f:
        json.dump(data, f, indent=2)


def register_document(doc_id: str, filename: str, chunk_count: int,
                       char_count: int) -> dict[str, Any]:
    registry = _load_registry()
    import time
    entry = {
        "doc_id": doc_id, "filename": filename,
        "chunk_count": chunk_count, "char_count": char_count,
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    registry[doc_id] = entry
    _save_registry(registry)
    return entry


def list_documents() -> list[dict[str, Any]]:
    return list(_load_registry().values())


def delete_document(doc_id: str) -> bool:
    registry = _load_registry()
    if doc_id not in registry:
        return False
    del registry[doc_id]
    _save_registry(registry)
    log.info("Deleted document %s from registry", doc_id)
    # Note: removing from FAISS index requires full re-index
    return True


def get_document(doc_id: str) -> dict[str, Any] | None:
    return _load_registry().get(doc_id)
