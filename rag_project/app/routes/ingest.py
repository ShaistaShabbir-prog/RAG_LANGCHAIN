from fastapi import APIRouter
from app.services.embedding_service import embedding_service

router = APIRouter()


@router.post("/ingest")
def ingest_data():
    try:
        embedding_service.ingest_documents("data/documents.txt")
        return {"message": "Documents ingested successfully!"}
    except Exception as e:
        return {"error": f"Failed to ingest documents: {str(e)}"}
