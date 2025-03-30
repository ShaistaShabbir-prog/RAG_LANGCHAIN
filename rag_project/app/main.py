from fastapi import FastAPI
from app.routes import ingest, query
from app.services.embedding_service import embedding_service

app = FastAPI(title="RAG API")

app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")


@app.get("/")
def home():
    return {"message": "API is running!"}


@app.on_event("startup")
async def startup():
    try:
        embedding_service.ingest_documents("data/documents.txt")
    except Exception as e:
        print(f"Error during startup: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
