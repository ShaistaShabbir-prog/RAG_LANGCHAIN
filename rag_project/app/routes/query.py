from fastapi import APIRouter
from app.models import QueryRequest, QueryResponse
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_model(request: QueryRequest):
    try:
        # Retrieve the top k most relevant documents
        retrieved_docs = embedding_service.retrieve(request.query)
        if not retrieved_docs:
            return QueryResponse(answer="No relevant documents found.")

        # Concatenate retrieved document contents into context
        context = " ".join([doc.page_content for doc in retrieved_docs])

        # Get the answer using the LLM service
        answer = llm_service.generate_answer(request.query, context)
        if answer is None:
            return QueryResponse(answer="Error: Failed to generate an answer.")

        print(f"Generated answer: {answer}")
        return QueryResponse(answer=answer)
    except Exception as e:
        return QueryResponse(answer=f"Error: {str(e) or 'Unknown error'}")
