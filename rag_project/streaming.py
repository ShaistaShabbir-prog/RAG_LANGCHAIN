"""
Issue #2: Streaming response support for /query endpoint.
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio, json

router = APIRouter()

@router.get("/query/stream")
async def query_stream(q: str):
    """
    Stream RAG answer token by token using Server-Sent Events.
    Requires: langchain with streaming support.
    """
    async def generate():
        try:
            from rag_project.chain import get_chain
            chain = get_chain()
            async for chunk in chain.astream({"question": q}):
                text = chunk if isinstance(chunk, str) else chunk.get("answer", "")
                if text:
                    yield f"data: {json.dumps({'token': text})}\n\n"
                    await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
