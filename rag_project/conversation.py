"""
Issue #10: Multi-turn conversation memory for /chat endpoint.
"""
from __future__ import annotations
import time, logging
from collections import defaultdict
log = logging.getLogger(__name__)

_sessions: dict[str, dict] = defaultdict(lambda: {"history":[], "ts":0.0})
SESSION_TTL = 1800  # 30 minutes


def _prune():
    now = time.time()
    for sid in [k for k,v in _sessions.items() if now-v["ts"]>SESSION_TTL]:
        del _sessions[sid]


def chat_with_memory(question: str, session_id: str, chain) -> dict[str, Any]:
    """Run RAG chain with conversation history injected into context."""
    _prune()
    s = _sessions[session_id]
    s["ts"] = time.time()

    # Build context-aware question
    history_text = ""
    for turn in s["history"][-4:]:
        history_text += f"User: {turn['q']}\nAssistant: {turn['a']}\n"

    augmented = f"{history_text}User: {question}" if history_text else question

    result = chain.invoke({"question": augmented})
    answer = result.get("answer", result) if isinstance(result, dict) else str(result)

    s["history"].append({"q": question, "a": answer})
    return {"answer": answer, "session_id": session_id, "turn": len(s["history"])}


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def get_history(session_id: str) -> list[dict]:
    return _sessions[session_id]["history"]
