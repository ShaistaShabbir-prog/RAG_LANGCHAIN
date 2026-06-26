"""
Issue #10: Multi-turn conversation memory for RAG chat endpoint.
"""
from __future__ import annotations
import time
import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

SESSION_TTL = 1800  # 30 min
MAX_HISTORY = 10    # last 10 turns


class ConversationMemory:
    """In-memory session store for multi-turn RAG conversations."""

    def __init__(self):
        self._sessions: dict[str, dict] = defaultdict(
            lambda: {"history": [], "last_access": 0.0, "context": {}}
        )

    def get_history(self, session_id: str) -> list[dict]:
        s = self._sessions[session_id]
        s["last_access"] = time.time()
        return s["history"][-MAX_HISTORY:]

    def add_turn(self, session_id: str, question: str, answer: str,
                 sources: list[dict] | None = None) -> None:
        s = self._sessions[session_id]
        s["last_access"] = time.time()
        s["history"].append({
            "role": "user", "content": question, "ts": time.time()
        })
        s["history"].append({
            "role": "assistant", "content": answer,
            "sources": sources or [], "ts": time.time()
        })
        # Keep only last MAX_HISTORY * 2 messages
        s["history"] = s["history"][-(MAX_HISTORY * 2):]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        log.info("Cleared session %s", session_id)

    def prune_expired(self) -> int:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s["last_access"] > SESSION_TTL]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def stats(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "session_ttl_seconds": SESSION_TTL,
            "max_history_turns": MAX_HISTORY,
        }


# Global memory instance
memory = ConversationMemory()
