"""
rag_project/memory/redis_session.py

Redis-based conversation memory for RAG chatbot.
Persists multi-turn conversation history across API requests.

Usage:
    from rag_project.memory.redis_session import RedisSessionMemory
    memory = RedisSessionMemory(session_id="user_123")
    memory.add_message("user", "What is RAG?")
    memory.add_message("assistant", "RAG stands for Retrieval-Augmented Generation...")
    history = memory.get_history()
"""
from __future__ import annotations
import json
from datetime import datetime
from typing import Optional


class RedisSessionMemory:
    """
    Manages per-session conversation history using Redis.
    Falls back to in-memory dict if Redis is unavailable.

    Args:
        session_id: Unique identifier for this conversation session
        max_messages: Maximum messages to retain per session (default: 20)
        ttl_seconds: Redis key TTL in seconds (default: 3600 = 1 hour)
        redis_url: Redis connection URL (default: redis://localhost:6379)
    """

    def __init__(
        self,
        session_id: str,
        max_messages: int = 20,
        ttl_seconds: int = 3600,
        redis_url: str = "redis://localhost:6379",
    ):
        self.session_id = session_id
        self.max_messages = max_messages
        self.ttl = ttl_seconds
        self._key = f"rag:session:{session_id}"
        self._memory_fallback: dict[str, list] = {}

        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._use_redis = True
        except Exception:
            self._use_redis = False

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        msg = {"role": role, "content": content,
               "timestamp": datetime.now().isoformat()}

        if self._use_redis:
            try:
                history = self._get_raw()
                history.append(msg)
                if len(history) > self.max_messages:
                    history = history[-self.max_messages:]
                self._redis.setex(self._key, self.ttl, json.dumps(history))
                return
            except Exception:
                pass

        # Fallback
        self._memory_fallback.setdefault(self._key, []).append(msg)
        if len(self._memory_fallback[self._key]) > self.max_messages:
            self._memory_fallback[self._key] =                 self._memory_fallback[self._key][-self.max_messages:]

    def get_history(self) -> list[dict]:
        """Return full conversation history for this session."""
        return self._get_raw()

    def get_context_string(self) -> str:
        """Format history as a context string for the LLM prompt."""
        history = self._get_raw()
        lines = []
        for msg in history:
            prefix = "Human" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all history for this session."""
        if self._use_redis:
            try:
                self._redis.delete(self._key)
                return
            except Exception:
                pass
        self._memory_fallback.pop(self._key, None)

    def _get_raw(self) -> list[dict]:
        if self._use_redis:
            try:
                raw = self._redis.get(self._key)
                return json.loads(raw) if raw else []
            except Exception:
                pass
        return list(self._memory_fallback.get(self._key, []))

    def __repr__(self) -> str:
        msgs = len(self._get_raw())
        backend = "Redis" if self._use_redis else "in-memory"
        return f"RedisSessionMemory(session={self.session_id!r}, messages={msgs}, backend={backend})"
