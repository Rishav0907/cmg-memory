"""
CMG Memory — Python Client SDK
================================
Use this when you want to call the hosted API instead of
running the library locally.

Install:
    pip install cmg-memory[api]

Usage:
    from cmg.client import CMGClient

    client = CMGClient(api_key="cmg_yourkey_here")

    # Chat with memory
    response = client.chat("I always use PyTorch for deep learning")
    print(response.response)

    # List memories
    memories = client.list_memories()
    for m in memories:
        print(m.score, m.content)

    # Manual memory
    client.remember("User is a PhD researcher at IISc", memory_type="fact")

    # Search
    results = client.search("what framework do I use?")

    # End session (runs consolidation)
    client.end_session()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Memory:
    id: str
    content: str
    memory_type: str
    layer: str
    score: float
    access_count: int
    created_at: float
    metadata: dict

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        return cls(
            id          = d["id"],
            content     = d["content"],
            memory_type = d["memory_type"],
            layer       = d["layer"],
            score       = d["score"],
            access_count= d["access_count"],
            created_at  = d["created_at"],
            metadata    = d.get("metadata", {}),
        )

    def __repr__(self):
        return f"Memory(score={self.score:.3f}, type={self.memory_type}, content={self.content[:60]})"


@dataclass
class ChatResult:
    response: str
    session_id: str
    memories_used: int
    memories_stored: int
    latency_ms: int


@dataclass
class Stats:
    total_memories: int
    by_layer: dict
    by_type: dict
    provider: str
    namespace: str


class CMGClient:
    """
    HTTP client for the CMG Memory hosted API.

    All methods are synchronous. For async usage, use CMGAsyncClient.

    Args:
        api_key:  Your CMG API key (starts with cmg_)
        base_url: API base URL. Default: https://api.cmg-memory.com
                  Use http://localhost:8000 for local development.
        timeout:  Request timeout in seconds (default: 60)
    """

    def __init__(
        self,
        api_key:  str,
        base_url: str = "https://api.cmg-memory.com",
        timeout:  float = 60.0,
    ):
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "Install the API client: pip install cmg-memory[api]\n"
                "Or: pip install httpx"
            )
        self._base    = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"}
        self._client  = httpx.Client(
            base_url = self._base,
            headers  = self._headers,
            timeout  = timeout,
        )
        self._session_id: Optional[str] = None

    # ── Core methods ──────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        system_prompt:  Optional[str] = None,
        temperature:    float = 0.7,
        max_tokens:     int   = 1024,
    ) -> ChatResult:
        """
        Send a message. Memory is retrieved before the LLM call and
        extracted + stored after. Returns the assistant response.
        """
        payload = {
            "message":       message,
            "session_id":    self._session_id,
            "temperature":   temperature,
            "max_tokens":    max_tokens,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        r = self._post("/v1/chat", payload)
        result = ChatResult(
            response        = r["response"],
            session_id      = r["session_id"],
            memories_used   = r["memories_used"],
            memories_stored = r["memories_stored"],
            latency_ms      = r["latency_ms"],
        )
        self._session_id = result.session_id
        return result

    def remember(
        self,
        content:     str,
        memory_type: str = "fact",
        layer:       str = "episodic",
    ) -> Memory:
        """Manually store a memory."""
        r = self._post("/v1/memories", {
            "content":     content,
            "memory_type": memory_type,
            "layer":       layer,
        })
        return Memory.from_dict(r)

    def list_memories(
        self,
        layer:       Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[Memory]:
        """List all stored memories."""
        params = {}
        if layer:       params["layer"]       = layer
        if memory_type: params["memory_type"] = memory_type
        r = self._get("/v1/memories", params=params)
        return [Memory.from_dict(m) for m in r]

    def search(self, query: str, top_k: int = 5) -> list[Memory]:
        """Semantic search across memories."""
        r = self._post("/v1/memories/search", {"query": query, "top_k": top_k})
        return [Memory.from_dict(m) for m in r]

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        r = self._delete(f"/v1/memories/{memory_id}")
        return r.get("deleted", False)

    def extract(self, text: str, store: bool = True) -> list[Memory]:
        """
        Extract memories from arbitrary text (notes, documents, conversation history).
        If store=True, extracted memories are saved immediately.
        """
        r = self._post("/v1/extract", {
            "text":       text,
            "store":      store,
            "session_id": self._session_id,
        })
        return [Memory.from_dict(m) for m in r.get("memories", [])]

    def end_session(self, run_consolidation: bool = True) -> dict:
        """End session and run consolidation (decay, merge, promote)."""
        if not self._session_id:
            return {}
        r = self._post("/v1/sessions/end", {
            "session_id":        self._session_id,
            "run_consolidation": run_consolidation,
        })
        self._session_id = None
        return r

    def stats(self) -> Stats:
        """Return memory statistics."""
        r = self._get("/v1/stats")
        return Stats(
            total_memories = r["total_memories"],
            by_layer       = r["by_layer"],
            by_type        = r["by_type"],
            provider       = r["provider"],
            namespace      = r["namespace"],
        )

    def health(self) -> dict:
        """Check API health."""
        return self._get("/health")

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "CMGClient":
        return self

    def __exit__(self, *_) -> None:
        self.end_session()
        self._client.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        resp = self._client.post(path, json=payload)
        self._raise_for_status(resp)
        return resp.json()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = self._client.get(path, params=params)
        self._raise_for_status(resp)
        return resp.json()

    def _delete(self, path: str) -> dict:
        resp = self._client.delete(path)
        self._raise_for_status(resp)
        return resp.json()

    def _raise_for_status(self, resp) -> None:
        if resp.status_code == 401:
            raise PermissionError(
                "Invalid API key. Get yours at https://cmg-memory.com/dashboard"
            )
        if resp.status_code == 404:
            raise KeyError(f"Resource not found: {resp.url}")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"API error {resp.status_code}: {detail}")


# ── Async client ──────────────────────────────────────────────────────────────

class CMGAsyncClient:
    """
    Async version of CMGClient. Use with `await` in async code.

    Usage:
        async with CMGAsyncClient(api_key="cmg_...") as client:
            result = await client.chat("I use PyTorch")
            print(result.response)
    """

    def __init__(
        self,
        api_key:  str,
        base_url: str = "https://api.cmg-memory.com",
        timeout:  float = 60.0,
    ):
        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url = base_url.rstrip("/"),
                headers  = {"Authorization": f"Bearer {api_key}",
                            "Content-Type":  "application/json"},
                timeout  = timeout,
            )
        except ImportError:
            raise ImportError("pip install cmg-memory[api]")
        self._session_id: Optional[str] = None

    async def chat(self, message: str, **kwargs) -> ChatResult:
        payload = {"message": message, "session_id": self._session_id, **kwargs}
        r = await self._post("/v1/chat", payload)
        result = ChatResult(**r)
        self._session_id = result.session_id
        return result

    async def remember(self, content: str, memory_type: str = "fact", layer: str = "episodic") -> Memory:
        return Memory.from_dict(await self._post("/v1/memories",
            {"content": content, "memory_type": memory_type, "layer": layer}))

    async def list_memories(self, layer: Optional[str] = None) -> list[Memory]:
        params = {"layer": layer} if layer else {}
        data   = await self._get("/v1/memories", params=params)
        return [Memory.from_dict(m) for m in data]

    async def search(self, query: str, top_k: int = 5) -> list[Memory]:
        data = await self._post("/v1/memories/search", {"query": query, "top_k": top_k})
        return [Memory.from_dict(m) for m in data]

    async def forget(self, memory_id: str) -> bool:
        r = await self._delete(f"/v1/memories/{memory_id}")
        return r.get("deleted", False)

    async def stats(self) -> Stats:
        return Stats(**(await self._get("/v1/stats")))

    async def end_session(self, run_consolidation: bool = True) -> dict:
        if not self._session_id:
            return {}
        r = await self._post("/v1/sessions/end",
            {"session_id": self._session_id, "run_consolidation": run_consolidation})
        self._session_id = None
        return r

    async def __aenter__(self) -> "CMGAsyncClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.end_session()
        await self._client.aclose()

    async def _post(self, path: str, payload: dict) -> dict:
        r = await self._client.post(path, json=payload)
        self._check(r)
        return r.json()

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        r = await self._client.get(path, params=params)
        self._check(r)
        return r.json()

    async def _delete(self, path: str) -> dict:
        r = await self._client.delete(path)
        self._check(r)
        return r.json()

    def _check(self, r) -> None:
        if r.status_code >= 400:
            try:   detail = r.json().get("detail", r.text)
            except: detail = r.text
            raise RuntimeError(f"API error {r.status_code}: {detail}")