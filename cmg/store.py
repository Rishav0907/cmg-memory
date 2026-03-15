"""
CMG Memory System - Vector Store
Lightweight vector store with cosine similarity search.
No external DB required — uses numpy for fast dot-product search.
Persists to JSON on disk. Swap this for ChromaDB / Pinecone / Weaviate
by implementing the VectorStore ABC.
"""

from __future__ import annotations
import json
import math
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from .types import MemoryChunk, MemoryLayer, MemoryType


class VectorStore(ABC):
    """Abstract base class — implement this to swap in any vector DB."""

    @abstractmethod
    def upsert(self, chunk: MemoryChunk) -> None: ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None: ...

    @abstractmethod
    def get(self, chunk_id: str) -> Optional[MemoryChunk]: ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        layer_filter: Optional[list[MemoryLayer]] = None,
    ) -> list[tuple[MemoryChunk, float]]: ...

    @abstractmethod
    def all_chunks(self) -> list[MemoryChunk]: ...

    @abstractmethod
    def count(self) -> int: ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(VectorStore):
    """
    Fast in-memory vector store backed by a JSON file.
    
    Suitable for personal use / development. For production with millions
    of chunks, replace with ChromaDB (see ChromaVectorStore below).
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._chunks: dict[str, MemoryChunk] = {}
        self._persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self._load()

    def upsert(self, chunk: MemoryChunk) -> None:
        self._chunks[chunk.id] = chunk
        if self._persist_path:
            self._save()

    def delete(self, chunk_id: str) -> None:
        self._chunks.pop(chunk_id, None)
        if self._persist_path:
            self._save()

    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        return self._chunks.get(chunk_id)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        layer_filter: Optional[list[MemoryLayer]] = None,
    ) -> list[tuple[MemoryChunk, float]]:
        results = []
        for chunk in self._chunks.values():
            if chunk.superseded_by:
                continue  # Skip replaced chunks
            if layer_filter and chunk.layer not in layer_filter:
                continue
            if chunk.embedding is None:
                continue
            sim = _cosine_similarity(query_embedding, chunk.embedding)
            results.append((chunk, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def all_chunks(self) -> list[MemoryChunk]:
        return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)

    def _save(self) -> None:
        data = {cid: c.to_dict() for cid, c in self._chunks.items()}
        # Also save embeddings separately (they're big)
        embeddings = {cid: c.embedding for cid, c in self._chunks.items() if c.embedding}
        payload = {"chunks": data, "embeddings": embeddings}
        tmp = self._persist_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self._persist_path)

    def _load(self) -> None:
        with open(self._persist_path) as f:
            payload = json.load(f)
        chunks_data = payload.get("chunks", {})
        embeddings = payload.get("embeddings", {})
        for cid, d in chunks_data.items():
            chunk = MemoryChunk.from_dict(d)
            chunk.embedding = embeddings.get(cid)
            self._chunks[cid] = chunk


# ---------------------------------------------------------------------------
# Optional: ChromaDB backend (install chromadb to use)
# ---------------------------------------------------------------------------

class ChromaVectorStore(VectorStore):
    """
    Production-grade vector store using ChromaDB.
    Install: pip install chromadb
    
    Usage:
        store = ChromaVectorStore(collection_name="cmg_user_123", persist_dir="./chroma_db")
    """

    def __init__(self, collection_name: str = "cmg_memory", persist_dir: str = "./chroma_db"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Keep a local cache for chunk metadata
        self._meta_cache: dict[str, MemoryChunk] = {}

    def upsert(self, chunk: MemoryChunk) -> None:
        if chunk.embedding is None:
            return
        meta = chunk.to_dict()
        # ChromaDB metadata values must be str/int/float/bool
        meta = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in meta.items()}
        self._col.upsert(
            ids=[chunk.id],
            embeddings=[chunk.embedding],
            metadatas=[meta],
            documents=[chunk.content],
        )
        self._meta_cache[chunk.id] = chunk

    def delete(self, chunk_id: str) -> None:
        self._col.delete(ids=[chunk_id])
        self._meta_cache.pop(chunk_id, None)

    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        if chunk_id in self._meta_cache:
            return self._meta_cache[chunk_id]
        result = self._col.get(ids=[chunk_id], include=["metadatas"])
        if result["ids"]:
            return MemoryChunk.from_dict(result["metadatas"][0])
        return None

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        layer_filter: Optional[list[MemoryLayer]] = None,
    ) -> list[tuple[MemoryChunk, float]]:
        where = None
        if layer_filter:
            layer_values = [l.value for l in layer_filter]
            where = {"layer": {"$in": layer_values}}

        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._col.count() or 1),
            where=where,
            include=["metadatas", "distances"],
        )
        out = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            chunk = MemoryChunk.from_dict(meta)
            if chunk_cached := self._meta_cache.get(chunk.id):
                chunk = chunk_cached
            similarity = 1 - dist  # ChromaDB cosine returns distance
            out.append((chunk, similarity))
        return out

    def all_chunks(self) -> list[MemoryChunk]:
        result = self._col.get(include=["metadatas"])
        return [MemoryChunk.from_dict(m) for m in result["metadatas"]]

    def count(self) -> int:
        return self._col.count()


# ---------------------------------------------------------------------------
# Pinecone backend
# ---------------------------------------------------------------------------

class PineconeVectorStore(VectorStore):
    """
    Production vector store using Pinecone serverless.

    Embeddings live in Pinecone (fast ANN search, auto-scaled).
    Chunk metadata is stored alongside as Pinecone metadata fields.
    A local in-memory cache keeps hot chunks fast and avoids
    redundant fetch calls.

    Install:
        pip install pinecone

    Setup:
        1. Create a free account at https://app.pinecone.io
        2. Create a serverless index:
              Name:   cmg-memory   (or any name)
              Metric: cosine
              Dimension: must match your embedding model
                - nomic-embed-text       -> 768
                - text-embedding-3-small -> 1536
                - all-MiniLM-L6-v2      -> 384
        3. Copy your API key from the Pinecone console

    Usage:
        from cmg.store import PineconeVectorStore
        from cmg import CMGMemory, create_adapter

        store = PineconeVectorStore(
            api_key="pcsk-...",
            index_name="cmg-memory",
            namespace="user-rishav",
            embedding_dim=768,
        )
        adapter = create_adapter("ollama", model="llama3.2")
        memory  = CMGMemory(adapter, store=store)

    Namespace:
        Use one namespace per user so memories never bleed across users.

    Migration from JSON:
        from cmg.store import PineconeVectorStore, InMemoryVectorStore
        old = InMemoryVectorStore(persist_path="./my_memory.json")
        new = PineconeVectorStore(api_key="...", index_name="cmg-memory")
        PineconeVectorStore.migrate_from(old, new)
    """

    _SCALAR_FIELDS = {
        "content", "layer", "memory_type", "score", "access_count",
        "created_at", "last_accessed", "last_updated", "confidence", "version",
    }
    _STRING_FIELDS = {"session_id", "source_text", "superseded_by"}
    _JSON_FIELDS   = {"tags", "metadata"}

    def __init__(
        self,
        api_key: str,
        index_name: str = "cmg-memory",
        namespace: str  = "default",
        embedding_dim: int = 768,
        cache_size: int = 1000,
    ):
        """
        Args:
            api_key:       Pinecone API key (from console.pinecone.io)
            index_name:    Name of the Pinecone index (must already exist)
            namespace:     Namespace within the index — one per user
            embedding_dim: Dimension matching your embedding model
            cache_size:    Max hot-cache size (chunks, not bytes)
        """
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "Install pinecone: pip install pinecone\n"
                "Then create a serverless index at https://app.pinecone.io"
            )

        self._pc        = Pinecone(api_key=api_key)
        self._namespace = namespace
        self._dim       = embedding_dim
        self._cache_size = cache_size
        self._cache: dict[str, MemoryChunk] = {}
        self._count: Optional[int] = None

        # Create the index if it does not exist yet
        existing = [idx.name for idx in self._pc.list_indexes()]
        if index_name not in existing:
            from pinecone import ServerlessSpec
            print(f"  Creating Pinecone index '{index_name}' (dim={embedding_dim}, metric=cosine)...")
            self._pc.create_index(
                name      = index_name,
                dimension = embedding_dim,
                metric    = "cosine",
                spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait until the index is ready
            import time
            for _ in range(30):
                status = self._pc.describe_index(index_name).status
                if getattr(status, "ready", False) or str(status).find("Ready") >= 0:
                    break
                time.sleep(2)
            print(f"  Index '{index_name}' ready.")

        self._index = self._pc.Index(index_name)

    # -- Write operations -----------------------------------------------------

    def upsert(self, chunk: MemoryChunk) -> None:
        if chunk.embedding is None:
            return
        self._index.upsert(
            vectors=[{
                "id":       chunk.id,
                "values":   chunk.embedding,
                "metadata": self._to_pinecone_meta(chunk),
            }],
            namespace=self._namespace,
        )
        self._cache[chunk.id] = chunk
        if len(self._cache) > self._cache_size:
            del self._cache[next(iter(self._cache))]
        self._count = None

    def upsert_batch(self, chunks: list[MemoryChunk], batch_size: int = 100) -> None:
        """Batch upsert — much faster than individual upserts for bulk loads."""
        valid = [c for c in chunks if c.embedding is not None]
        for i in range(0, len(valid), batch_size):
            batch = valid[i:i + batch_size]
            vectors = [
                {"id": c.id, "values": c.embedding,
                 "metadata": self._to_pinecone_meta(c)}
                for c in batch
            ]
            self._index.upsert(vectors=vectors, namespace=self._namespace)
            for c in batch:
                self._cache[c.id] = c
        self._count = None

    def delete(self, chunk_id: str) -> None:
        self._index.delete(ids=[chunk_id], namespace=self._namespace)
        self._cache.pop(chunk_id, None)
        self._count = None

    def delete_namespace(self) -> None:
        """Wipe ALL vectors in this namespace. Use with caution."""
        self._index.delete(delete_all=True, namespace=self._namespace)
        self._cache.clear()
        self._count = 0

    # -- Read operations ------------------------------------------------------

    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        if chunk_id in self._cache:
            return self._cache[chunk_id]
        result = self._index.fetch(ids=[chunk_id], namespace=self._namespace)
        vectors = result.get("vectors", {})
        if chunk_id not in vectors:
            return None
        vec   = vectors[chunk_id]
        chunk = self._from_pinecone_meta(chunk_id, vec["metadata"])
        chunk.embedding = vec["values"]
        self._cache[chunk_id] = chunk
        return chunk

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        layer_filter: Optional[list[MemoryLayer]] = None,
    ) -> list[tuple[MemoryChunk, float]]:
        """
        ANN search via Pinecone.
        Layer filter is pushed to Pinecone as a metadata filter — no
        client-side filtering needed, which keeps latency constant.
        """
        pinecone_filter = self._build_filter(layer_filter)

        response = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self._namespace,
            include_metadata=True,
            filter=pinecone_filter or None,
        )

        results = []
        for match in response.get("matches", []):
            chunk_id = match["id"]
            score    = float(match["score"])
            meta     = match.get("metadata", {})

            chunk = self._cache.get(chunk_id) or self._from_pinecone_meta(chunk_id, meta)
            self._cache[chunk_id] = chunk

            if chunk.superseded_by:
                continue
            results.append((chunk, score))

        return results

    def all_chunks(self) -> list[MemoryChunk]:
        """
        Fetch all chunks in this namespace.
        Uses Pinecone list() to get IDs, then batch-fetches metadata.
        Falls back to local cache if list() is unavailable.
        """
        try:
            all_ids = []
            for id_batch in self._index.list(namespace=self._namespace):
                all_ids.extend(id_batch)
            if not all_ids:
                return list(self._cache.values())
            chunks = []
            for i in range(0, len(all_ids), 100):
                batch_ids = all_ids[i:i + 100]
                to_fetch  = [cid for cid in batch_ids if cid not in self._cache]
                if to_fetch:
                    result = self._index.fetch(ids=to_fetch, namespace=self._namespace)
                    for cid, vec in result.get("vectors", {}).items():
                        chunk = self._from_pinecone_meta(cid, vec["metadata"])
                        chunk.embedding = vec["values"]
                        self._cache[cid] = chunk
                for cid in batch_ids:
                    if cid in self._cache:
                        chunks.append(self._cache[cid])
            return chunks
        except Exception:
            return list(self._cache.values())

    def count(self) -> int:
        if self._count is not None:
            return self._count
        try:
            stats = self._index.describe_index_stats()
            ns    = stats.get("namespaces", {}).get(self._namespace, {})
            self._count = ns.get("vector_count", 0)
        except Exception:
            self._count = len(self._cache)
        return self._count

    # -- Migration ------------------------------------------------------------

    @staticmethod
    def migrate_from(
        source: "VectorStore",
        target: "PineconeVectorStore",
        batch_size: int = 100,
    ) -> int:
        """
        Migrate all chunks from any VectorStore to Pinecone.
        Returns the number of chunks migrated.

        Example:
            old = InMemoryVectorStore(persist_path="./my_memory.json")
            new = PineconeVectorStore(api_key="...", index_name="cmg-memory")
            n   = PineconeVectorStore.migrate_from(old, new)
        """
        import logging
        log = logging.getLogger("cmg.store.migration")
        all_chunks = source.all_chunks()
        valid   = [c for c in all_chunks if c.embedding is not None]
        skipped = len(all_chunks) - len(valid)
        if skipped:
            log.warning(f"Skipping {skipped} chunks without embeddings")
        for i in range(0, len(valid), batch_size):
            target.upsert_batch(valid[i:i + batch_size], batch_size=batch_size)
            log.info(f"Migrated {min(i + batch_size, len(valid))}/{len(valid)} chunks")
        log.info(f"Done — {len(valid)} chunks in namespace '{target._namespace}'")
        return len(valid)

    # -- Metadata serialisation -----------------------------------------------

    def _to_pinecone_meta(self, chunk: MemoryChunk) -> dict:
        """
        Flatten MemoryChunk to Pinecone-compatible metadata.
        Pinecone accepts: str, int, float, bool, list[str] only.
        Nested dicts (tags, metadata) are JSON-serialised to strings.
        """
        import json
        return {
            "content":       chunk.content,
            "layer":         chunk.layer.value,
            "memory_type":   chunk.memory_type.value,
            "score":         float(chunk.score),
            "access_count":  int(chunk.access_count),
            "created_at":    float(chunk.created_at),
            "last_accessed": float(chunk.last_accessed),
            "last_updated":  float(chunk.last_updated),
            "confidence":    float(chunk.confidence),
            "version":       int(chunk.version),
            "session_id":    chunk.session_id or "",
            "source_text":   (chunk.source_text or "")[:1000],
            "superseded_by": chunk.superseded_by or "",
            "tags":          json.dumps(chunk.tags),
            "metadata_json": json.dumps(chunk.metadata),
        }

    def _from_pinecone_meta(self, chunk_id: str, meta: dict) -> MemoryChunk:
        """Reconstruct a MemoryChunk from Pinecone metadata."""
        import json
        chunk = MemoryChunk(
            id            = chunk_id,
            content       = meta.get("content", ""),
            layer         = MemoryLayer(meta.get("layer", "episodic")),
            memory_type   = MemoryType(meta.get("memory_type", "generic")),
            score         = float(meta.get("score", 0.5)),
            access_count  = int(meta.get("access_count", 0)),
            created_at    = float(meta.get("created_at", 0)),
            last_accessed = float(meta.get("last_accessed", 0)),
            last_updated  = float(meta.get("last_updated", 0)),
            confidence    = float(meta.get("confidence", 1.0)),
            version       = int(meta.get("version", 1)),
            session_id    = meta.get("session_id") or None,
            source_text   = meta.get("source_text") or None,
            superseded_by = meta.get("superseded_by") or None,
        )
        try:
            chunk.tags = json.loads(meta.get("tags", "[]"))
        except (json.JSONDecodeError, TypeError):
            chunk.tags = []
        try:
            chunk.metadata = json.loads(meta.get("metadata_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            chunk.metadata = {}
        return chunk

    def _build_filter(
        self, layer_filter: Optional[list[MemoryLayer]]
    ) -> Optional[dict]:
        if not layer_filter:
            return None
        if len(layer_filter) == 1:
            return {"layer": {"$eq": layer_filter[0].value}}
        return {"layer": {"$in": [l.value for l in layer_filter]}}