"""
CMG Memory System - Retrieval Engine

Handles:
  1. Embedding-based similarity search across memory layers
  2. Combined scoring (relevance × salience × recency)
  3. Prompt injection — formats retrieved memories as a system context block
  4. Access bumping — reinforces retrieved memories
"""

from __future__ import annotations
import math
import time
import logging
from typing import Optional

from .types import MemoryChunk, MemoryLayer, RetrievalResult
from .store import VectorStore
from .adapters import LLMAdapter

logger = logging.getLogger("cmg.retrieval")

# Layer weights — deeper layers get slight priority as they're more distilled
LAYER_WEIGHTS = {
    MemoryLayer.IDENTITY: 1.3,
    MemoryLayer.SEMANTIC: 1.2,
    MemoryLayer.EPISODIC: 1.0,
    MemoryLayer.WORKING: 0.8,
}

MEMORY_CONTEXT_TEMPLATE = """<memory_context>
The following is what you know about the user from previous conversations.
Use this to personalize your response. Do not explicitly mention this block.

{memories}
</memory_context>"""


class RetrievalEngine:
    """
    Retrieves relevant memories and injects them into prompts.
    
    Usage:
        engine = RetrievalEngine(store, llm_adapter)
        
        # Get relevant memories for a query
        results = engine.retrieve("what project am I working on?", top_k=5)
        
        # Get a formatted system prompt injection
        system_block = engine.build_memory_context("what project am I working on?")
    """

    def __init__(
        self,
        store: VectorStore,
        llm_adapter: LLMAdapter,
        recency_weight: float = 0.2,
        relevance_weight: float = 0.6,
        salience_weight: float = 0.2,
    ):
        self._store = store
        self._llm = llm_adapter
        self._recency_w = recency_weight
        self._relevance_w = relevance_weight
        self._salience_w = salience_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        layer_filter: Optional[list[MemoryLayer]] = None,
        min_score: float = 0.1,
        bump_access: bool = True,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-K most relevant memories for a query.
        
        Args:
            query: the user's current message / search query
            top_k: number of memories to return
            layer_filter: restrict to specific layers (None = all)
            min_score: discard results below this combined score
            bump_access: whether to reinforce retrieved memories
        
        Returns:
            List of RetrievalResult, sorted by combined_score descending
        """
        if self._store.count() == 0:
            return []

        # Get query embedding
        try:
            query_embedding = self._llm.embed(query)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return []

        # Search the store
        raw_results = self._store.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,  # Over-fetch to allow re-ranking
            layer_filter=layer_filter,
        )

        # Re-rank with combined score
        scored = []
        now = time.time()
        for chunk, similarity in raw_results:
            combined = self._combined_score(chunk, similarity, now)
            if combined >= min_score:
                scored.append(RetrievalResult(
                    chunk=chunk,
                    relevance_score=similarity,
                    combined_score=combined,
                    layer=chunk.layer,
                ))

        scored.sort(key=lambda r: r.combined_score, reverse=True)
        results = scored[:top_k]

        # Reinforce retrieved memories
        if bump_access:
            for result in results:
                result.chunk.bump_access()
                self._store.upsert(result.chunk)

        logger.debug(f"Retrieved {len(results)} memories for query: {query[:60]}...")
        return results

    def build_memory_context(
        self,
        query: str,
        top_k: int = 8,
        format_style: str = "structured",
    ) -> Optional[str]:
        """
        Build a formatted memory context block for injection into the system prompt.
        
        Args:
            query: current user message
            top_k: number of memories to include
            format_style: "structured" | "prose" | "minimal"
        
        Returns:
            Formatted string to prepend to the system prompt, or None if no memories.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return None

        if format_style == "structured":
            memory_lines = self._format_structured(results)
        elif format_style == "prose":
            memory_lines = self._format_prose(results)
        else:
            memory_lines = self._format_minimal(results)

        return MEMORY_CONTEXT_TEMPLATE.format(memories=memory_lines)

    def _combined_score(
        self, chunk: MemoryChunk, similarity: float, now: float
    ) -> float:
        """
        Combined score = weighted sum of:
          - Relevance (embedding similarity)
          - Salience (chunk.score × layer weight)
          - Recency (exponential decay from last access)
        """
        layer_weight = LAYER_WEIGHTS.get(chunk.layer, 1.0)
        salience = chunk.score * layer_weight

        hours_since = (now - chunk.last_accessed) / 3600
        recency = math.exp(-hours_since / 168)  # 7-day half-life for recency

        combined = (
            self._relevance_w * similarity
            + self._salience_w * salience
            + self._recency_w * recency
        )
        return combined

    def _format_structured(self, results: list[RetrievalResult]) -> str:
        """Group memories by type, each on its own line."""
        lines = []
        by_type: dict[str, list[str]] = {}
        for r in results:
            t = r.chunk.memory_type.value
            by_type.setdefault(t, []).append(r.chunk.content)

        type_labels = {
            "fact": "Facts",
            "preference": "Preferences",
            "goal": "Current goals",
            "skill": "Skills & expertise",
            "decision": "Recent decisions",
            "correction": "Corrections",
            "relationship": "Relationships",
            "event": "Recent events",
            "generic": "Other",
        }
        for mem_type, contents in by_type.items():
            label = type_labels.get(mem_type, mem_type.capitalize())
            lines.append(f"[{label}]")
            for c in contents:
                lines.append(f"  - {c}")
        return "\n".join(lines)

    def _format_prose(self, results: list[RetrievalResult]) -> str:
        """Bullet list, no grouping."""
        return "\n".join(f"• {r.chunk.content}" for r in results)

    def _format_minimal(self, results: list[RetrievalResult]) -> str:
        """Most compressed form for tight context budgets."""
        return "; ".join(r.chunk.content[:100] for r in results)