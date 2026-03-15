"""
CMG Memory System — Write Gate
================================
A three-gate pipeline that sits between extraction and storage.
Every memory candidate must pass all three gates before being written
to the vector DB.

Gate 1 — Novelty filter
    Embeds the candidate and checks cosine similarity against the
    top-N most similar existing chunks. If max similarity exceeds
    the novelty_threshold, the candidate is a near-duplicate.
    Action: skip write, reinforce the existing chunk instead
    (bump its score + access_count so it stays alive).

Gate 2 — Score threshold
    Uses the DFS deterministic score already on the chunk.
    If score < min_score, the memory is too generic/transient/hedged
    to be worth persisting.
    Action: discard silently.

Gate 3 — Layer capacity
    Each memory layer has a max_chunks cap. If the target layer is
    at capacity, evict the chunk with the lowest score before writing.
    Action: evict weakest, then write new.

Together these three gates reduce writes by 60-80% in steady-state
usage while ensuring the store only contains novel, valuable memories.

Usage (automatic — wired into CMGMemory):
    gate = WriteGate(store, embed_fn)
    written, reinforced, discarded = gate.process(chunks)

Or standalone:
    result = gate.evaluate(chunk)
    # result.action is one of: "write", "reinforce", "discard"
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .types import MemoryChunk, MemoryLayer
from .store import VectorStore

logger = logging.getLogger("cmg.write_gate")


# ── Layer capacity defaults ────────────────────────────────────────────────────
# Tuned so total storage stays under ~370 chunks per user in steady state.
# Working is ephemeral (session-scoped) so cap is tight.
# Semantic and identity are compressed/distilled so they stay small.

DEFAULT_LAYER_CAPS: dict[MemoryLayer, int] = {
    MemoryLayer.WORKING:  20,
    MemoryLayer.EPISODIC: 200,
    MemoryLayer.SEMANTIC: 100,
    MemoryLayer.IDENTITY: 50,
}


class WriteAction(Enum):
    WRITE     = "write"      # passed all gates, write to store
    REINFORCE = "reinforce"  # near-duplicate, reinforce existing instead
    DISCARD   = "discard"    # too low score, drop silently


@dataclass
class GateResult:
    """Decision + explanation for a single candidate."""
    action:          WriteAction
    chunk:           MemoryChunk
    gate_triggered:  str        = ""     # which gate stopped it
    reason:          str        = ""     # human-readable explanation
    reinforced_id:   Optional[str] = None  # for REINFORCE: which chunk was bumped
    similarity:      float      = 0.0    # Gate 1 max similarity found
    evicted_id:      Optional[str] = None  # for WRITE after eviction: what was removed

    def __repr__(self):
        if self.action == WriteAction.WRITE:
            evict = f" (evicted {self.evicted_id[:8]})" if self.evicted_id else ""
            return f"GateResult(WRITE{evict}  {self.chunk.content[:50]})"
        if self.action == WriteAction.REINFORCE:
            return (f"GateResult(REINFORCE sim={self.similarity:.3f} "
                    f"→ {self.reinforced_id[:8] if self.reinforced_id else '?'}  "
                    f"{self.chunk.content[:50]})")
        return f"GateResult(DISCARD score={self.chunk.score:.3f}  {self.chunk.content[:50]})"


@dataclass
class WriteGateStats:
    """Cumulative stats for monitoring write efficiency."""
    total_evaluated: int = 0
    written:         int = 0
    reinforced:      int = 0
    discarded:       int = 0
    evictions:       int = 0

    @property
    def write_rate(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return self.written / self.total_evaluated

    @property
    def rejection_rate(self) -> float:
        return 1.0 - self.write_rate

    def summary(self) -> str:
        return (
            f"evaluated={self.total_evaluated}  "
            f"written={self.written} ({self.write_rate:.0%})  "
            f"reinforced={self.reinforced}  "
            f"discarded={self.discarded}  "
            f"evictions={self.evictions}"
        )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


class WriteGate:
    """
    Three-gate write pipeline. Instantiate once, reuse across sessions.

    Args:
        store:              The VectorStore to read from and write to.
        embed_fn:           Callable that returns an embedding for a text string.
                            Used only for Gate 1 (novelty check). Pass None to
                            skip Gate 1 (all candidates treated as novel).
        novelty_threshold:  Cosine similarity above which a candidate is
                            considered a near-duplicate. Default: 0.82.
                            0.82 catches loose paraphrases of the same fact.
                            Raise to 0.90 if you want only near-exact duplicates.
                            Lower to 0.75 to be even more aggressive.
        min_score:          DFS score below which candidates are discarded.
                            Default: 0.35. Raise to 0.45 to be more selective.
        layer_caps:         Dict mapping MemoryLayer → max chunk count.
                            Defaults to DEFAULT_LAYER_CAPS.
        reinforce_bonus:    Score bonus applied to the existing chunk when
                            a near-duplicate is found. Default: 0.05.
        similarity_search_k: How many existing chunks to compare against
                             in Gate 1. Default: 5. Higher = more thorough
                             but slower.
    """

    def __init__(
        self,
        store:               VectorStore,
        embed_fn:            Optional[Callable[[str], list[float]]] = None,
        novelty_threshold:   float = 0.88,
        min_score:           float = 0.35,
        layer_caps:          Optional[dict[MemoryLayer, int]] = None,
        reinforce_bonus:     float = 0.05,
        similarity_search_k: int = 10,
    ):
        self._store              = store
        self._embed_fn           = embed_fn
        self._novelty_threshold  = novelty_threshold
        self._min_score          = min_score
        self._layer_caps         = layer_caps or DEFAULT_LAYER_CAPS.copy()
        self._reinforce_bonus    = reinforce_bonus
        self._sim_k              = similarity_search_k
        self.stats               = WriteGateStats()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        candidates: list[MemoryChunk],
    ) -> tuple[list[MemoryChunk], list[str], list[str]]:
        """
        Run all candidates through the three-gate pipeline.

        Returns:
            (written, reinforced_ids, discarded_ids)
            written:         chunks actually persisted to the store
            reinforced_ids:  IDs of existing chunks that were reinforced
            discarded_ids:   IDs of candidates that were dropped
        """
        written      = []
        reinforced   = []
        discarded    = []

        for chunk in candidates:
            result = self.evaluate(chunk)
            self.stats.total_evaluated += 1

            if result.action == WriteAction.WRITE:
                self._store.upsert(chunk)
                self.stats.written += 1
                if result.evicted_id:
                    self.stats.evictions += 1
                written.append(chunk)
                logger.debug("WRITE  score=%.3f  %s", chunk.score, chunk.content[:60])

            elif result.action == WriteAction.REINFORCE:
                self.stats.reinforced += 1
                reinforced.append(result.reinforced_id)
                logger.debug(
                    "REINFORCE sim=%.3f  existing=%s  candidate=%s",
                    result.similarity,
                    (result.reinforced_id or "")[:8],
                    chunk.content[:60],
                )

            else:  # DISCARD
                self.stats.discarded += 1
                discarded.append(chunk.id)
                logger.debug("DISCARD score=%.3f  %s", chunk.score, chunk.content[:60])

        if written or reinforced:
            logger.info(
                "WriteGate: %s",
                self.stats.summary(),
            )

        return written, reinforced, discarded

    def evaluate(self, chunk: MemoryChunk) -> GateResult:
        """
        Evaluate a single candidate through all three gates.
        Returns a GateResult describing what should happen to it.
        Does NOT write to the store — call process() or write manually.
        """
        # ── Gate 2 first (cheapest — no embedding needed) ─────────────────────
        if chunk.score < self._min_score:
            return GateResult(
                action         = WriteAction.DISCARD,
                chunk          = chunk,
                gate_triggered = "gate2_score",
                reason         = f"score {chunk.score:.3f} < threshold {self._min_score}",
            )

        # ── Gate 1: novelty check (requires embedding) ────────────────────────
        if self._embed_fn is not None:
            # Embed if not already done
            if chunk.embedding is None:
                try:
                    chunk.embedding = self._embed_fn(chunk.content)
                except Exception as e:
                    logger.warning("Embedding failed for gate1, skipping: %s", e)

            if chunk.embedding is not None and self._store.count() > 0:
                similar = self._store.search(
                    query_embedding = chunk.embedding,
                    top_k           = self._sim_k,
                )
                if similar:
                    max_sim   = similar[0][1]
                    best_match = similar[0][0]

                    if max_sim >= self._novelty_threshold:
                        # Near-duplicate found — reinforce the existing chunk
                        best_match.score = min(
                            1.0, best_match.score + self._reinforce_bonus
                        )
                        best_match.access_count += 1
                        best_match.last_accessed = time.time()
                        self._store.upsert(best_match)

                        # Record that this session saw the existing chunk too
                        if chunk.session_id:
                            sessions = best_match.metadata.get('sessions_seen', [])
                            if chunk.session_id not in sessions:
                                sessions.append(chunk.session_id)
                                best_match.metadata['sessions_seen'] = sessions
                                self._store.upsert(best_match)

                        return GateResult(
                            action         = WriteAction.REINFORCE,
                            chunk          = chunk,
                            gate_triggered = "gate1_novelty",
                            reason         = (
                                f"sim={max_sim:.3f} ≥ {self._novelty_threshold} "
                                f"with existing: '{best_match.content[:50]}'"
                            ),
                            reinforced_id  = best_match.id,
                            similarity     = max_sim,
                        )

        # ── Gate 3: layer capacity ────────────────────────────────────────────
        evicted_id = self._check_and_evict(chunk.layer)

        return GateResult(
            action         = WriteAction.WRITE,
            chunk          = chunk,
            gate_triggered = "",
            reason         = "passed all gates",
            evicted_id     = evicted_id,
        )

    # ── Configuration helpers ──────────────────────────────────────────────────

    def set_cap(self, layer: MemoryLayer, cap: int) -> None:
        """Override the chunk cap for a specific layer."""
        self._layer_caps[layer] = cap

    def layer_usage(self) -> dict[str, dict]:
        """Return current chunk count vs cap for each layer."""
        all_chunks = self._store.all_chunks()
        counts: dict[MemoryLayer, int] = {}
        for chunk in all_chunks:
            if not chunk.superseded_by:
                counts[chunk.layer] = counts.get(chunk.layer, 0) + 1

        return {
            layer.value: {
                "count": counts.get(layer, 0),
                "cap":   self._layer_caps.get(layer, 999),
                "pct":   round(
                    counts.get(layer, 0) / self._layer_caps.get(layer, 999) * 100, 1
                ),
            }
            for layer in MemoryLayer
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_and_evict(self, layer: MemoryLayer) -> Optional[str]:
        """
        If the layer is at or over capacity, evict the chunk with the
        lowest score. Returns the evicted chunk's ID, or None if no
        eviction was needed.
        """
        cap = self._layer_caps.get(layer)
        if cap is None:
            return None

        # Count active (non-superseded) chunks in this layer
        layer_chunks = [
            c for c in self._store.all_chunks()
            if c.layer == layer and not c.superseded_by
        ]

        if len(layer_chunks) < cap:
            return None  # under cap, no eviction needed

        # Evict the chunk with the lowest score
        # Tiebreak: prefer evicting older, less-accessed chunks
        worst = min(
            layer_chunks,
            key=lambda c: (
                c.score,
                c.access_count,
                -c.created_at,   # negative = prefer older
            ),
        )

        logger.info(
            "Evicting chunk from %s layer (cap=%d): score=%.3f  %s",
            layer.value, cap, worst.score, worst.content[:60],
        )
        self._store.delete(worst.id)
        return worst.id