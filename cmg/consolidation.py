"""
CMG Memory System - Consolidation Engine

This is the "sleep pass" — the background process that:
  1. Decays stale memories (exponential decay based on time + access)
  2. Promotes salient memories to deeper layers
  3. Merges semantically duplicate chunks
  4. Detects and resolves contradictions
  5. Forgets chunks whose score drops below threshold

Runs asynchronously after each session, never in the critical path.
"""

from __future__ import annotations
import time
import math
import re
import logging
from typing import Optional

from .types import MemoryChunk, MemoryLayer, MemoryType, ConsolidationReport
# MemoryType used in _compute_salience fallback path
from .store import VectorStore

logger = logging.getLogger("cmg.consolidation")


# ---------------------------------------------------------------------------
# Decay functions
# ---------------------------------------------------------------------------

def _ebbinghaus_decay(score: float, hours_since_access: float, decay_rate: float = 0.1) -> float:
    """
    Ebbinghaus forgetting curve: R = e^(-t/S)
    where t = time, S = stability (higher score = more stable memory).
    """
    stability = max(0.5, score * 10)  # Score maps to stability [0.5, 10]
    retention = math.exp(-hours_since_access / stability / decay_rate)
    return score * retention


def _compute_salience(chunk: MemoryChunk) -> float:
    """
    Salience for layer promotion decisions.

    When a chunk was scored by the LLM (metadata["scoring"]["llm_score"] exists),
    we use that as the quality anchor — the LLM already reasoned about persistence,
    specificity, and utility, so we don't re-apply type weights on top.

    For chunks scored by the old type-prior fallback, we apply a light type
    multiplier to compensate for the weaker signal.

    Formula:
        salience = quality_anchor
                 + log(1 + access_count) / 10   ← reinforcement bonus
                 + recency_bonus * 0.15          ← 48h recency half-life

    quality_anchor:
        LLM-scored  → chunk.score  (already high-quality)
        Type-prior  → chunk.score × type_weight  (needs the boost)
    """
    scoring_meta = chunk.metadata.get("scoring", {})
    has_llm_score = "llm_score" in scoring_meta and "[fallback" not in scoring_meta.get("llm_reasoning", "")

    if has_llm_score:
        # LLM scored — trust the score directly, no type multiplier
        quality_anchor = chunk.score
    else:
        # Type-prior fallback — apply multiplier to compensate
        type_weights = {
            MemoryType.CORRECTION: 1.4,
            MemoryType.GOAL: 1.3,
            MemoryType.DECISION: 1.2,
            MemoryType.SKILL: 1.15,
            MemoryType.PREFERENCE: 1.1,
            MemoryType.FACT: 1.0,
            MemoryType.RELATIONSHIP: 1.0,
            MemoryType.EVENT: 0.9,
            MemoryType.GENERIC: 0.8,
        }
        type_weight = type_weights.get(chunk.memory_type, 1.0)
        quality_anchor = chunk.score * type_weight

    # Access frequency: log scale — each retrieval adds diminishing returns
    access_bonus = math.log1p(chunk.access_count) / 10.0

    # Recency: recent access boosts salience, 48h half-life
    hours_ago = chunk.hours_since_access()
    recency_bonus = math.exp(-hours_ago / 48)

    salience = quality_anchor + access_bonus + recency_bonus * 0.15
    return min(1.0, salience)


# ---------------------------------------------------------------------------
# Layer promotion thresholds
# ---------------------------------------------------------------------------

# Promotion thresholds — now session-count based, not time-based.
# Each tuple: (min_salience, min_sessions_seen, min_score, target_layer)
#
# sessions_seen = number of distinct sessions in which this chunk was
# reinforced (either re-extracted or retrieved). This is a much stronger
# signal than time elapsed, because it means the user mentioned this fact
# across multiple independent conversations.
#
# Identity types (name, role, core prefs) get an accelerated path:
# they only need 2 sessions to reach semantic instead of 3.

PROMOTION_THRESHOLDS = {
    # layer            min_salience  min_sessions  min_score   target
    MemoryLayer.WORKING:  (0.45,        1,           0.40,   MemoryLayer.EPISODIC),
    MemoryLayer.EPISODIC: (0.60,        2,           0.50,   MemoryLayer.SEMANTIC),
    MemoryLayer.SEMANTIC: (0.75,        4,           0.70,   MemoryLayer.IDENTITY),
}

# Memory types that get fast-tracked to semantic after just 1 session
# because they are inherently stable identity facts
FAST_TRACK_TYPES = {
    MemoryType.FACT,
    MemoryType.PREFERENCE,
    MemoryType.SKILL,
    MemoryType.GOAL,
    MemoryType.CORRECTION,
}
FAST_TRACK_SCORE_THRESHOLD = 0.72  # must be high-confidence to fast-track

FORGETTING_THRESHOLD = 0.05   # Score below this → chunk is deleted
MERGE_SIMILARITY_THRESHOLD = 0.65  # Cosine similarity above this → candidates for merge




def _get_sessions_seen(chunk) -> int:
    """
    Return the number of distinct sessions in which this chunk has been
    reinforced (re-extracted or retrieved and bumped).
    Stored in chunk.metadata['sessions_seen'] as a list of session IDs.
    """
    return len(set(chunk.metadata.get('sessions_seen', [chunk.session_id or 'init'])))


def _record_session(chunk, session_id: str) -> None:
    """Record that this chunk was active in session_id."""
    if not session_id:
        return
    sessions = chunk.metadata.get('sessions_seen', [])
    if session_id not in sessions:
        sessions.append(session_id)
        chunk.metadata['sessions_seen'] = sessions

# ---------------------------------------------------------------------------
# Consolidation engine
# ---------------------------------------------------------------------------

class ConsolidationEngine:
    """
    Runs the memory maintenance cycle.
    
    Usage:
        engine = ConsolidationEngine(store)
        report = engine.run()
    """

    def __init__(
        self,
        store: VectorStore,
        decay_rate: float = 0.1,
        forgetting_threshold: float = FORGETTING_THRESHOLD,
        merge_threshold: float = MERGE_SIMILARITY_THRESHOLD,
    ):
        self._store = store
        self._decay_rate = decay_rate
        self._forgetting_threshold = forgetting_threshold
        self._merge_threshold = merge_threshold

    def run(self, llm_adapter=None) -> ConsolidationReport:
        """
        Full consolidation pass. Pass llm_adapter to enable LLM-powered
        contradiction detection and intelligent merging.
        """
        report = ConsolidationReport()

        chunks = self._store.all_chunks()
        logger.info(f"Consolidation: processing {len(chunks)} chunks")

        # Step 1: Decay all chunks
        for chunk in chunks:
            if chunk.layer == MemoryLayer.WORKING:
                continue  # Working memory has its own TTL (session-scoped)
            if chunk.superseded_by:
                continue

            hours = chunk.hours_since_access()
            new_score = _ebbinghaus_decay(chunk.score, hours, self._decay_rate)
            chunk.score = new_score
            chunk.last_updated = time.time()
            self._store.upsert(chunk)
            report.decayed += 1

        # Step 2: Forget low-score chunks
        for chunk in self._store.all_chunks():
            if chunk.layer == MemoryLayer.IDENTITY:
                continue  # Never auto-forget identity layer
            if chunk.superseded_by:
                # Clean up superseded chunks
                self._store.delete(chunk.id)
                report.forgotten += 1
                continue
            if chunk.score < self._forgetting_threshold:
                logger.debug(f"Forgetting chunk {chunk.id[:8]}: score={chunk.score:.3f}")
                self._store.delete(chunk.id)
                report.forgotten += 1

        # Step 3: Promote chunks that meet thresholds
        # Primary signal: sessions_seen (how many distinct sessions mentioned this)
        # Secondary: salience score and DFS score
        # Fast-track: high-confidence identity facts go episodic→semantic in 1 session
        for chunk in self._store.all_chunks():
            if chunk.layer not in PROMOTION_THRESHOLDS:
                continue
            if chunk.superseded_by:
                continue

            min_salience, min_sessions, min_score, target_layer = PROMOTION_THRESHOLDS[chunk.layer]
            salience      = _compute_salience(chunk)
            sessions_seen = _get_sessions_seen(chunk)

            # Fast-track path: high-quality identity facts after just 1 session
            # Goals use a lower threshold — a stated goal is inherently stable
            goal_threshold = 0.45 if chunk.memory_type == MemoryType.GOAL else FAST_TRACK_SCORE_THRESHOLD
            fast_track = (
                chunk.layer == MemoryLayer.EPISODIC
                and target_layer == MemoryLayer.SEMANTIC
                and chunk.memory_type in FAST_TRACK_TYPES
                and chunk.score >= goal_threshold
                and sessions_seen >= 1
            )

            # Standard promotion path
            standard = (
                salience      >= min_salience
                and sessions_seen >= min_sessions
                and chunk.score   >= min_score
            )

            if fast_track or standard:
                path = "fast-track" if fast_track else "standard"
                logger.info(
                    "Promoting [%s] %s from %s → %s  "
                    "(sessions=%d score=%.3f salience=%.3f)",
                    path, chunk.id[:8], chunk.layer.value, target_layer.value,
                    sessions_seen, chunk.score, salience,
                )
                chunk.layer = target_layer
                chunk.last_updated = time.time()
                self._store.upsert(chunk)
                report.promoted += 1

        # Step 4: Merge near-duplicate chunks (embedding-based)
        report.merged += self._merge_duplicates()

        # Step 5: LLM-powered contradiction detection (optional)
        if llm_adapter:
            report.contradictions += self._detect_contradictions(llm_adapter)

        logger.info(
            f"Consolidation done: promoted={report.promoted}, merged={report.merged}, "
            f"decayed={report.decayed}, forgotten={report.forgotten}, "
            f"contradictions={report.contradictions}"
        )
        return report

    def _merge_duplicates(self) -> int:
        """
        Find chunks with high embedding similarity within the same layer
        and merge them into a single, higher-confidence chunk.
        Returns number of merges performed.
        """
        merged_count = 0
        chunks = [c for c in self._store.all_chunks() if c.embedding and not c.superseded_by]

        visited = set()
        for i, chunk_a in enumerate(chunks):
            if chunk_a.id in visited:
                continue
            candidates = []
            for j, chunk_b in enumerate(chunks):
                if i == j or chunk_b.id in visited:
                    continue
                if chunk_a.layer != chunk_b.layer:
                    continue
                if chunk_a.memory_type != chunk_b.memory_type:
                    continue
                from .store import _cosine_similarity
                if chunk_a.embedding and chunk_b.embedding:
                    sim = _cosine_similarity(chunk_a.embedding, chunk_b.embedding)
                    if sim >= self._merge_threshold:
                        candidates.append((chunk_b, sim))

            if candidates:
                # Keep the chunk with highest score; mark others as superseded
                best = max([(chunk_a, 0.0)] + candidates, key=lambda x: x[0].score)[0]
                for cand, _ in candidates:
                    if cand.id != best.id:
                        logger.debug(f"Merging chunk {cand.id[:8]} → {best.id[:8]}")
                        # Boost the surviving chunk's confidence
                        best.confidence = min(1.0, best.confidence + 0.1)
                        best.score = min(1.0, best.score + 0.05)
                        best.access_count += cand.access_count
                        best.last_updated = time.time()
                        # Mark superseded
                        cand.superseded_by = best.id
                        self._store.upsert(cand)
                        visited.add(cand.id)
                        merged_count += 1
                self._store.upsert(best)
                visited.add(best.id)

        return merged_count

    def _detect_contradictions(self, llm_adapter) -> int:
        """
        Use the LLM to detect contradicting facts in the semantic layer.
        Returns number of contradictions resolved.
        """
        semantic_facts = [
            c for c in self._store.all_chunks()
            if c.layer in (MemoryLayer.SEMANTIC, MemoryLayer.IDENTITY)
            and c.memory_type in (MemoryType.FACT, MemoryType.PREFERENCE)
            and not c.superseded_by
        ]

        if len(semantic_facts) < 2:
            return 0

        facts_text = "\n".join(
            f"[{i}] (score={c.score:.2f}, created={c.created_at:.0f}) {c.content}"
            for i, c in enumerate(semantic_facts)
        )

        prompt = f"""You are a memory auditor. Review these stored facts about a user.
Identify any pairs that DIRECTLY CONTRADICT each other (not just related).
Return JSON: [{{"older_index": int, "newer_index": int, "reason": str}}]
Return [] if no contradictions. Only return valid JSON.

Facts:
{facts_text}
"""
        try:
            response = llm_adapter.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            import json, re
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if not match:
                return 0
            contradictions = json.loads(match.group())
        except Exception as e:
            logger.warning(f"Contradiction detection failed: {e}")
            return 0

        resolved = 0
        for item in contradictions:
            try:
                older = semantic_facts[item["older_index"]]
                newer = semantic_facts[item["newer_index"]]
                logger.info(f"Contradiction: '{older.content[:50]}' vs '{newer.content[:50]}'")
                # Keep newer, mark older as superseded
                older.superseded_by = newer.id
                newer.confidence = min(1.0, newer.confidence + 0.1)
                newer.metadata["resolved_contradiction"] = item.get("reason", "")
                self._store.upsert(older)
                self._store.upsert(newer)
                resolved += 1
            except (IndexError, KeyError):
                continue

        return resolved