"""
CMG Memory System — Contextual Resonance Retrieval (CRR)
==========================================================
A noise-resistant retrieval engine built on four orthogonal signals
fused through an adaptive gate, followed by MMR diversification.

Replaces the cosine-only retrieval in retrieval.py.

Signal architecture:

  S1  Semantic affinity      max(query·memory_emb, query·source_emb) × confidence
                             Uses source context when memory text is sparse.

  S2  Temporal context       Decay shaped by age + access recency,
                             with query-time temporal frame bonuses.
                             ("recently" boosts recent memories, etc.)

  S3  Type resonance         Infers desired memory type from query vocabulary,
                             applies a multiplier per memory_type match.
                             Purely deterministic — no LLM call.

  S4  Reinforcement history  log(1 + access_count) × recency_of_access.
                             Encodes "this memory has been useful before."

Fusion:
  resonance = α·S1 + β·S2 + γ·S3 + δ·S4
  Weights are tunable. Default: α=0.45, β=0.20, γ=0.20, δ=0.15.

MMR diversification (post-fusion):
  Iteratively selects next result maximising:
    score(c) - λ · max_similarity(c, already_selected)
  Eliminates near-duplicate memories crowding the top-K.
  λ=0.5 by default (balance between relevance and diversity).

Zero LLM calls. Fully deterministic. Model-agnostic.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .types import MemoryChunk, MemoryLayer, MemoryType, RetrievalResult
from .store import VectorStore


# ─────────────────────────────────────────────────────────────────────────────
# Signal scores dataclass — transparent per-result breakdown
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResonanceScore:
    """
    Full score breakdown for a single retrieved chunk.
    Stored so you can inspect exactly why a chunk ranked where it did.
    """
    chunk_id: str
    s1_semantic:      float = 0.0
    s2_temporal:      float = 0.0
    s3_type:          float = 0.0
    s4_reinforcement: float = 0.0
    resonance:        float = 0.0   # Fused pre-MMR score
    mmr_score:        float = 0.0   # Final post-MMR score
    mmr_penalty:      float = 0.0   # How much MMR subtracted

    # S1 sub-signals
    s1_memory_sim:  float = 0.0
    s1_source_sim:  float = 0.0

    # S3 sub-signals
    s3_detected_intent: str = ""
    s3_type_match:      bool = False

    def summary(self) -> str:
        return (
            f"S1={self.s1_semantic:.3f} "
            f"S2={self.s2_temporal:.3f} "
            f"S3={self.s3_type:.3f}({'✓' if self.s3_type_match else '·'}) "
            f"S4={self.s4_reinforcement:.3f} "
            f"→ res={self.resonance:.3f} "
            f"mmr={self.mmr_score:.3f}"
        )


@dataclass
class CRRWeights:
    """Fusion weights for the four signals. Must be non-negative."""
    alpha: float = 0.45   # S1 semantic affinity    (primary)
    beta:  float = 0.20   # S2 temporal context
    gamma: float = 0.20   # S3 type resonance
    delta: float = 0.15   # S4 reinforcement history

    # MMR diversity parameter λ ∈ [0, 1]
    # 0.0 = pure relevance (no diversity), 1.0 = pure diversity (no relevance)
    mmr_lambda: float = 0.5

    def normalised(self) -> "CRRWeights":
        """Return a copy with weights summing to 1.0."""
        total = self.alpha + self.beta + self.gamma + self.delta
        if total == 0:
            return CRRWeights()
        return CRRWeights(
            alpha=self.alpha / total,
            beta=self.beta / total,
            gamma=self.gamma / total,
            delta=self.delta / total,
            mmr_lambda=self.mmr_lambda,
        )


# ─────────────────────────────────────────────────────────────────────────────
# S1 — Semantic affinity
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _s1_semantic(
    query_emb: list[float],
    chunk: MemoryChunk,
    source_embeddings: dict[str, list[float]],  # chunk_id -> source_emb
) -> tuple[float, float, float]:
    """
    S1 = max(similarity_to_memory, similarity_to_source) × confidence

    Using the source embedding catches cases where the memory was distilled
    to a short sentence that lost semantic richness.

    Returns (s1_score, memory_sim, source_sim).
    """
    memory_sim = _cosine(query_emb, chunk.embedding) if chunk.embedding else 0.0
    source_sim = _cosine(query_emb, source_embeddings.get(chunk.id, []))

    # Soft max: weight towards whichever is more informative
    # (pure max can over-reward a noisy source embedding)
    best = max(memory_sim, source_sim)
    second = min(memory_sim, source_sim)
    blended = 0.75 * best + 0.25 * second

    s1 = blended * chunk.confidence
    return round(s1, 4), round(memory_sim, 4), round(source_sim, 4)


# ─────────────────────────────────────────────────────────────────────────────
# S2 — Temporal context
# ─────────────────────────────────────────────────────────────────────────────

# Temporal frame markers in queries → which memories they boost
_TEMPORAL_FRAMES: list[tuple[str, str, float]] = [
    # (pattern, frame_name, recency_bias) — higher bias = prefer newer memories
    (r"\b(recently|lately|these days|nowadays)\b",     "recent",    1.8),
    (r"\blast\s+(week|month|year|session|time)\b",     "recent",    1.6),
    (r"\b(just|just\s+now|right\s+now)\b",             "immediate", 2.0),
    (r"\b(always|consistently|generally|usually)\b",   "habitual",  0.5),  # prefer older stable
    (r"\b(still|anymore|any\s+longer)\b",              "ongoing",   1.2),
    (r"\b(first|originally|initially|used\s+to)\b",    "historical",0.3),  # prefer older
    (r"\b(current|currently|now|today)\b",             "present",   1.4),
    (r"\b(long.term|permanently|forever)\b",           "permanent", 0.4),
]

def _detect_temporal_frame(query: str) -> tuple[str, float]:
    """
    Detect what temporal frame the query is asking about.
    Returns (frame_name, recency_bias_multiplier).
    """
    query_lower = query.lower()
    for pattern, frame, bias in _TEMPORAL_FRAMES:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return frame, bias
    return "neutral", 1.0


def _s2_temporal(
    chunk: MemoryChunk,
    now: float,
    recency_bias: float = 1.0,
) -> float:
    """
    S2 = base_decay × access_recency × temporal_frame_fit

    base_decay: Ebbinghaus-inspired, modulated by chunk score (stability)
    access_recency: how recently was this chunk last accessed?
    temporal_frame_fit: does the memory's age match what the query is asking about?
    """
    age_hours     = (now - chunk.created_at) / 3600
    access_hours  = (now - chunk.last_accessed) / 3600

    # Base decay: stable memories (high score) decay slower
    stability = max(0.5, chunk.score * 8.0)
    base_decay = math.exp(-age_hours / (stability * 24))

    # Access recency: memories accessed recently are hot
    access_recency = math.exp(-access_hours / 72)   # 72h half-life

    # Temporal frame fit: boost or penalise based on what query asked for
    # "recently" → prefer memories created/accessed in last 7 days
    # "always/habitual" → prefer older, stable memories
    if recency_bias > 1.0:
        # Query wants recent memories — boost memories accessed in last 7 days
        recency_match = math.exp(-access_hours / (24 * 7 * (1 / (recency_bias - 0.8))))
        frame_fit = 0.5 + 0.5 * recency_match
    elif recency_bias < 0.8:
        # Query wants older/stable memories — boost older ones
        age_days = age_hours / 24
        frame_fit = 0.3 + 0.7 * (1 - math.exp(-age_days / 30))
    else:
        frame_fit = 1.0  # neutral

    s2 = base_decay * (0.6 + 0.4 * access_recency) * frame_fit
    return round(min(1.0, s2), 4)


# ─────────────────────────────────────────────────────────────────────────────
# S3 — Type resonance
# ─────────────────────────────────────────────────────────────────────────────

# Maps query intent signals → {memory_type: multiplier}
# Multiple intents can fire; the highest multiplier for the chunk's type wins
_INTENT_RULES: list[tuple[str, str, dict[str, float]]] = [
    # (pattern, intent_name, {memory_type: multiplier})
    (
        r"\b(tool|method|technique|use|using|framework|library|stack)\b",
        "tool_query",
        {"skill": 1.5, "preference": 1.4, "fact": 1.1, "generic": 0.8},
    ),
    (
        r"\b(goal|plan|aim|objective|trying|want\s+to|intend\s+to|roadmap)\b",
        "goal_query",
        {"goal": 1.6, "decision": 1.3, "fact": 0.9, "event": 0.8},
    ),
    (
        r"\b(prefer|like|hate|love|dislike|favourite|style|format)\b",
        "preference_query",
        {"preference": 1.6, "correction": 1.3, "generic": 0.8},
    ),
    (
        r"\b(work|job|role|position|company|employer|research|project)\b",
        "work_query",
        {"fact": 1.4, "skill": 1.3, "goal": 1.2, "relationship": 1.1},
    ),
    (
        r"\b(who|colleague|team|advisor|professor|supervisor|mentor|friend)\b",
        "relationship_query",
        {"relationship": 1.7, "fact": 1.1, "event": 0.9},
    ),
    (
        r"\b(happen|did|when|event|occurred|meeting|conference|yesterday)\b",
        "event_query",
        {"event": 1.5, "decision": 1.3, "fact": 0.9},
    ),
    (
        r"\b(decided|chose|switched|changed|moved|adopted)\b",
        "decision_query",
        {"decision": 1.6, "correction": 1.4, "goal": 1.1},
    ),
    (
        r"\b(wrong|correct|update|mistake|no longer|changed\s+my\s+mind)\b",
        "correction_query",
        {"correction": 1.8, "decision": 1.3, "preference": 1.1},
    ),
    (
        r"\b(know|remember|recall|tell\s+me\s+about\s+me|about\s+my)\b",
        "general_query",
        {"fact": 1.2, "skill": 1.1, "preference": 1.1, "goal": 1.1},
    ),
]

_BASE_TYPE_SCORES: dict[str, float] = {
    # Starting multiplier before intent is applied
    # Reflects inherent importance of memory type
    "correction":   0.80,
    "goal":         0.75,
    "decision":     0.72,
    "skill":        0.70,
    "preference":   0.68,
    "fact":         0.65,
    "relationship": 0.65,
    "event":        0.55,
    "generic":      0.45,
}


def _s3_type_resonance(
    query: str,
    chunk: MemoryChunk,
) -> tuple[float, str, bool]:
    """
    S3 = base_type_score × intent_multiplier

    Infers what type of memory the query is seeking from its vocabulary,
    then rewards memories of that type.

    Returns (s3_score, detected_intent, type_matched).
    """
    query_lower = query.lower()
    chunk_type  = chunk.memory_type.value

    base = _BASE_TYPE_SCORES.get(chunk_type, 0.5)
    best_multiplier = 1.0
    best_intent = "none"
    type_matched = False

    for pattern, intent_name, type_multipliers in _INTENT_RULES:
        if re.search(pattern, query_lower, re.IGNORECASE):
            mult = type_multipliers.get(chunk_type, 1.0)
            if mult > best_multiplier:
                best_multiplier = mult
                best_intent = intent_name
                type_matched = (mult > 1.0)

    s3 = min(1.0, base * best_multiplier)
    return round(s3, 4), best_intent, type_matched


# ─────────────────────────────────────────────────────────────────────────────
# S4 — Reinforcement history
# ─────────────────────────────────────────────────────────────────────────────

def _s4_reinforcement(chunk: MemoryChunk, now: float) -> float:
    """
    S4 = log(1 + access_count) × recency_factor × layer_weight

    access_count:   how many times has this been retrieved and used?
    recency_factor: was it accessed recently? (recent utility > old utility)
    layer_weight:   deeper layers have earned their place — slight bonus

    New memories start at S4 ≈ 0.2 (neutral, not penalised).
    Frequently retrieved memories can reach S4 ≈ 0.85.
    """
    # Log-compressed access count, normalised to [0, 1]
    # access_count=0 → 0.0, 5 → 0.70, 10 → 0.83, 20 → 0.93
    access_score = math.log1p(chunk.access_count) / math.log1p(20)
    access_score = min(1.0, access_score)

    # Recency of access: memories accessed in last 48h get a boost
    hours_since_access = (now - chunk.last_accessed) / 3600
    recency_factor = 0.4 + 0.6 * math.exp(-hours_since_access / 48)

    # Layer weight: identity > semantic > episodic > working
    layer_weights = {
        MemoryLayer.IDENTITY: 1.20,
        MemoryLayer.SEMANTIC: 1.10,
        MemoryLayer.EPISODIC: 1.00,
        MemoryLayer.WORKING:  0.85,
    }
    layer_w = layer_weights.get(chunk.layer, 1.0)

    # New memory base: ensures new memories aren't heavily penalised
    new_memory_base = 0.20

    s4 = new_memory_base + (1 - new_memory_base) * access_score * recency_factor * layer_w
    return round(min(1.0, s4), 4)


# ─────────────────────────────────────────────────────────────────────────────
# MMR diversifier
# ─────────────────────────────────────────────────────────────────────────────

def _mmr_select(
    candidates: list[tuple[MemoryChunk, float]],   # (chunk, resonance_score)
    top_k: int,
    mmr_lambda: float,                              # 0=diversity, 1=relevance
) -> list[tuple[MemoryChunk, float, float]]:       # (chunk, resonance, mmr_score)
    """
    Maximal Marginal Relevance selection.

    Iteratively picks the next candidate that maximises:
      mmr_score = λ × resonance(c) - (1-λ) × max_sim(c, selected)

    This penalises near-duplicate memories so the top-K is diverse.
    A memory about 'I use PyTorch' won't crowd out 'I use Grad-CAM'
    just because both are about tools.

    Returns list of (chunk, original_resonance, mmr_score).
    """
    if not candidates:
        return []

    selected: list[tuple[MemoryChunk, float, float]] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        best_mmr  = -float("inf")
        best_item = None

        for chunk, resonance in remaining:
            if not selected:
                mmr_score = resonance
            else:
                # Max similarity to any already-selected chunk
                max_sim = max(
                    _cosine(chunk.embedding or [], s[0].embedding or [])
                    for s in selected
                ) if any(s[0].embedding for s in selected) else 0.0

                mmr_score = mmr_lambda * resonance - (1 - mmr_lambda) * max_sim

            if mmr_score > best_mmr:
                best_mmr  = mmr_score
                best_item = (chunk, resonance, mmr_score)

        if best_item is None:
            break

        selected.append(best_item)
        remaining = [(c, r) for c, r in remaining if c.id != best_item[0].id]

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# CRR Retrieval Engine — public class
# ─────────────────────────────────────────────────────────────────────────────

class CRREngine:
    """
    Contextual Resonance Retrieval — drop-in replacement for RetrievalEngine.

    Usage:
        from cmg.crr import CRREngine, CRRWeights

        engine = CRREngine(store, embed_fn)

        results = engine.retrieve(
            query="What tools do I use for explainability?",
            query_embedding=embed_fn(query),
            top_k=8,
        )

        for r in results:
            print(r.chunk.content, r.combined_score)
            print(r.metadata["resonance"].summary())

    Tuning:
        engine = CRREngine(store, embed_fn,
                           weights=CRRWeights(alpha=0.5, gamma=0.30, mmr_lambda=0.6))

    The embed_fn is any callable: text -> list[float].
    It's used to embed source texts for S1 source-context matching.
    Pass None to skip source embedding (falls back to memory-only S1).
    """

    def __init__(
        self,
        store: VectorStore,
        embed_fn=None,                    # callable: str -> list[float], or None
        weights: Optional[CRRWeights] = None,
        min_resonance: float = 0.05,      # discard below this threshold
        source_cache_size: int = 512,     # LRU cache for source embeddings
    ):
        self._store           = store
        self._embed_fn        = embed_fn
        self._weights         = (weights or CRRWeights()).normalised()
        self._min_resonance   = min_resonance
        self._source_emb_cache: dict[str, list[float]] = {}
        self._source_cache_size = source_cache_size

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        top_k: int = 8,
        layer_filter: Optional[list[MemoryLayer]] = None,
        bump_access: bool = True,
        return_scores: bool = False,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-K chunks using Contextual Resonance Retrieval.

        Args:
            query:           The user's current message (text)
            query_embedding: Pre-computed embedding of query (computed if None)
            top_k:           Number of results to return after MMR
            layer_filter:    Restrict to specific layers (None = all)
            bump_access:     Reinforce retrieved chunks (bump access_count + score)
            return_scores:   If True, RetrievalResult.metadata["resonance"] = ResonanceScore

        Returns:
            List of RetrievalResult, sorted by mmr_score descending.
        """
        if self._store.count() == 0:
            return []

        # Compute query embedding if not provided
        if query_embedding is None:
            if self._embed_fn is None:
                return []
            try:
                query_embedding = self._embed_fn(query)
            except Exception:
                return []

        now = time.time()

        # Detect temporal frame from query text
        frame_name, recency_bias = _detect_temporal_frame(query)

        # Get candidate pool (over-fetch to allow MMR to have enough to work with)
        fetch_k = min(self._store.count(), top_k * 4)
        raw_candidates = self._store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            layer_filter=layer_filter,
        )

        if not raw_candidates:
            return []

        # Compute source embeddings for S1 (cached)
        source_embs = self._get_source_embeddings(
            [chunk for chunk, _ in raw_candidates]
        )

        # Score each candidate across all four signals
        scored: list[tuple[MemoryChunk, float, ResonanceScore]] = []

        for chunk, _ in raw_candidates:
            rs = self._score_chunk(
                chunk=chunk,
                query=query,
                query_emb=query_embedding,
                source_embs=source_embs,
                now=now,
                recency_bias=recency_bias,
                frame_name=frame_name,
            )
            if rs.resonance >= self._min_resonance:
                scored.append((chunk, rs.resonance, rs))

        if not scored:
            return []

        # Sort by resonance before MMR
        scored.sort(key=lambda x: x[1], reverse=True)
        candidates_for_mmr = [(chunk, resonance) for chunk, resonance, _ in scored]
        score_map = {chunk.id: rs for chunk, _, rs in scored}

        # MMR diversification
        mmr_results = _mmr_select(
            candidates=candidates_for_mmr,
            top_k=top_k,
            mmr_lambda=self._weights.mmr_lambda,
        )

        # Build output
        results: list[RetrievalResult] = []
        for chunk, resonance, mmr_score in mmr_results:
            rs = score_map[chunk.id]
            rs.mmr_score  = mmr_score
            rs.mmr_penalty = resonance - mmr_score

            result = RetrievalResult(
                chunk=chunk,
                relevance_score=rs.s1_semantic,
                combined_score=mmr_score,
                layer=chunk.layer,
            )
            if return_scores:
                result.chunk.metadata["_resonance"] = rs
            results.append(result)

            # Reinforce retrieved chunks
            if bump_access:
                chunk.bump_access()
                self._store.upsert(chunk)

        return results

    def build_memory_context(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        top_k: int = 8,
        format_style: str = "structured",
    ) -> Optional[str]:
        """
        Build a formatted memory context string for system prompt injection.
        Drop-in replacement for RetrievalEngine.build_memory_context().
        """
        results = self.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            bump_access=True,
        )
        if not results:
            return None

        TEMPLATE = """<memory_context>
The following is what you know about the user from previous conversations.
Use this to personalise your response. Do not explicitly mention this block.

{memories}
</memory_context>"""

        if format_style == "structured":
            by_type: dict[str, list[str]] = {}
            for r in results:
                t = r.chunk.memory_type.value
                by_type.setdefault(t, []).append(r.chunk.content)
            labels = {
                "fact": "Facts", "preference": "Preferences", "goal": "Goals",
                "skill": "Skills", "decision": "Decisions", "correction": "Corrections",
                "relationship": "Relationships", "event": "Events", "generic": "Other",
            }
            lines = []
            for mtype, contents in by_type.items():
                lines.append(f"[{labels.get(mtype, mtype)}]")
                for c in contents:
                    lines.append(f"  - {c}")
            body = "\n".join(lines)

        elif format_style == "prose":
            body = "\n".join(f"• {r.chunk.content}" for r in results)
        else:
            body = "; ".join(r.chunk.content[:100] for r in results)

        return TEMPLATE.format(memories=body)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score_chunk(
        self,
        chunk: MemoryChunk,
        query: str,
        query_emb: list[float],
        source_embs: dict[str, list[float]],
        now: float,
        recency_bias: float,
        frame_name: str,
    ) -> ResonanceScore:
        w = self._weights

        s1, mem_sim, src_sim = _s1_semantic(query_emb, chunk, source_embs)
        s2  = _s2_temporal(chunk, now, recency_bias)
        s3, intent, type_match = _s3_type_resonance(query, chunk)
        s4  = _s4_reinforcement(chunk, now)

        resonance = w.alpha * s1 + w.beta * s2 + w.gamma * s3 + w.delta * s4

        return ResonanceScore(
            chunk_id        = chunk.id,
            s1_semantic     = s1,
            s2_temporal     = s2,
            s3_type         = s3,
            s4_reinforcement= s4,
            resonance       = round(resonance, 4),
            s1_memory_sim   = mem_sim,
            s1_source_sim   = src_sim,
            s3_detected_intent = f"{intent}({frame_name})",
            s3_type_match   = type_match,
        )

    def _get_source_embeddings(
        self, chunks: list[MemoryChunk]
    ) -> dict[str, list[float]]:
        """
        Compute (and cache) source text embeddings for S1.
        Only runs if embed_fn is available.
        """
        if self._embed_fn is None:
            return {}

        result = {}
        for chunk in chunks:
            if not chunk.source_text:
                continue
            cid = chunk.id
            if cid in self._source_emb_cache:
                result[cid] = self._source_emb_cache[cid]
            else:
                try:
                    emb = self._embed_fn(chunk.source_text[:512])
                    # LRU eviction: drop oldest if cache full
                    if len(self._source_emb_cache) >= self._source_cache_size:
                        oldest = next(iter(self._source_emb_cache))
                        del self._source_emb_cache[oldest]
                    self._source_emb_cache[cid] = emb
                    result[cid] = emb
                except Exception:
                    pass
        return result

    def explain(self, query: str, query_embedding: list[float], top_k: int = 5) -> str:
        """
        Human-readable explanation of why each retrieved chunk was ranked where it was.
        Useful for debugging retrieval quality.
        """
        results = self.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            bump_access=False,
            return_scores=True,
        )
        lines = [f"CRR Retrieval explanation for: \"{query[:80]}\"",
                 f"Temporal frame detected: {_detect_temporal_frame(query)}",
                 "─" * 60]
        for i, r in enumerate(results, 1):
            rs: ResonanceScore = r.chunk.metadata.get("_resonance")
            lines.append(f"[{i}] {r.chunk.content[:70]}")
            lines.append(f"     layer={r.layer.value} type={r.chunk.memory_type.value}")
            if rs:
                lines.append(f"     {rs.summary()}")
                lines.append(f"     S1: mem_sim={rs.s1_memory_sim:.3f} src_sim={rs.s1_source_sim:.3f}")
                lines.append(f"     S3: intent={rs.s3_detected_intent} match={rs.s3_type_match}")
                if rs.mmr_penalty > 0.01:
                    lines.append(f"     MMR: penalised by {rs.mmr_penalty:.3f} for similarity to higher-ranked result")
        return "\n".join(lines)