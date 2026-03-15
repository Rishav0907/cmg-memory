"""
CMG Memory System - Core Types
Contextual Memory Gradient: A biologically-inspired persistent memory layer for LLMs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import uuid


class MemoryLayer(Enum):
    """The four gradient layers, ordered from most volatile to most permanent."""
    WORKING = "working"       # Current context window - full fidelity, session-scoped
    EPISODIC = "episodic"     # Key events, decisions, outcomes - lightly compressed
    SEMANTIC = "semantic"     # Distilled facts, preferences, patterns - dense
    IDENTITY = "identity"     # Goals, values, persistent persona - near-permanent


class MemoryType(Enum):
    """Semantic classification of a memory chunk."""
    FACT = "fact"                   # Declarative facts (user's name, job, location)
    PREFERENCE = "preference"       # Likes/dislikes, style choices
    DECISION = "decision"           # A choice made or action taken
    CORRECTION = "correction"       # Explicit correction of old info
    GOAL = "goal"                   # Ongoing objective
    SKILL = "skill"                 # Known capability or expertise
    RELATIONSHIP = "relationship"   # Social/professional connections
    EVENT = "event"                 # Something that happened
    GENERIC = "generic"             # Unclassified


@dataclass
class MemoryChunk:
    """
    A single unit of stored memory.
    
    Moves through layers (WORKING → EPISODIC → SEMANTIC → IDENTITY)
    as it accrues reinforcement, or decays and is forgotten.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""                          # The actual text content
    layer: MemoryLayer = MemoryLayer.EPISODIC
    memory_type: MemoryType = MemoryType.GENERIC
    score: float = 1.0                         # Salience score [0, 1]
    access_count: int = 0                      # Number of times retrieved
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    session_id: Optional[str] = None           # Which session created this
    source_text: Optional[str] = None          # Original text it was extracted from
    tags: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None    # Cached embedding vector
    superseded_by: Optional[str] = None        # ID of newer chunk that replaces this
    confidence: float = 1.0                    # How confident we are in this fact
    version: int = 1                           # Increments on update
    metadata: dict = field(default_factory=dict)

    def bump_access(self) -> None:
        """Call when this chunk is retrieved — reinforces it."""
        self.access_count += 1
        self.last_accessed = time.time()
        # Reinforcement: each access boosts score slightly (caps at 1.0)
        self.score = min(1.0, self.score + 0.05)

    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    def hours_since_access(self) -> float:
        return (time.time() - self.last_accessed) / 3600

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "layer": self.layer.value,
            "memory_type": self.memory_type.value,
            "score": self.score,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "last_updated": self.last_updated,
            "session_id": self.session_id,
            "source_text": self.source_text,
            "tags": self.tags,
            "confidence": self.confidence,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryChunk":
        chunk = cls(
            id=d["id"],
            content=d["content"],
            layer=MemoryLayer(d["layer"]),
            memory_type=MemoryType(d.get("memory_type", "generic")),
            score=d.get("score", 1.0),
            access_count=d.get("access_count", 0),
            created_at=d.get("created_at", time.time()),
            last_accessed=d.get("last_accessed", time.time()),
            last_updated=d.get("last_updated", time.time()),
            session_id=d.get("session_id"),
            source_text=d.get("source_text"),
            tags=d.get("tags", []),
            confidence=d.get("confidence", 1.0),
            version=d.get("version", 1),
            metadata=d.get("metadata", {}),
        )
        return chunk


@dataclass
class RetrievalResult:
    """A retrieved memory chunk with its relevance score for the current query."""
    chunk: MemoryChunk
    relevance_score: float      # Similarity to query
    combined_score: float       # relevance × salience × recency
    layer: MemoryLayer

    def __repr__(self):
        return f"RetrievalResult(layer={self.layer.value}, score={self.combined_score:.3f}, content={self.chunk.content[:60]}...)"


@dataclass
class ConsolidationReport:
    """Summary of what happened during a consolidation pass."""
    promoted: int = 0       # Chunks moved to a deeper layer
    merged: int = 0         # Duplicate chunks merged
    decayed: int = 0        # Chunks whose score dropped
    forgotten: int = 0      # Chunks removed (score too low)
    contradictions: int = 0 # Conflicting facts detected and resolved
    timestamp: float = field(default_factory=time.time)