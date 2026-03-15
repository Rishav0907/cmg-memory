"""
CMG Memory System
=================
Contextual Memory Gradient: biologically-inspired persistent memory for LLMs.

Supports: OpenAI · Anthropic Claude · HuggingFace · Ollama

Quick start:
    from cmg import CMGMemory, create_adapter

    adapter = create_adapter("openai", api_key="sk-...", model="gpt-4o")
    memory = CMGMemory(adapter, persist_path="./memory.json")

    response = memory.chat("My name is Rishav, I work on autonomous driving XAI")
    response = memory.chat("Summarize what you know about me")  # Uses memory!

    memory.end_session()  # Consolidates memory (decay, merge, promote)
"""

from .memory import CMGMemory
from .adapters import (
    LLMAdapter,
    OpenAIAdapter,
    ClaudeAdapter,
    HuggingFaceAdapter,
    OllamaAdapter,
    create_adapter,
)
from .types import (
    MemoryChunk,
    MemoryLayer,
    MemoryType,
    RetrievalResult,
    ConsolidationReport,
)
from .store import InMemoryVectorStore, VectorStore, PineconeVectorStore
from .retrieval import RetrievalEngine
from .consolidation import ConsolidationEngine
from .extractor import MemoryExtractor, ScoringResult
from .scorer import DeterministicScorer, ScorerWeights, FeatureVector
from .crr import CRREngine, CRRWeights, ResonanceScore
from .write_gate import WriteGate, WriteGateStats, GateResult, WriteAction, DEFAULT_LAYER_CAPS

__version__ = "0.1.0"
__all__ = [
    "CMGMemory",
    "create_adapter",
    "LLMAdapter",
    "OpenAIAdapter",
    "ClaudeAdapter",
    "HuggingFaceAdapter",
    "OllamaAdapter",
    "MemoryChunk",
    "MemoryLayer",
    "MemoryType",
    "RetrievalResult",
    "ConsolidationReport",
    "InMemoryVectorStore",
    "VectorStore",
    "PineconeVectorStore",
    "RetrievalEngine",
    "ConsolidationEngine",
    "MemoryExtractor",
    "ScoringResult",
    "DeterministicScorer",
    "ScorerWeights",
    "FeatureVector",
    "CRREngine",
    "CRRWeights",
    "ResonanceScore",
    "WriteGate",
    "WriteGateStats",
    "GateResult",
    "WriteAction",
    "DEFAULT_LAYER_CAPS",
]

# Hosted API client
from .client import CMGClient, CMGAsyncClient, Memory, ChatResult, Stats