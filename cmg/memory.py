"""
CMG Memory System - Main Interface

CMGMemory is the top-level object you integrate with.
It wraps the store, extractor, retrieval engine, and consolidation engine
behind a clean API that works identically regardless of which LLM you use.

Quick start:
    from cmg import CMGMemory, create_adapter

    adapter = create_adapter("openai", api_key="sk-...", model="gpt-4o")
    # or: create_adapter("claude", model="claude-sonnet-4-5")
    # or: create_adapter("ollama", model="llama3.2")
    # or: create_adapter("huggingface", model="mistralai/Mistral-7B-Instruct-v0.2")

    memory = CMGMemory(adapter, persist_path="./my_memory.json")

    # Drop-in chat with automatic memory
    response = memory.chat("My name is Rishav and I work on autonomous driving XAI")
    response = memory.chat("What do you know about my work?")  # Remembers!

    # End session — triggers consolidation
    memory.end_session()
"""

from __future__ import annotations
import logging
import time
import uuid
from typing import Optional

from .types import MemoryChunk, MemoryLayer, MemoryType, ConsolidationReport
from .store import InMemoryVectorStore, VectorStore
from .adapters import LLMAdapter, create_adapter
from .extractor import MemoryExtractor
from .retrieval import RetrievalEngine
from .consolidation import ConsolidationEngine
from .crr import CRREngine, CRRWeights
from .write_gate import WriteGate, DEFAULT_LAYER_CAPS

logger = logging.getLogger("cmg")


class CMGMemory:
    """
    Contextual Memory Gradient — persistent memory for any LLM.

    Integrates with: OpenAI, Anthropic Claude, HuggingFace, Ollama.

    Architecture:
        User message
            ↓
        [Retrieval] — fetch relevant memories from store
            ↓
        [Augmented prompt] — inject memories into system prompt
            ↓
        [LLM call] — your provider, your model
            ↓
        [Extraction] — pull new memories from this turn
            ↓
        [Store] — persist new memories to episodic buffer
            ↓
        [Consolidation] — background: decay, merge, promote, forget
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        store: Optional[VectorStore] = None,
        persist_path: Optional[str] = None,
        user_id: str = "default",
        system_prompt: Optional[str] = None,
        top_k_retrieve: int = 8,
        memory_format: str = "structured",
        auto_extract: bool = True,
        auto_consolidate_after: int = 10,  # Consolidate every N turns
        log_level: int = logging.WARNING,
    ):
        """
        Args:
            llm_adapter: Provider adapter (OpenAIAdapter, ClaudeAdapter, etc.)
            store: Custom vector store (default: InMemoryVectorStore with JSON persistence)
            persist_path: File path for JSON persistence (e.g., "./memory.json")
            user_id: Identifier for this user's memory namespace
            system_prompt: Base system prompt (memory context is prepended to this)
            top_k_retrieve: Number of memories to inject per turn
            memory_format: "structured" | "prose" | "minimal"
            auto_extract: Automatically extract memories after each turn
            auto_consolidate_after: Run consolidation every N turns
            log_level: Logging verbosity
        """
        logging.basicConfig(level=log_level)

        self._llm = llm_adapter
        self._user_id = user_id
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._top_k = top_k_retrieve
        self._memory_format = memory_format
        self._auto_extract = auto_extract
        self._auto_consolidate_after = auto_consolidate_after

        # Store
        self._store = store or InMemoryVectorStore(persist_path=persist_path)

        # Sub-systems
        self._extractor = MemoryExtractor(llm_adapter)
        # CRR is the default retriever — model-agnostic, noise-resistant
        # Falls back to RetrievalEngine if embed_fn is unavailable
        try:
            self._retriever = CRREngine(
                store=self._store,
                embed_fn=llm_adapter.embed,
            )
        except Exception:
            self._retriever = RetrievalEngine(self._store, llm_adapter)
        self._consolidator = ConsolidationEngine(self._store)
        # Write gate — prevents redundant/low-value writes to the store
        self._write_gate = WriteGate(
            store             = self._store,
            embed_fn          = self._llm.embed,
            novelty_threshold = 0.88,
            min_score         = 0.35,
        )

        # Session state
        self._session_id = str(uuid.uuid4())
        self._conversation: list[dict] = []
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Main chat interface
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        system_override: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        skip_memory: bool = False,
    ) -> str:
        """
        Send a message and get a response, with automatic memory integration.

        Memory is retrieved before the LLM call and extracted after.

        Args:
            user_message: The user's input
            system_override: Override the system prompt for this turn only
            temperature: LLM temperature
            max_tokens: Max response tokens
            skip_memory: Disable memory for this turn (useful for meta-queries)

        Returns:
            The assistant's response string
        """
        # 1. Build system prompt with memory context
        if skip_memory:
            system = system_override or self._system_prompt
        else:
            memory_context = self._retriever.build_memory_context(
                user_message,
                top_k=self._top_k,
                format_style=self._memory_format,
            )
            base_system = system_override or self._system_prompt
            system = f"{memory_context}\n\n{base_system}" if memory_context else base_system

        # 2. Add user message to conversation
        self._conversation.append({"role": "user", "content": user_message})

        # 3. LLM call
        response = self._llm.chat(
            messages=self._conversation.copy(),
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 4. Add response to conversation
        self._conversation.append({"role": "assistant", "content": response})
        self._turn_count += 1

        # 5. Extract memories from this turn (async-friendly: does not block)
        if self._auto_extract and not skip_memory:
            self._extract_and_store(self._conversation[-4:])

        # 6. Periodic consolidation
        if self._turn_count % self._auto_consolidate_after == 0:
            logger.info(f"Running periodic consolidation at turn {self._turn_count}")
            self._consolidator.run()

        return response

    # ------------------------------------------------------------------
    # Manual memory management
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        layer: str = "episodic",
        score: float = 0.8,
        tags: Optional[list[str]] = None,
    ) -> MemoryChunk:
        """
        Manually add a memory.

        Example:
            memory.remember("User prefers dark mode", memory_type="preference")
            memory.remember("User's name is Rishav", memory_type="fact", layer="semantic")
        """
        chunk = MemoryChunk(
            content=content,
            layer=MemoryLayer(layer),
            memory_type=MemoryType(memory_type),
            score=score,
            session_id=self._session_id,
            tags=tags or [],
        )
        embedding = self._llm.embed(content)
        chunk.embedding = embedding
        self._store.upsert(chunk)
        logger.info(f"Manual memory added: [{memory_type}] {content[:60]}")
        return chunk

    def forget(self, content_or_id: str) -> bool:
        """
        Delete a memory by ID or by partial content match.

        Example:
            memory.forget("prefers dark mode")
            memory.forget("chunk-uuid-here")
        """
        # Try direct ID lookup
        chunk = self._store.get(content_or_id)
        if chunk:
            self._store.delete(content_or_id)
            return True

        # Content-based search
        matches = [
            c for c in self._store.all_chunks()
            if content_or_id.lower() in c.content.lower()
        ]
        if matches:
            for m in matches:
                self._store.delete(m.id)
            logger.info(f"Forgot {len(matches)} memory/memories matching: {content_or_id}")
            return True
        return False

    def search(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        """Search memories by semantic similarity."""
        results = self._retriever.retrieve(query, top_k=top_k, bump_access=False)
        return [r.chunk for r in results]

    def list_memories(
        self,
        layer: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[MemoryChunk]:
        """List all stored memories, optionally filtered."""
        chunks = self._store.all_chunks()
        if layer:
            chunks = [c for c in chunks if c.layer.value == layer]
        if memory_type:
            chunks = [c for c in chunks if c.memory_type.value == memory_type]
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    def stats(self) -> dict:
        """Return memory statistics."""
        all_chunks = self._store.all_chunks()
        by_layer = {}
        by_type = {}
        for c in all_chunks:
            by_layer[c.layer.value] = by_layer.get(c.layer.value, 0) + 1
            by_type[c.memory_type.value] = by_type.get(c.memory_type.value, 0) + 1
        return {
            "total_memories": len(all_chunks),
            "by_layer": by_layer,
            "by_type": by_type,
            "session_turns": self._turn_count,
            "session_id": self._session_id,
            "provider": self._llm.name(),
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def end_session(self, run_consolidation: bool = True) -> Optional[ConsolidationReport]:
        """
        Call at the end of a session.
        Promotes working memory to episodic buffer and runs consolidation.
        """
        # Move working memory chunks to episodic
        for chunk in self._store.all_chunks():
            if chunk.layer == MemoryLayer.WORKING:
                chunk.layer = MemoryLayer.EPISODIC
                self._store.upsert(chunk)

        if run_consolidation:
            report = self._consolidator.run(llm_adapter=self._llm)
            logger.info(f"Session ended. Consolidation: {report}")
            return report

        # Reset session state
        self._session_id = str(uuid.uuid4())
        self._conversation = []
        return None

    def new_session(self) -> None:
        """Start a fresh session (conversation reset, memories persist)."""
        self._session_id = str(uuid.uuid4())
        self._conversation = []
        self._turn_count = 0

    def clear_conversation(self) -> None:
        """Clear current conversation context (keeps memories)."""
        self._conversation = []

    # ------------------------------------------------------------------
    # Direct LLM access (bypasses memory entirely)
    # ------------------------------------------------------------------

    def raw_chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Direct LLM call without memory integration."""
        return self._llm.chat(messages, system=system, **kwargs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_and_store(self, conversation_slice: list[dict]) -> None:
        """
        Extract memories from the latest conversation slice and pass them
        through the WriteGate before storing.

        Gate 1 (novelty): near-duplicates reinforce existing chunks instead
        Gate 2 (score):   low-value candidates are discarded
        Gate 3 (capacity): weakest chunk evicted when layer is at cap
        """
        try:
            chunks = self._extractor.extract(
                conversation_slice,
                session_id=self._session_id,
                existing_chunks=self._store.all_chunks(),
            )
            if not chunks:
                return

            # Embed all candidates (needed for Gate 1 novelty check)
            for chunk in chunks:
                if chunk.embedding is None:
                    try:
                        chunk.embedding = self._llm.embed(chunk.content)
                    except Exception:
                        pass

            # Initialise sessions_seen on new chunks before writing
            for chunk in chunks:
                if 'sessions_seen' not in chunk.metadata:
                    chunk.metadata['sessions_seen'] = [self._session_id] if self._session_id else []

            written, reinforced, discarded = self._write_gate.process(chunks)

            if written or reinforced or discarded:
                logger.debug(
                    "WriteGate: extracted=%d written=%d reinforced=%d discarded=%d",
                    len(chunks), len(written), len(reinforced), len(discarded),
                )
        except Exception as e:
            logger.warning("Memory extraction failed silently: %s", e)