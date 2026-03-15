"""
CMG Memory System - LLM Provider Adapters

Unified interface for:
  - OpenAI (GPT-4o, GPT-4, GPT-3.5)
  - Anthropic Claude (claude-3-5-sonnet, claude-3-opus, etc.)
  - HuggingFace (local or inference API)
  - Ollama (local models: llama3, mistral, gemma, etc.)

All adapters implement two methods:
  - chat(messages, system) -> str          : generate a response
  - embed(text) -> list[float]             : produce an embedding vector

For providers without a native embedding API (Claude, Ollama),
we fall back to a local sentence-transformers model.
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class LLMAdapter(ABC):
    """
    Implement this to add a new provider.
    
    chat()  → text completion
    embed() → embedding vector (for memory similarity search)
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str: ...

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Fallback local embedder (used when provider has no embedding endpoint)
# ---------------------------------------------------------------------------

class _LocalEmbedder:
    """
    Uses sentence-transformers for local embeddings.
    Install: pip install sentence-transformers
    Falls back to a naive hash-based pseudo-embedding if not available.
    """
    _model = None

    @classmethod
    def embed(cls, text: str) -> list[float]:
        if cls._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                cls._model = "fallback"

        if cls._model == "fallback":
            return cls._hash_embed(text)

        vec = cls._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> list[float]:
        """Deterministic pseudo-embedding via hashing (low quality, no deps)."""
        import hashlib, struct
        seed = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(0, dim * 4, 4):
            idx = i % len(seed)
            val = struct.unpack_from("f", bytes([seed[idx], seed[(idx+1)%32],
                                                  seed[(idx+2)%32], seed[(idx+3)%32]]))[0]
            vec.append(val)
        norm = sum(v*v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIAdapter(LLMAdapter):
    """
    OpenAI GPT adapter with native text-embedding-3-small support.
    
    Usage:
        adapter = OpenAIAdapter(api_key="sk-...", model="gpt-4o")
    
    Install: pip install openai
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,   # For Azure OpenAI or compatible APIs
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self._model = model
        self._embedding_model = embedding_model
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._embedding_model,
            input=text[:8000],  # Truncate to model limit
        )
        return resp.data[0].embedding

    def name(self) -> str:
        return f"OpenAI({self._model})"


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class ClaudeAdapter(LLMAdapter):
    """
    Anthropic Claude adapter.
    Embeddings fall back to local sentence-transformers (Claude has no embedding API).
    
    Usage:
        adapter = ClaudeAdapter(api_key="sk-ant-...", model="claude-sonnet-4-5")
    
    Install: pip install anthropic
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        self._model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text

    def embed(self, text: str) -> list[float]:
        # Claude doesn't have an embedding API — use local model
        return _LocalEmbedder.embed(text)

    def name(self) -> str:
        return f"Claude({self._model})"


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

class HuggingFaceAdapter(LLMAdapter):
    """
    HuggingFace adapter — supports both:
      1. Local models via transformers pipeline
      2. HuggingFace Inference API (hosted)
    
    For local:
        adapter = HuggingFaceAdapter(model="mistralai/Mistral-7B-Instruct-v0.2", local=True)
    
    For Inference API:
        adapter = HuggingFaceAdapter(model="mistralai/Mistral-7B-Instruct-v0.2", api_key="hf_...")
    
    Install: pip install transformers torch   (for local)
             pip install huggingface_hub      (for API)
    """

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        local: bool = False,
        device: str = "auto",
    ):
        self._model_name = model
        self._embedding_model_name = embedding_model
        self._local = local
        self._pipeline = None
        self._embed_pipeline = None

        if local:
            self._init_local(model, embedding_model, device)
        else:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(
                    model=model,
                    token=api_key or os.environ.get("HF_API_KEY"),
                )
            except ImportError:
                raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    def _init_local(self, model: str, embedding_model: str, device: str) -> None:
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            device_map=device,
            trust_remote_code=True,
        )
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_pipeline = SentenceTransformer(embedding_model)
        except ImportError:
            self._embed_pipeline = None

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        if system:
            messages = [{"role": "system", "content": system}] + messages

        if self._local and self._pipeline:
            result = self._pipeline(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
            return result[0]["generated_text"][-1]["content"]
        else:
            # HuggingFace Inference API
            resp = self._client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content

    def embed(self, text: str) -> list[float]:
        if self._embed_pipeline:
            vec = self._embed_pipeline.encode(text, normalize_embeddings=True)
            return vec.tolist()
        return _LocalEmbedder.embed(text)

    def name(self) -> str:
        return f"HuggingFace({self._model_name})"


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaAdapter(LLMAdapter):
    """
    Ollama local model adapter.
    Supports any model pulled via `ollama pull <model>`.
    
    Usage:
        adapter = OllamaAdapter(model="llama3.2", embedding_model="nomic-embed-text")
    
    Requires Ollama running locally: https://ollama.com
    Install client: pip install ollama
    
    Recommended embedding model: ollama pull nomic-embed-text
    Falls back to local sentence-transformers if embedding model not available.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")

        self._model = model
        self._embedding_model = embedding_model
        self._host = host
        self._client = _ollama.Client(host=host)

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        if system:
            messages = [{"role": "system", "content": system}] + messages

        resp = self._client.chat(
            model=self._model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return resp["message"]["content"]

    def embed(self, text: str) -> list[float]:
        try:
            resp = self._client.embeddings(
                model=self._embedding_model,
                prompt=text,
            )
            return resp["embedding"]
        except Exception:
            # Fallback if embedding model not pulled
            return _LocalEmbedder.embed(text)

    def name(self) -> str:
        return f"Ollama({self._model})"


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_adapter(provider: str, **kwargs) -> LLMAdapter:
    """
    Convenience factory.
    
    Examples:
        adapter = create_adapter("openai", model="gpt-4o", api_key="sk-...")
        adapter = create_adapter("claude", model="claude-sonnet-4-5")
        adapter = create_adapter("ollama", model="llama3.2")
        adapter = create_adapter("huggingface", model="mistralai/Mistral-7B-Instruct-v0.2")
    """
    registry = {
        "openai": OpenAIAdapter,
        "claude": ClaudeAdapter,
        "anthropic": ClaudeAdapter,
        "huggingface": HuggingFaceAdapter,
        "hf": HuggingFaceAdapter,
        "ollama": OllamaAdapter,
    }
    provider_lower = provider.lower()
    if provider_lower not in registry:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(registry.keys())}")
    return registry[provider_lower](**kwargs)