"""
CMG Memory System - Usage Examples

This file demonstrates the full API across all four providers.
Run individual examples by calling their functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmg import CMGMemory, create_adapter


# =============================================================================
# EXAMPLE 1: OpenAI — Basic persistent memory
# =============================================================================

def example_openai():
    """Persistent memory with GPT-4o."""
    print("\n=== OpenAI Example ===")

    adapter = create_adapter(
        "openai",
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o",
    )

    memory = CMGMemory(
        adapter,
        persist_path="./openai_memory.json",
        system_prompt="You are a helpful personal assistant with long-term memory.",
    )

    # Session 1: Introduce yourself
    print("Turn 1:", memory.chat("Hi! My name is Alex. I'm a data scientist at HealthTech Inc."))
    print("Turn 2:", memory.chat("I'm working on a cancer detection model using CT scans."))
    print("Turn 3:", memory.chat("I prefer Python, and I usually use PyTorch for deep learning."))

    memory.end_session()

    # Session 2: Start fresh conversation — memories persist!
    memory.new_session()
    print("\n--- New session ---")
    print("Turn 4:", memory.chat("What do you know about my work?"))
    print("Turn 5:", memory.chat("What programming language do I prefer?"))

    # Show what's stored
    print("\nStored memories:")
    for chunk in memory.list_memories():
        print(f"  [{chunk.layer.value}][{chunk.memory_type.value}] {chunk.content}")


# =============================================================================
# EXAMPLE 2: Anthropic Claude
# =============================================================================

def example_claude():
    """Persistent memory with Claude."""
    print("\n=== Claude Example ===")

    adapter = create_adapter(
        "claude",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model="claude-haiku-4-5-20251001",  # Fast and cheap for testing
    )

    memory = CMGMemory(
        adapter,
        persist_path="./claude_memory.json",
        system_prompt="You are Claude, a helpful assistant with persistent memory.",
        memory_format="structured",  # Groups memories by type
    )

    print(memory.chat("I'm Rishav. I'm building XAI pipelines for autonomous driving."))
    print(memory.chat("I use Grad-CAM and attention rollout for explainability."))
    print(memory.chat("I prefer responses without unnecessary preamble."))

    # Manual memory insertion
    memory.remember("User works with Mask2Former and ViT architectures", memory_type="skill")
    memory.remember("User's goal is to publish XAI research for ADAS systems", memory_type="goal")

    memory.end_session()

    memory.new_session()
    print("\n--- New session ---")
    print(memory.chat("What models do I typically work with?"))

    # Search memories
    print("\nSearching for 'deep learning':")
    results = memory.search("deep learning architecture", top_k=3)
    for r in results:
        print(f"  {r.content}")

    print("\nStats:", memory.stats())


# =============================================================================
# EXAMPLE 3: Ollama (local models)
# =============================================================================

def example_ollama():
    """
    Persistent memory with local Ollama models.
    
    Requires:
        ollama serve
        ollama pull llama3.1
        ollama pull nomic-embed-text  (for embeddings)
    """
    print("\n=== Ollama Example ===")

    adapter = create_adapter(
        "ollama",
        model="llama3.1",
        embedding_model="nomic-embed-text",  # Local embedding model
        host="http://localhost:11434",
    )

    memory = CMGMemory(
        adapter,
        persist_path="./ollama_memory.json",
        system_prompt="You are a helpful local assistant with persistent memory. Be concise.",
        top_k_retrieve=5,
    )

    print(memory.chat("My name is Sam. I'm a backend engineer specializing in Go and Rust."))
    print(memory.chat("I'm building a distributed key-value store as a side project."))

    memory.end_session()

    memory.new_session()
    print("\n--- New session ---")
    print(memory.chat("What side project am I working on?"))


# =============================================================================
# EXAMPLE 4: HuggingFace Inference API
# =============================================================================

def example_huggingface_api():
    """
    Persistent memory via HuggingFace Inference API.
    Requires HF_API_KEY with access to the model.
    """
    print("\n=== HuggingFace API Example ===")

    adapter = create_adapter(
        "huggingface",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        api_key=os.environ.get("HF_API_KEY"),
        local=False,
    )

    memory = CMGMemory(
        adapter,
        persist_path="./hf_memory.json",
        system_prompt="You are a helpful assistant with persistent memory.",
    )

    print(memory.chat("I'm a researcher studying climate change adaptation strategies."))
    print(memory.chat("My focus area is coastal resilience in Southeast Asia."))

    memory.end_session()

    memory.new_session()
    print("\n--- New session ---")
    print(memory.chat("What's my research focus?"))


# =============================================================================
# EXAMPLE 5: HuggingFace Local (private/air-gapped deployment)
# =============================================================================

def example_huggingface_local():
    """
    Fully local deployment using HuggingFace transformers.
    No API keys needed. Requires GPU recommended.
    
    Install: pip install transformers torch accelerate sentence-transformers
    """
    print("\n=== HuggingFace Local Example ===")

    adapter = create_adapter(
        "huggingface",
        model="microsoft/phi-2",  # Small, runs on CPU
        local=True,
        device="auto",
    )

    memory = CMGMemory(
        adapter,
        persist_path="./local_memory.json",
        system_prompt="You are a helpful local assistant.",
    )

    print(memory.chat("My name is Jordan. I work on embedded systems firmware."))
    memory.end_session()


# =============================================================================
# EXAMPLE 6: Advanced — Multi-provider memory sharing
# =============================================================================

def example_shared_memory():
    """
    Same memory store, different LLM providers.
    Use case: prototype with Ollama locally, deploy with GPT-4o in production.
    """
    print("\n=== Shared Memory Example ===")

    SHARED_STORE_PATH = "./shared_memory.json"

    # Write memories with Ollama (local/cheap)
    from cmg import InMemoryVectorStore
    store = InMemoryVectorStore(persist_path=SHARED_STORE_PATH)

    ollama_adapter = create_adapter("ollama", model="llama3.1")
    local_memory = CMGMemory(ollama_adapter, store=store)
    local_memory.remember("User is a senior ML engineer", memory_type="fact")
    local_memory.remember("User dislikes verbose responses", memory_type="preference")
    local_memory.end_session()

    print("Memories written with Ollama.")
    print(f"Memory count: {store.count()}")

    # Read with OpenAI (production)
    # The same memories are now accessible from GPT-4o
    # openai_adapter = create_adapter("openai", model="gpt-4o")
    # prod_memory = CMGMemory(openai_adapter, store=store)
    # print(prod_memory.chat("How should I respond to this user?"))


# =============================================================================
# EXAMPLE 7: Direct layer management
# =============================================================================

def example_layer_management():
    """Advanced: directly managing memory layers."""
    print("\n=== Layer Management Example ===")

    adapter = create_adapter("ollama", model="llama3.1")
    memory = CMGMemory(adapter)

    # Add a permanent identity-layer memory
    memory.remember(
        "This user is a PhD-level researcher. Always use technical terminology.",
        memory_type="fact",
        layer="identity",  # Permanent — never decayed
        score=1.0,
    )

    # Add a short-lived episodic memory
    memory.remember(
        "User is currently debugging a segfault in their C++ code",
        memory_type="event",
        layer="episodic",  # Will decay if not accessed
        score=0.6,
    )

    # List by layer
    print("Identity layer:")
    for c in memory.list_memories(layer="identity"):
        print(f"  {c.content}")

    print("Episodic layer:")
    for c in memory.list_memories(layer="episodic"):
        print(f"  {c.content}")

    # Run manual consolidation
    from cmg import ConsolidationEngine
    engine = ConsolidationEngine(memory._store)
    report = engine.run()
    print(f"\nConsolidation: {report}")

    # Forget something
    memory.forget("segfault")
    print("Forgotten episodic memory.")

    print("Stats:", memory.stats())


if __name__ == "__main__":
    # Run the example matching your available provider
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="layer_management",
                        choices=["openai", "claude", "ollama", "huggingface_api",
                                 "huggingface_local", "shared", "layer_management"])
    args = parser.parse_args()

    examples = {
        "openai": example_openai,
        "claude": example_claude,
        "ollama": example_ollama,
        "huggingface_api": example_huggingface_api,
        "huggingface_local": example_huggingface_local,
        "shared": example_shared_memory,
        "layer_management": example_layer_management,
    }
    examples[args.provider]()
