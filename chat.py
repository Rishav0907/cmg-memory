import sys
sys.path.insert(0, ".")

from cmg import CMGMemory, create_adapter

adapter = create_adapter(
    "ollama",
    model="llama3.1",
    embedding_model="nomic-embed-text",
)

memory = CMGMemory(
    adapter,
    persist_path="./my_memory.json",   # memories survive across runs
    system_prompt="You are a helpful assistant with persistent memory.",
    log_level=20,                       # set to 30 to silence extraction logs
)

print(f"CMG Memory ready. Using: {adapter.name()}")
print("Type 'quit' to exit, 'stats' to see memory stats, 'memories' to list all stored memories.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "stats":
        import json
        print(json.dumps(memory.stats(), indent=2))
        continue
    if user_input.lower() == "memories":
        chunks = memory.list_memories()
        if not chunks:
            print("  (no memories stored yet)")
        for c in chunks:
            scoring = c.metadata.get("scoring", {})
            score_str = f"  [{c.layer.value}][{c.memory_type.value}] score={c.score:.2f}"
            if "summary" in scoring:
                score_str += f"\n    └─ {scoring['summary']}"
            print(f"{score_str}\n    {c.content}")
        continue

    response = memory.chat(user_input)
    print(f"\nAssistant: {response}\n")

memory.end_session()
print("Session ended. Memories consolidated and saved.")