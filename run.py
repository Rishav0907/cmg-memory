"""
CMG Memory System — Runner
===========================
Usage:
  python run.py                              interactive chat, JSON store (default)
  python run.py --store pinecone             interactive chat, Pinecone store
  python run.py --store json --file my.json  custom JSON file path
  python run.py --mode demo                  automated demo
  python run.py --mode inspect               browse stored memories
  python run.py --mode consolidate           run decay/merge pass now
  python run.py --mode reset                 wipe memory and start fresh
  python run.py --mode models                list available Ollama models
  python run.py --model mistral              use a different Ollama model
  python run.py --store pinecone --mode inspect  inspect Pinecone memories

Environment variables:
  PINECONE_API_KEY       required when --store pinecone
  PINECONE_INDEX_NAME    optional, default: cmg-memory
  PINECONE_NAMESPACE     optional, default: cmg-default
  OLLAMA_MODEL           override default model (llama3.1)
  OLLAMA_EMBED_MODEL     override embed model (nomic-embed-text)
  OLLAMA_HOST            override Ollama host (http://localhost:11434)

Install:
  pip install ollama sentence-transformers                  (JSON store)
  pip install ollama sentence-transformers pinecone-client  (Pinecone store)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Colour helpers ──────────────────────────────────────────────────────────────
RESET = "\033[0m";  BOLD  = "\033[1m"
GREEN = "\033[32m"; YELLOW = "\033[33m"
RED   = "\033[31m"; CYAN  = "\033[36m"; DIM = "\033[2m"

def _g(s): return f"{GREEN}{s}{RESET}"
def _y(s): return f"{YELLOW}{s}{RESET}"
def _r(s): return f"{RED}{s}{RESET}"
def _b(s): return f"{BOLD}{CYAN}{s}{RESET}"
def _d(s): return f"{DIM}{s}{RESET}"

# ── Defaults from environment ───────────────────────────────────────────────────
_DEFAULT_MODEL     = os.environ.get("OLLAMA_MODEL",       "llama3.1")
_DEFAULT_EMBED     = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
_DEFAULT_HOST      = os.environ.get("OLLAMA_HOST",        "http://localhost:11434")
_DEFAULT_JSON_FILE = "./cmg_memory_store.json"
_SYSTEM_PROMPT     = "You are a helpful assistant with persistent long-term memory."


# ── Store factory ───────────────────────────────────────────────────────────────

def build_store(args: argparse.Namespace):
    """
    Construct the correct VectorStore from CLI args.
    Pinecone API key is ONLY read from the environment — never from the CLI.
    """
    from cmg.store import InMemoryVectorStore, PineconeVectorStore

    if args.store == "json":
        path = getattr(args, "file", None) or _DEFAULT_JSON_FILE
        print(f"  Store   : {_g('JSON')}  →  {path}")
        return InMemoryVectorStore(persist_path=path)

    # ── Pinecone ──────────────────────────────────────────────────────────────
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        print(_r("\n  Error: PINECONE_API_KEY environment variable is not set."))
        print("  Set it before running:\n")
        print("    Linux / macOS:")
        print("      export PINECONE_API_KEY=pcsk_your_key_here\n")
        print("    Windows cmd:")
        print("      set PINECONE_API_KEY=pcsk_your_key_here\n")
        print("    PowerShell:")
        print("      $env:PINECONE_API_KEY='pcsk_your_key_here'\n")
        print("  Get your key at: https://app.pinecone.io → API Keys\n")
        sys.exit(1)

    index_name = (
        getattr(args, "pinecone_index", None)
        or os.environ.get("PINECONE_INDEX_NAME", "cmg-memory")
    )
    namespace = (
        getattr(args, "pinecone_namespace", None)
        or os.environ.get("PINECONE_NAMESPACE", "cmg-default")
    )
    dimension = getattr(args, "pinecone_dimension", None) or 768

    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
    print(f"  Store   : {_g('Pinecone')}")
    print(f"  Key     : {_g(masked_key)}")
    print(f"  Index   : {index_name}   namespace: {namespace}")
    print(f"  Dim     : {dimension}  (must match your embed model output)")

    return PineconeVectorStore(
        api_key    = api_key,
        index_name = index_name,
        namespace  = namespace,
        embedding_dim = dimension,
    )


# ── Memory factory ──────────────────────────────────────────────────────────────

def build_memory(args: argparse.Namespace):
    from cmg import CMGMemory, create_adapter

    model = getattr(args, "model", None) or _DEFAULT_MODEL
    embed = getattr(args, "embed",  None) or _DEFAULT_EMBED
    host  = getattr(args, "host",   None) or _DEFAULT_HOST

    print(f"  Model   : {model}")
    print(f"  Embed   : {embed}")

    adapter = create_adapter("ollama", model=model, embedding_model=embed, host=host)
    store   = build_store(args)
    verbose = getattr(args, "verbose", False)

    memory = CMGMemory(
        adapter,
        store         = store,
        system_prompt = _SYSTEM_PROMPT,
        log_level     = logging.DEBUG if verbose else logging.WARNING,
    )
    return memory, adapter


# ── Mode: chat ──────────────────────────────────────────────────────────────────

HELP_TEXT = """
Commands:
  memories              List all stored memories with scores
  memories <layer>      Filter: working / episodic / semantic / identity
  search <query>        Semantic search across memories
  stats                 Memory statistics
  forget <text>         Delete a memory by partial content match
  remember <text>       Manually add a fact memory
  explain <query>       Show CRR retrieval breakdown for a query
  debug on / off        Toggle extraction debug logging
  consolidate           Run decay + merge pass now
  clear                 Clear conversation history (keeps memories)
  migrate               Migrate memories to the other store backend
  help                  Show this message
  quit / exit           End session and save
"""

def _fmt_chunk(chunk, verbose: bool = False) -> str:
    scoring = chunk.metadata.get("dfs") or chunk.metadata.get("scoring") or {}
    summary = scoring.get("summary", "")
    bar_len = int(chunk.score * 16)
    bar     = "[" + "█" * bar_len + "░" * (16 - bar_len) + "]"
    col     = GREEN if chunk.score >= 0.65 else (YELLOW if chunk.score >= 0.45 else RED)
    lines   = [
        f"  {col}{bar}{RESET} {chunk.score:.3f}  "
        f"[{chunk.layer.value}][{chunk.memory_type.value}]  "
        f"access={chunk.access_count}",
        f"  {chunk.content}",
    ]
    if verbose and summary:
        lines.append(f"  {_d(summary)}")
    return "\n".join(lines)


def run_chat(args: argparse.Namespace) -> None:
    store_label = "Pinecone" if args.store == "pinecone" else "JSON"

    print(f"\n{_b('╔══════════════════════════════════════════╗')}")
    print(f"{_b('║       CMG Memory  ·  Chat               ║')}")
    print(f"{_b('╚══════════════════════════════════════════╝')}\n")

    # Quick Ollama connectivity check before spending time on store init
    host = getattr(args, "host", None) or _DEFAULT_HOST
    try:
        from cmg import create_adapter
        probe_adapter = create_adapter(
            "ollama",
            model           = getattr(args, "model", None) or _DEFAULT_MODEL,
            embedding_model = getattr(args, "embed", None)  or _DEFAULT_EMBED,
            host            = host,
        )
        probe_adapter.embed("ping")
    except Exception as e:
        print(_r(f"  Ollama connection failed: {e}"))
        print(f"  Is `ollama serve` running at {host}?")
        sys.exit(1)

    memory, adapter = build_memory(args)
    existing = memory.stats()["total_memories"]
    if existing:
        print(f"  Loaded  : {_g(str(existing))} memories from previous sessions")
    else:
        print(f"  Loaded  : fresh start")
    print(HELP_TEXT)

    debug_on = getattr(args, "verbose", False)

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        low = raw.lower()

        if low in ("quit", "exit"):
            break
        if low == "help":
            print(HELP_TEXT); continue
        if low == "clear":
            memory.clear_conversation()
            print("  Conversation cleared. Memories intact.\n"); continue
        if low == "consolidate":
            print("  Running consolidation pass...")
            r = memory._consolidator.run(llm_adapter=adapter)
            print(f"  {_g('Done')}  promoted={r.promoted} merged={r.merged} "
                  f"decayed={r.decayed} forgotten={r.forgotten}\n"); continue
        if low == "stats":
            s = memory.stats()
            print(f"\n  Total   : {s['total_memories']}")
            print(f"  Layer   : {s['by_layer']}")
            print(f"  Type    : {s['by_type']}")
            print(f"  Turns   : {s['session_turns']}")
            print(f"  Backend : {_g(store_label)}\n"); continue
        if low == "debug on":
            debug_on = True
            logging.getLogger("cmg").setLevel(logging.DEBUG)
            print("  Debug on.\n"); continue
        if low == "debug off":
            debug_on = False
            logging.getLogger("cmg").setLevel(logging.WARNING)
            print("  Debug off.\n"); continue
        if low.startswith("memories"):
            parts = raw.split(maxsplit=1)
            lf    = parts[1].strip() if len(parts) > 1 else None
            chunks = memory.list_memories(layer=lf)
            if not chunks:
                print(f"  (no memories{' in ' + lf if lf else ''})\n")
            else:
                print(f"\n  {len(chunks)} memor{'y' if len(chunks)==1 else 'ies'}:\n")
                for c in chunks:
                    print(_fmt_chunk(c, verbose=debug_on))
                print()
            continue
        if low.startswith("search "):
            q = raw[7:].strip()
            results = memory.search(q, top_k=5)
            print(f"\n  Top {len(results)} for \"{q}\":\n")
            for c in results:
                print(_fmt_chunk(c, verbose=debug_on))
            print(); continue
        if low.startswith("forget "):
            deleted = memory.forget(raw[7:].strip())
            print(f"  {'Forgotten.' if deleted else 'No match found.'}\n"); continue
        if low.startswith("remember "):
            text = raw[9:].strip()
            if text:
                chunk = memory.remember(text, memory_type="fact")
                print(f"  Stored: score={chunk.score:.3f} [{chunk.memory_type.value}]\n")
            continue
        if low.startswith("explain "):
            q = raw[8:].strip()
            if q and hasattr(memory._retriever, "explain"):
                try:
                    print("\n" + memory._retriever.explain(q, adapter.embed(q), top_k=4) + "\n")
                except Exception as e:
                    print(f"  explain error: {e}\n")
            continue
        if low == "migrate":
            _run_migrate(memory, args); continue

        # Regular chat
        t0       = time.time()
        response = memory.chat(raw)
        elapsed  = time.time() - t0
        print(f"\nAssistant: {response}")
        print(_d(f"  ({elapsed:.1f}s  ·  {memory.stats()['total_memories']} memories stored)\n"))

    print("  Saving...")
    memory.end_session()
    print(f"  {_g('Session saved')} to {_g(store_label)} store.\n")


# ── Mode: demo ──────────────────────────────────────────────────────────────────

def run_demo(args: argparse.Namespace) -> None:
    store_label = "Pinecone" if args.store == "pinecone" else "JSON"
    print(f"\n{_b('CMG Memory — Automated Demo')}")
    print(f"  Store: {_g(store_label)}\n")

    memory, adapter = build_memory(args)

    seeds = [
        "My name is Rishav and I work on XAI for autonomous driving at IISc.",
        "I always use Grad-CAM on Mask2Former for explainability.",
        "My goal is to publish at CVPR by end of this year.",
        "I prefer concise answers without unnecessary preamble.",
    ]

    print("── Session 1: seeding ──\n")
    for msg in seeds:
        print(f"  {_d('User:')} {msg}")
        memory.chat(msg)
        print(f"  {_g(str(memory.stats()['total_memories']))} memories stored\n")

    memory.end_session(run_consolidation=True)

    print("── Session 2: recall ──\n")
    memory.new_session()
    for q in ["What XAI method do I use?",
               "What's my publication goal?",
               "What do you know about my work?"]:
        print(f"  {_d('User:')} {q}")
        resp = memory.chat(q)
        print(f"  Assistant: {resp[:200]}\n")

    memory.end_session(run_consolidation=False)


# ── Mode: inspect ────────────────────────────────────────────────────────────────

def run_inspect(args: argparse.Namespace) -> None:
    store_label = "Pinecone" if args.store == "pinecone" else "JSON"
    print(f"\n{_b('CMG Memory — Inspect')}  [{_g(store_label)}]\n")
    memory, _ = build_memory(args)
    s = memory.stats()
    print(f"  Total : {s['total_memories']} memories")
    print(f"  Layer : {s['by_layer']}")
    print(f"  Type  : {s['by_type']}\n")

    if s["total_memories"] == 0:
        print("  (store is empty)\n"); return

    for layer in ["identity", "semantic", "episodic", "working"]:
        chunks = memory.list_memories(layer=layer)
        if not chunks: continue
        print(f"  {_b(layer.upper())} ({len(chunks)})")
        for c in chunks:
            print(_fmt_chunk(c, verbose=True))
        print()


# ── Mode: consolidate ─────────────────────────────────────────────────────────────

def run_consolidate(args: argparse.Namespace) -> None:
    print(f"\n{_b('CMG Memory — Consolidation')}\n")
    memory, adapter = build_memory(args)
    before = memory.stats()["total_memories"]
    print(f"  Before : {before} memories")
    t0 = time.time()
    r  = memory._consolidator.run(llm_adapter=adapter)
    print(f"  After  : {memory.stats()['total_memories']} memories  "
          f"(removed {_g(str(before - memory.stats()['total_memories']))})")
    print(f"  promoted={r.promoted}  merged={r.merged}  "
          f"forgotten={r.forgotten}  contradictions={r.contradictions}")
    print(f"  Time   : {time.time()-t0:.1f}s\n")


# ── Mode: reset ────────────────────────────────────────────────────────────────────

def run_reset(args: argparse.Namespace) -> None:
    store_label = "Pinecone" if args.store == "pinecone" else "JSON"
    print(f"\n{_b('CMG Memory — Reset')}  [{_r(store_label)}]\n")
    confirm = input(f"  {_r('Delete ALL memories?')} Type YES to confirm: ").strip()
    if confirm != "YES":
        print("  Aborted.\n"); return

    if args.store == "json":
        path = getattr(args, "file", None) or _DEFAULT_JSON_FILE
        if os.path.exists(path):
            os.remove(path)
            print(f"  {_g('Deleted')} {path}\n")
        else:
            print(f"  File not found: {path}\n")
    else:
        from cmg.store import PineconeVectorStore
        api_key    = os.environ.get("PINECONE_API_KEY", "")
        index_name = getattr(args, "pinecone_index", None) or os.environ.get("PINECONE_INDEX_NAME", "cmg-memory")
        namespace  = getattr(args, "pinecone_namespace", None) or os.environ.get("PINECONE_NAMESPACE", "cmg-default")
        store = PineconeVectorStore(api_key=api_key, index_name=index_name, namespace=namespace)
        store.delete_namespace()
        print(f"  {_g('Cleared')} namespace '{namespace}' in index '{index_name}'\n")


# ── Mode: models ───────────────────────────────────────────────────────────────────

def run_models(args: argparse.Namespace) -> None:
    print(f"\n{_b('Available Ollama Models')}\n")
    host = getattr(args, "host", None) or _DEFAULT_HOST
    try:
        import ollama
        models = ollama.Client(host=host).list().get("models", [])
        for m in models:
            name = m.get("name", "?")
            size = m.get("size", 0)
            print(f"  {_g(name):<42} {size/1e9:.1f} GB" if size else f"  {_g(name)}")
        if not models:
            print("  (no models pulled yet — try: ollama pull llama3.1)")
        print()
    except Exception as e:
        print(_r(f"  Failed: {e}  —  is ollama running at {host}?\n"))


# ── Migration helper ─────────────────────────────────────────────────────────────

def _run_migrate(memory, args: argparse.Namespace) -> None:
    from cmg.store import InMemoryVectorStore, PineconeVectorStore
    current = "Pinecone" if args.store == "pinecone" else "JSON"
    target_label = "JSON" if args.store == "pinecone" else "Pinecone"
    print(f"\n  Migrate {_g(current)} → {_g(target_label)}")
    if input("  Proceed? (yes/no): ").strip().lower() != "yes":
        print("  Aborted.\n"); return

    if args.store == "json":
        api_key = os.environ.get("PINECONE_API_KEY", "").strip()
        if not api_key:
            print(_r("  PINECONE_API_KEY not set.\n")); return
        target = PineconeVectorStore(
            api_key    = api_key,
            index_name = os.environ.get("PINECONE_INDEX_NAME", "cmg-memory"),
            namespace  = os.environ.get("PINECONE_NAMESPACE",  "cmg-default"),
        )
    else:
        path   = getattr(args, "file", None) or _DEFAULT_JSON_FILE
        target = InMemoryVectorStore(persist_path=path)

    n = PineconeVectorStore.migrate_from(memory._store, target)
    print(f"  {_g('Done')} — migrated {n} chunks\n")


# ── Argument parser ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="CMG Memory — Ollama chat with persistent memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Quick start:
          python run.py                         chat, local JSON store
          python run.py --store pinecone        chat, Pinecone cloud store
          python run.py --mode demo             automated memory demo
          python run.py --mode inspect          browse all stored memories
          python run.py --mode reset            wipe all memories
          python run.py --mode models           list Ollama models

        Pinecone (set env vars first):
          export PINECONE_API_KEY=pcsk_...
          export PINECONE_INDEX_NAME=my-index   (optional, default: cmg-memory)
          export PINECONE_NAMESPACE=user-abc    (optional, default: cmg-default)
          python run.py --store pinecone

        Override Ollama defaults:
          python run.py --model mistral --embed nomic-embed-text
          OLLAMA_MODEL=mistral python run.py
        """),
    )

    p.add_argument(
        "--store",
        choices=["json", "pinecone"],
        default="json",
        help=(
            "Storage backend. "
            "json  = local JSON file (default, no extra deps). "
            "pinecone = Pinecone cloud vector DB (requires PINECONE_API_KEY env var, "
            "pip install pinecone-client)."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["chat", "demo", "inspect", "consolidate", "reset", "models"],
        default="chat",
        help="Operating mode (default: chat).",
    )
    p.add_argument(
        "--file",
        default=None,
        metavar="PATH",
        help="JSON file path (--store json only). Default: ./cmg_memory_store.json",
    )

    pc = p.add_argument_group("Pinecone overrides (all also readable from env vars)")
    pc.add_argument("--pinecone-index",     default=None, metavar="NAME",
                    help="Index name. Overrides PINECONE_INDEX_NAME. (default: cmg-memory)")
    pc.add_argument("--pinecone-namespace", default=None, metavar="NS",
                    help="Namespace. Overrides PINECONE_NAMESPACE. (default: cmg-default)")
    pc.add_argument("--pinecone-dimension", default=None, type=int, metavar="DIM",
                    help="Embedding dim for new index. 768=nomic, 1536=openai, 384=minilm. (default: 768)")

    ol = p.add_argument_group("Ollama overrides (all also readable from env vars)")
    ol.add_argument("--model", default=None, help=f"Chat model. Overrides OLLAMA_MODEL. (default: {_DEFAULT_MODEL})")
    ol.add_argument("--embed", default=None, help=f"Embed model. Overrides OLLAMA_EMBED_MODEL. (default: {_DEFAULT_EMBED})")
    ol.add_argument("--host",  default=None, help=f"Ollama URL. Overrides OLLAMA_HOST. (default: {_DEFAULT_HOST})")

    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging and show scoring details on memories.")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "chat":        run_chat,
        "demo":        run_demo,
        "inspect":     run_inspect,
        "consolidate": run_consolidate,
        "reset":       run_reset,
        "models":      run_models,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()