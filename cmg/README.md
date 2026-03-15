# CMG Memory — Contextual Memory Gradient

**Persistent, portable, user-owned memory for any LLM.**

Works with OpenAI · Anthropic Claude · Ollama · HuggingFace · Pinecone · local JSON

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Why CMG?

Every major AI provider is building memory — but only for their own platform. Your memories in ChatGPT cannot move to Claude. Your context in Claude cannot reach a local Llama model. They have no incentive to build portability because lock-in is the business model.

CMG is the memory layer that lives outside any provider. You own the store. You choose the model. Your memories follow you across every AI tool you use.

Three things no provider will build:

- **Cross-provider portability** — one memory store works with GPT-4o on Monday, Claude on Tuesday, local Llama on Wednesday
- **Local-first privacy** — JSON store + Ollama means zero data leaves your machine
- **Intelligent forgetting** — memories decay, deduplicate, and promote through layers automatically, unlike platform memory that accumulates everything forever

---

## How it works

CMG implements a **four-layer memory gradient** that mirrors how human memory actually works — not a flat key-value store, but a hierarchy where important facts rise and trivial ones fade.

```
Working memory      session-scoped, full fidelity
      ↓  consolidate after session ends
Episodic buffer     recent events, lightly filtered, decays over days
      ↓  distill when seen in multiple sessions
Semantic core       stable facts, preferences, skills — months lifespan
      ↓  crystallize after repeated reinforcement
Identity layer      permanent: name, role, core values, communication style
```

**Promotion is session-count based, not time-based.** A high-confidence fact seen in 2 different sessions goes to semantic. The same fact seen across 4 sessions goes to identity. This reflects real patterns in how you use AI rather than arbitrary time delays.

**Three-gate write pipeline** prevents the store from filling with noise:

1. **Novelty gate** — cosine similarity check against existing memories. Near-duplicates reinforce the existing chunk instead of creating a new one.
2. **Score gate** — deterministic feature scoring rejects fragments, hedged statements, and transient context.
3. **Capacity gate** — each layer has a hard cap. When full, the lowest-scoring chunk is evicted before writing.

**Contextual Resonance Retrieval (CRR)** replaces cosine-only search with four signals fused per query: semantic affinity (including source context), temporal frame matching, type resonance (goal queries boost goal-type memories), and reinforcement history. Results are diversified with Maximal Marginal Relevance to eliminate near-duplicate chunks crowding the top-K.

---

## Quick start

```bash
pip install cmg-memory[ollama]
ollama pull llama3.1
ollama pull nomic-embed-text
```

```python
from cmg import CMGMemory, create_adapter

adapter = create_adapter("ollama", model="llama3.1", embedding_model="nomic-embed-text")
memory  = CMGMemory(adapter, persist_path="./memory.json")

# Session 1
memory.chat("I'm a PhD researcher at IISc Bangalore working on XAI for autonomous driving.")
memory.chat("I always use Grad-CAM. I hate verbose responses — always be concise.")
memory.end_session()

# Session 2 — memory persists across restarts
memory.new_session()
response = memory.chat("What do you know about my work?")
# recalls IISc, XAI, Grad-CAM, and your preference for concise answers
```

---

## Installation

```bash
# Core library — no required dependencies
pip install cmg-memory

# With your LLM provider
pip install cmg-memory[openai]        # OpenAI GPT models
pip install cmg-memory[anthropic]     # Anthropic Claude
pip install cmg-memory[ollama]        # Ollama local models
pip install cmg-memory[huggingface]   # HuggingFace API or local inference

# Vector store backends
pip install cmg-memory[pinecone]      # Pinecone serverless cloud
pip install cmg-memory[chromadb]      # ChromaDB local persistent store

# Better embeddings (recommended for Claude and Ollama)
pip install cmg-memory[embeddings]    # sentence-transformers

# Hosted API client
pip install cmg-memory[api]           # httpx-based REST client

# Install everything
pip install cmg-memory[all]
```

---

## Provider setup

### Ollama (recommended for local/private use)

```bash
ollama serve
ollama pull llama3.1              # llama3.1 or mistral — better JSON compliance than llama3.2
ollama pull nomic-embed-text      # local embedding model (768 dimensions)
```

```python
adapter = create_adapter(
    "ollama",
    model           = "llama3.1",
    embedding_model = "nomic-embed-text",
    host            = "http://localhost:11434",
)
```

> **Model note:** Use `llama3.1` or `mistral`, not `llama3.2`. The 3.2 model frequently produces malformed JSON from the extraction prompt. CMG has a multi-strategy JSON repair fallback but a better model produces better memories.

### OpenAI

```python
adapter = create_adapter("openai", api_key="sk-...", model="gpt-4o")
# Uses text-embedding-3-small natively — no extra embedding setup needed
```

### Anthropic Claude

```python
adapter = create_adapter("claude", api_key="sk-ant-...", model="claude-sonnet-4-5")
# Embeddings via sentence-transformers: pip install cmg-memory[embeddings]
```

### HuggingFace API

```python
adapter = create_adapter("huggingface",
    model   = "mistralai/Mistral-7B-Instruct-v0.2",
    api_key = "hf_...",
    local   = False,
)
```

### HuggingFace local (fully air-gapped)

```python
adapter = create_adapter("huggingface", model="microsoft/phi-2", local=True)
```

---

## Storage backends

### JSON (default — no extra setup)

```python
memory = CMGMemory(adapter, persist_path="./memory.json")
```

Atomic writes — a crash mid-save cannot corrupt the file. Suitable for personal use on a single machine.

### Pinecone (cloud, multi-device sync)

```bash
pip install pinecone
export PINECONE_API_KEY=pcsk_your_key_here
```

```python
from cmg.store import PineconeVectorStore

store = PineconeVectorStore(
    api_key       = os.environ["PINECONE_API_KEY"],
    index_name    = "cmg-memory",   # created automatically if it doesn't exist
    namespace     = "user-rishav",  # one namespace per user — memories never bleed across users
    embedding_dim = 768,            # must match your embedding model
)
memory = CMGMemory(adapter, store=store)
```

The free tier supports up to 100K vectors — sufficient for years of personal use. Layer filters push to Pinecone as metadata filters so latency stays constant regardless of total store size.

### ChromaDB (local, production-grade indexing)

```python
from cmg.store import ChromaVectorStore

store = ChromaVectorStore(collection_name="my_memory", persist_dir="./chroma_db")
memory = CMGMemory(adapter, store=store)
```

### Migrate between backends

```python
from cmg.store import InMemoryVectorStore, PineconeVectorStore

old = InMemoryVectorStore(persist_path="./memory.json")
new = PineconeVectorStore(api_key="...", index_name="cmg-memory", namespace="me")
PineconeVectorStore.migrate_from(old, new)
```

---

## CLI

```bash
# Interactive chat — JSON store (default)
python run.py

# Interactive chat — Pinecone store
python run.py --store pinecone

# Set PINECONE_API_KEY first:
#   Linux/macOS: export PINECONE_API_KEY=pcsk_...
#   Windows cmd: set PINECONE_API_KEY=pcsk_...
#   PowerShell:  $env:PINECONE_API_KEY='pcsk_...'

# One-shot task
python run.py "summarise our last conversation"

# Other modes
python run.py --mode demo            # automated demo showing memory across sessions
python run.py --mode inspect         # browse all stored memories
python run.py --mode consolidate     # run decay + merge pass manually
python run.py --mode reset           # wipe all memories
python run.py --mode models          # list available Ollama models

# Use a different model or custom file
python run.py --model mistral --file ~/work_memory.json
```

### In-chat commands

| Command | What it does |
|---|---|
| `memories` | List all stored memories with scores and layers |
| `memories semantic` | Filter by layer |
| `search <query>` | Semantic search across memories |
| `forget <text>` | Delete a memory by partial content match |
| `remember <text>` | Manually store a fact |
| `explain <query>` | Show CRR retrieval scores for a query |
| `consolidate` | Run promotion/decay pass now |
| `stats` | Memory count by layer and type |
| `debug on/off` | Toggle verbose extraction logging |

---

## Python API

```python
from cmg import CMGMemory, create_adapter

memory = CMGMemory(
    llm_adapter,
    persist_path           = "./memory.json",
    system_prompt          = "You are a helpful assistant.",
    top_k_retrieve         = 8,
    memory_format          = "structured",   # "structured" | "prose" | "minimal"
    auto_extract           = True,
    auto_consolidate_after = 10,
)

# Chat with automatic memory injection and extraction
response = memory.chat("I always use PyTorch for deep learning")
response = memory.chat("What framework do I prefer?")   # recalls PyTorch

# Manual memory management
memory.remember("User is a senior ML engineer", memory_type="fact", layer="semantic")
memory.forget("PyTorch")                     # by partial content match
memory.search("deep learning", top_k=5)     # semantic search, returns MemoryChunk list
memory.list_memories(layer="semantic")       # list by layer

# Session lifecycle
memory.end_session()     # consolidate, promote, decay, persist
memory.new_session()     # fresh conversation context, memories intact
memory.stats()           # {"total_memories": 24, "by_layer": {...}, ...}

# Inspect write gate efficiency
print(memory._write_gate.stats.summary())
# evaluated=47  written=12 (26%)  reinforced=18  discarded=17  evictions=0

# Inspect layer distribution
print(memory._write_gate.layer_usage())
# {"episodic": {"count": 8, "cap": 200, "pct": 4.0}, "semantic": {...}, ...}
```

---

## Memory types and layers

### Types

| Type | What it stores | Example |
|---|---|---|
| `fact` | Stable personal facts | "User is a PhD researcher at IISc" |
| `preference` | Likes, dislikes, style | "User hates verbose responses" |
| `skill` | Expertise and tools | "User uses Grad-CAM for explainability" |
| `goal` | Active objectives | "User's goal is to submit to CVPR 2026" |
| `decision` | Deliberate choices | "User switched from TensorFlow to PyTorch" |
| `correction` | Explicit corrections | "User corrected: dataset has 120K not 50K images" |
| `relationship` | People and connections | "User's PhD advisor is Prof. Venkatesh Babu" |
| `event` | Time-bounded occurrences | "User attended ICCV 2024" |

### Layer promotion

| Transition | Condition |
|---|---|
| Working → Episodic | Any chunk stored during a session |
| Episodic → Semantic (fast-track) | High-score `fact`, `preference`, `skill`, `goal`, or `correction` after 1 session |
| Episodic → Semantic (standard) | Any type seen across 2 sessions |
| Semantic → Identity | Score ≥ 0.75 seen across 4 sessions |

Consolidation runs automatically at `memory.end_session()` and every 10 turns during a session. Run it manually anytime with `memory._consolidator.run()`.

---

## Scoring system

CMG uses **Deterministic Feature Scoring (DFS)** — fully model-agnostic. The same message gets the same score regardless of which LLM you're using. Swap from GPT-4o to Ollama and scores are identical.

| Feature | What it measures | Default weight |
|---|---|---|
| F1 Lexical salience | Emphasis (`always`, `never`, `critical`) vs hedging (`maybe`, `sometimes`) | 0.25 |
| F2 Temporal stability | Stable traits vs transient states (`currently debugging`) | 0.30 |
| F3 Entity density | Named tools, institutions, people, version numbers | 0.20 |
| F4 Structural position | Volunteered unprompted vs answering a direct question | 0.15 |
| F5 Store novelty | Cosine distance from existing memories | 0.10 |

Scores are stored in `chunk.metadata["dfs"]` and visible in the `memories` output with `debug on`.

---

## Architecture

```
cmg/
├── types.py            MemoryChunk, MemoryLayer, MemoryType data classes
├── adapters.py         LLM provider adapters (OpenAI, Claude, HuggingFace, Ollama)
├── store.py            InMemoryVectorStore · PineconeVectorStore · ChromaVectorStore
├── scorer.py           Deterministic Feature Scorer — model-agnostic importance scoring
├── extractor.py        Extraction with pre-filter, JSON repair, dedup, normalisation
├── crr.py              Contextual Resonance Retrieval — four-signal fusion + MMR
├── write_gate.py       Three-gate write pipeline (novelty · score · capacity)
├── consolidation.py    Decay · session-based promotion · merge · contradiction detection
├── memory.py           CMGMemory — unified public interface
├── client.py           HTTP client for the hosted API
└── cli.py              pip-installed entry point

api/
└── server.py           FastAPI hosted API server

run.py                  Interactive CLI (--store json|pinecone, --mode chat|inspect|...)
eval.py                 Four-suite evaluation harness
diagnose.py             Recall failure diagnostic — traces 6 pipeline stages
```

---

## Evaluation

```bash
# Full evaluation — requires Ollama running
python eval.py

# Individual suites
python eval.py --suite recall        # cross-session fact recall accuracy
python eval.py --suite staleness     # contradiction + update handling
python eval.py --suite scoring       # score discrimination (no LLM chat needed — fastest)
python eval.py --suite retrieval     # retrieval precision

# Different model
python eval.py --model mistral --verbose
```

Results saved to `eval_results.json`. Grades: A (≥85%) · B (≥70%) · C (≥55%) · D (≥40%) · F

---

## Diagnosing recall failures

When the system fails to recall something you told it:

```bash
python diagnose.py
```

Edit the three constants at the top to match your case:

```python
SESSION1_MESSAGE = "I use Grad-CAM for explainability"
SESSION2_QUERY   = "What do you know about my work?"
KEYWORD          = "Grad-CAM"
```

The script traces six pipeline stages and identifies exactly where it failed:

| Stage | What it checks |
|---|---|
| 1. Extraction | Did the LLM extract the keyword at all? |
| 2. Score | Was the score above the Gate 2 threshold? |
| 3. Embedding | Does the query semantically match the stored chunk? |
| 4. Retrieval | Does CRR surface it in the top-K results? |
| 5. Injection | Is it in the context block sent to the LLM? |
| 6. End-to-end | Does the LLM actually use it in its response? |

At the end it prints the root cause and the specific fix.

---

## Hosted API

Deploy CMG as a REST API — users get an API key and call endpoints without running anything locally.

```bash
# Setup
pip install fastapi uvicorn[standard] python-dotenv pinecone sentence-transformers
cp .env.example .env   # fill in PINECONE_API_KEY and CMG_API_SECRET
uvicorn api.server:app --reload --port 8000

# Deploy to Render (free tier, 2 minutes)
# 1. Push to GitHub
# 2. Connect repo on render.com
# 3. Set PINECONE_API_KEY in Render dashboard
# 4. Deploy — render.yaml handles the rest
```

Interactive API docs at `http://localhost:8000/docs`.

```
POST   /v1/chat                  chat turn with memory injection + extraction
POST   /v1/memories              manually store a memory
GET    /v1/memories              list all memories (filterable by layer/type)
DELETE /v1/memories/{id}         delete a memory by ID
POST   /v1/memories/search       semantic search
POST   /v1/extract               extract memories from arbitrary text
POST   /v1/sessions/end          end session + run consolidation
GET    /v1/stats                 memory statistics
GET    /health                   health check
```

### Python client

```python
from cmg.client import CMGClient

with CMGClient(api_key="cmg_yourkey", base_url="https://your-api.onrender.com") as client:
    result   = client.chat("I always use PyTorch")
    print(result.response)
    print(f"Memories stored: {result.memories_stored}")

    memories = client.list_memories(layer="semantic")
    client.remember("User is at IISc Bangalore", memory_type="fact")
    results  = client.search("deep learning framework", top_k=5)
    client.end_session()   # runs consolidation, resets session
```

---

## CMG Agent

A separate codebase in `cmg_agent/` extends CMG into an autonomous coding agent — file read/write, shell execution, code search, Python REPL — with **decision memory** that logs why every action was taken.

```bash
cd cmg_agent
pip install ollama duckduckgo-search
python run_agent.py                           # interactive shell
python run_agent.py "fix the bug in auth.py"  # one-shot task
python run_agent.py --why auth.py             # explain past decisions about a file
python run_agent.py --decisions 10            # show recent decision log
python run_agent.py --yes                     # auto-approve all file/shell operations
```

Every tool call is logged with its full reasoning chain, alternatives considered, and outcome. The agent queries this log automatically when working on related files, so it never reverses a deliberate decision without understanding why it was made.

---

## Extending CMG

### Add a new LLM provider

```python
from cmg.adapters import LLMAdapter

class MyAdapter(LLMAdapter):
    def chat(self, messages, system=None, temperature=0.7, max_tokens=2048) -> str:
        # call your API, return the response string
        ...

    def embed(self, text: str) -> list[float]:
        # return a normalised embedding vector
        ...
```

### Add a new vector store

```python
from cmg.store import VectorStore

class MyVectorStore(VectorStore):
    def upsert(self, chunk): ...
    def delete(self, chunk_id): ...
    def get(self, chunk_id): ...
    def search(self, query_embedding, top_k, layer_filter): ...
    def all_chunks(self): ...
    def count(self): ...
```

### Tune the write gate

```python
from cmg.write_gate import WriteGate
from cmg.types import MemoryLayer

memory._write_gate.novelty_threshold = 0.75   # more aggressive dedup
memory._write_gate.min_score         = 0.45   # only store higher-confidence facts
memory._write_gate.set_cap(MemoryLayer.EPISODIC, 100)
```

### Tune CRR retrieval

```python
from cmg.crr import CRREngine, CRRWeights

memory._retriever = CRREngine(
    store    = memory._store,
    embed_fn = adapter.embed,
    weights  = CRRWeights(
        alpha      = 0.50,   # semantic affinity weight
        beta       = 0.15,   # temporal context weight
        gamma      = 0.25,   # type resonance weight
        delta      = 0.10,   # reinforcement history weight
        mmr_lambda = 0.6,    # 0=max diversity, 1=max relevance
    ),
)
```

---

## What companies won't build

OpenAI and Anthropic _can_ build what's here. They won't, for structural reasons:

- **Portability** reduces platform lock-in — your memories working on a competitor's model is not in their interest
- **Local-only, zero-telemetry memory** conflicts with a business model that requires data on their servers
- **Strategic forgetting** means losing data that could improve their models and ad targeting
- **Cross-agent, cross-session shared memory** commoditises the model layer — the infrastructure becomes more valuable than any individual model

CMG is built precisely for these gaps. The library is open source. The store is yours. The model is your choice.

---

## License

MIT

---

*Built as part of research into portable AI memory infrastructure at IISc Bangalore.*