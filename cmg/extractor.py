"""
CMG Memory System — Extractor (simplified)
==========================================
Extracts memorable facts from conversation turns.

Three responsibilities only:
  1. Pre-filter  — block filler/questions before any LLM call
  2. LLM extract — ask the model what's worth remembering
  3. Validate    — ensure output is well-formed before scoring

Quality filtering (score thresholds, novelty, dedup) is handled
downstream by DeterministicScorer and WriteGate — not here.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from .types import MemoryChunk, MemoryLayer, MemoryType
from .adapters import LLMAdapter
from .scorer import DeterministicScorer

logger = logging.getLogger("cmg.extractor")

VALID_TYPES = {t.value for t in MemoryType}

# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Extract stable personal facts about the user from this conversation.
Return a JSON array. Return [] if nothing stable to extract.

EXTRACT: name, job, location, skills, tools, preferences, goals, relationships, decisions.
SKIP: questions, thanks, temporary states, what the assistant said.

Rules:
- Every content must start with "User" and be a complete sentence
- Only extract facts that will still be true in 6 months
- Return [] for most turns — this is the correct response

Format: [{"content": "User ...", "type": "fact|preference|skill|goal|decision|correction|relationship|event"}]

Examples:
"my name is Rishav"                    -> [{"content": "User's name is Rishav", "type": "fact"}]
"I work at Mercedes as AI Scientist"   -> [{"content": "User works at Mercedes as AI Research Scientist", "type": "fact"}]
"I always use PyTorch"                 -> [{"content": "User always uses PyTorch", "type": "skill"}]
"thanks ok got it"                     -> []
"how does attention work?"             -> []

Return ONLY the JSON array, nothing else."""

# ── Pre-filter ────────────────────────────────────────────────────────────────

_FILLER = re.compile(
    r"^\s*(thanks?|thank\s+you|ok(ay)?|sure|got\s+it|i\s+see|understood|"
    r"makes?\s+sense|great|perfect|good|nice|cool|yes|no|yep|right|"
    r"agreed|sounds?\s+good|alright|noted|awesome|i\s+understand)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_QUESTION = re.compile(
    r"^\s*(what|how|why|when|where|who|which|can\s+you|could\s+you|"
    r"would\s+you|do\s+you|is\s+there|will\s+you|should\s+i|how\s+do\s+i)\b",
    re.IGNORECASE,
)

_IDENTITY = re.compile(
    r"\b(i\s+am|i'm|my\s+(name|job|role|work|company|advisor|goal|phd)|"
    r"i\s+(use|work|hate|love|always|never|prefer|switched|decided|live))\b",
    re.IGNORECASE,
)

MIN_WORDS = 5


def _should_extract(msg: str) -> bool:
    msg = msg.strip()
    if not msg:
        return False
    if _FILLER.match(msg):
        return False
    if len(msg.split()) < MIN_WORDS and not _IDENTITY.search(msg):
        return False
    if _QUESTION.match(msg) and not _IDENTITY.search(msg):
        return False
    return True


# ── Extractor ─────────────────────────────────────────────────────────────────

class MemoryExtractor:

    def __init__(self, llm_adapter: LLMAdapter, scorer=None):
        self._llm    = llm_adapter
        self._scorer = scorer or DeterministicScorer()

    def extract(
        self,
        conversation: list[dict],
        session_id: Optional[str] = None,
        max_chunks: int = 10,
        existing_chunks: Optional[list] = None,
    ) -> list[MemoryChunk]:

        if not conversation:
            return []

        # Last user message
        user_msgs = [m.get("content", "") for m in conversation if m.get("role") == "user"]
        last_msg  = user_msgs[-1] if user_msgs else ""

        if not _should_extract(last_msg):
            logger.debug("Pre-filter blocked: %s", last_msg[:60])
            return []

        # Format last 4 turns for context
        conv_text = "\n".join(
            f"{m.get('role','user').capitalize()}: {m.get('content','')[:400]}"
            for m in conversation[-4:]
        )

        # LLM extraction
        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": f"Extract facts:\n\n{conv_text}"}],
                system=EXTRACTION_PROMPT,
                temperature=0.1,
                max_tokens=800,
            )
            raw = self._parse(response)
        except Exception as e:
            logger.warning("Extraction failed: %s", e)
            return []

        if not raw:
            return []

        # Build chunks
        chunks = []
        for item in raw[:max_chunks]:
            if not isinstance(item, dict):
                continue

            content = item.get("content", "").strip()
            if not content or not content.lower().startswith("user"):
                continue
            if len(content.split()) < 3:
                continue

            mem_type_str = item.get("type", "fact").lower()
            if mem_type_str not in VALID_TYPES:
                mem_type_str = "fact"

            mem_type = MemoryType(mem_type_str)

            # Score via DFS
            score, fv = self._scorer.score(
                content        = content,
                source_message = conv_text,
            )

            chunk = MemoryChunk(
                content      = content,
                layer        = MemoryLayer.EPISODIC,
                memory_type  = mem_type,
                score        = score,
                session_id   = session_id,
                source_text  = conv_text[:400],
                created_at   = time.time(),
                last_accessed= time.time(),
                metadata     = {"dfs": fv.to_dict(self._scorer.weights)},
            )
            chunks.append(chunk)
            logger.debug("Extracted [%s] score=%.2f: %s", mem_type_str, score, content[:80])

        return chunks

    def _parse(self, text: str) -> list:
        """Parse JSON array from LLM response. Two attempts."""
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

        # Attempt 1: direct
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        # Attempt 2: fix trailing commas
        if m:
            fixed = re.sub(r",\s*([}\]])", r"\1", m.group())
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        return []