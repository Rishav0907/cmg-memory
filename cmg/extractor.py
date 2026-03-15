"""
CMG Memory System - Memory Extractor

After each conversation turn, this module:
  1. Uses the LLM to identify memorizable information
  2. Classifies it by MemoryType
  3. Scores each memory using LLM reasoning across four dimensions
  4. Applies fast linguistic + temporal heuristics as a correction layer
  5. Creates MemoryChunk objects and routes them to the episodic buffer

Scoring architecture (three layers, all combined into final score):

  Layer A — LLM scoring (primary signal)
    The LLM reasons about each memory across four dimensions and returns
    a score 0.0–1.0 alongside an explanation. Done in a single combined
    extraction+scoring prompt to avoid an extra round-trip.

    Dimensions the LLM evaluates:
      1. Persistence  — will this still be true in 6 months?
      2. Specificity  — is this uniquely about this user vs. generic?
      3. Utility      — how often will this be useful in future conversations?
      4. Signal       — was this volunteered unprompted (higher) or elicited (lower)?

  Layer B — Linguistic heuristics (fast correction, ±0.15 max)
    Applied on top of the LLM score. Catches emphasis signals the LLM
    might miss or underweight:
      - Emphasis words ("always", "never", "hate", "love") → +0.10
      - Importance markers ("important", "remember this") → +0.15
      - Hedging words ("maybe", "sometimes", "might") → −0.10
      - Exclamation marks → +0.05
      - Temporal instability ("right now", "currently", "today") → −0.15

  Layer C — Temporal stability (caps the score for transient facts)
    A memory about a current temporary state is capped at 0.65 regardless
    of how the LLM scored it, since it will become stale quickly.
"""

from __future__ import annotations
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .types import MemoryChunk, MemoryLayer, MemoryType
from .adapters import LLMAdapter
from .scorer import DeterministicScorer, ScorerWeights, FeatureVector

logger = logging.getLogger("cmg.extractor")

VALID_TYPES = {t.value for t in MemoryType}


# ---------------------------------------------------------------------------
# Scoring result — transparent, debuggable
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """
    Holds the full scoring breakdown for a single memory.
    Stored in chunk.metadata["scoring"] so you can inspect why a score was set.
    """
    llm_score: float            # Raw LLM score (0.0–1.0)
    llm_reasoning: str          # LLM's one-line explanation
    linguistic_delta: float     # Adjustment from linguistic heuristics
    temporal_cap: Optional[float]  # If set, score was capped at this value
    final_score: float          # The score actually written to the chunk

    # Breakdown of linguistic signals found
    emphasis_hits: list[str] = field(default_factory=list)
    hedging_hits: list[str] = field(default_factory=list)
    transient_hits: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"llm={self.llm_score:.2f}"]
        if self.linguistic_delta != 0:
            parts.append(f"linguistic={self.linguistic_delta:+.2f}")
        if self.temporal_cap is not None:
            parts.append(f"capped@{self.temporal_cap:.2f}")
        parts.append(f"→ final={self.final_score:.2f}")
        return " | ".join(parts) + f"  ({self.llm_reasoning})"


# ---------------------------------------------------------------------------
# Extraction + scoring prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a strict memory extraction assistant.
Extract ONLY durable facts about the user — things worth knowing in 6 months.
When in doubt, return []. It is always better to store nothing than to store noise.

EXTRACT — stable, personal, specific:
  facts:       job, location, education, name, background
  preferences: tools, languages, communication style, explicit likes/dislikes
  skills:      expertise areas, technologies they know well
  goals:       active objectives with real stakes
  corrections: explicit corrections of previously stated information
  decisions:   deliberate choices about their work or tools

DO NOT EXTRACT — these are the most common mistakes:
  - Questions the user is asking ("what is X?", "how do I Y?", "can you Z?")
  - Acknowledgements ("thanks", "got it", "ok", "sure", "I see", "that makes sense")
  - The current task or what the user is asking for help with right now
  - Things the ASSISTANT said (only extract user-stated facts)
  - Rephrasing of something already clearly known
  - Vague statements not specific to this user ("coding is hard")
  - Temporary/session context that changes every day

EXAMPLES — do NOT extract:
  "Can you explain Grad-CAM?"        → [] (pure question)
  "Thanks that was helpful"          → [] (filler)
  "I am trying to fix a bug"         → [] (temporary session context)
  "That makes sense"                 → [] (acknowledgement)
  "What is the best way to do X?"    → [] (question)
  "I think maybe I like Python"      → [] (hedged, vague)

EXAMPLES — DO extract:
  "I work as an ML engineer at IISc Bangalore"  → fact, score ~0.85
  "I always use PyTorch, never TensorFlow"      → preference, score ~0.82
  "I hate verbose code, always be concise"      → preference, score ~0.88
  "My PhD advisor is Prof. Venkatesh Babu"      → relationship, score ~0.75
  "I switched from Python to Go last month"     → correction/decision, score ~0.80

SCORING — rate each memory 0.0-1.0, average of:
  1. Persistence: still true in 6 months? stable trait=high, current task=low
  2. Specificity: unique to this user? "works at IISc"=high, "likes coffee"=low
  3. Utility: useful in future conversations? communication style=high, one-off=low
  4. Signal: volunteered unprompted=high, answering a direct question=medium

Return ONLY a JSON array. No other text. No markdown.
Return [] for most conversational turns — this is the correct expected response.

Format: [{"content": "User ...", "type": "fact|preference|skill|goal|decision|correction|relationship|event", "score": 0.0, "reasoning": "one sentence"}]"""


# Simpler extraction prompt — used as fallback when the full prompt fails
# Asks only for content + type, no scoring. Score is assigned by DFS scorer.
EXTRACTION_SIMPLE_PROMPT = """Extract facts about the user from this conversation.
Return a JSON array of objects. Return [] if nothing to extract.

ONLY extract stable personal facts: name, job, location, skills, tools, preferences, goals, relationships.
DO NOT extract: questions, thanks, temporary states, general knowledge.

Format: [{"content": "User ...", "type": "fact"}]
Types: fact, preference, skill, goal, decision, correction, relationship, event

Examples:
User: "I work at IISc on XAI" -> [{"content": "User works at IISc on XAI", "type": "fact"}]
User: "thanks ok" -> []
User: "How does SHAP work?" -> []

Return ONLY the JSON array, nothing else."""


# ---------------------------------------------------------------------------
# Linguistic heuristics
# ---------------------------------------------------------------------------

# Each entry: (pattern, delta, label)
# Positive delta = boost, negative = penalty
_LINGUISTIC_RULES: list[tuple[str, float, str]] = [
    # Strong emphasis — user is signalling this matters
    (r"\b(always|never)\b", +0.10, "always/never"),
    (r"\b(hate|love|despise|adore)\b", +0.08, "strong emotion"),
    (r"\b(absolutely|definitely|certainly)\b", +0.06, "strong certainty"),
    (r"\b(important|critical|crucial|essential)\b", +0.12, "importance marker"),
    (r"\bremember\s+this\b", +0.15, "explicit remember"),
    (r"\bby\s+the\s+way\b", +0.06, "volunteered aside"),
    (r"!{1,3}", +0.04, "exclamation"),

    # Corrections — very strong signal
    (r"\b(actually|correction|wrong|incorrect|no,?\s+i)\b", +0.12, "correction signal"),
    (r"\bi\s+meant\b", +0.08, "self-correction"),

    # Hedging — user is uncertain, lower signal
    (r"\b(maybe|perhaps|might|could\s+be|possibly)\b", -0.08, "hedging"),
    (r"\b(sometimes|occasionally|usually|often)\b", -0.05, "frequency hedge"),
    (r"\b(think|guess|suppose|believe)\b", -0.04, "belief hedge"),
    (r"\b(not\s+sure|unsure|i\s+don.t\s+know)\b", -0.10, "uncertainty"),
]

# Transient temporal signals — cap the score if found
_TRANSIENT_PATTERNS: list[tuple[str, str]] = [
    (r"\bright\s+now\b", "right now"),
    (r"\bcurrently\b", "currently"),
    (r"\btoday\b", "today"),
    (r"\bthis\s+(week|morning|afternoon|evening|hour)\b", "this week/day"),
    (r"\bat\s+the\s+moment\b", "at the moment"),
    (r"\bjust\s+(started|began|finished|realized)\b", "just started/finished"),
    (r"\bfor\s+now\b", "for now"),
    (r"\btemporarily\b", "temporarily"),
    (r"\bdebugging\b", "debugging (transient task)"),
    (r"\bstuck\s+on\b", "stuck on (transient)"),
]

TRANSIENT_SCORE_CAP = 0.65  # Max score for any memory flagged as transient


def _apply_linguistic_heuristics(content: str, source_text: str) -> tuple[float, list[str], list[str], list[str]]:
    """
    Scan the memory content and its source context for linguistic signals.
    Returns: (delta, emphasis_hits, hedging_hits, transient_hits)
    """
    # Check both the extracted content and the raw source text for signals
    combined = (content + " " + (source_text or "")).lower()

    delta = 0.0
    emphasis_hits = []
    hedging_hits = []

    for pattern, adjustment, label in _LINGUISTIC_RULES:
        if re.search(pattern, combined, re.IGNORECASE):
            delta += adjustment
            if adjustment > 0:
                emphasis_hits.append(label)
            else:
                hedging_hits.append(label)

    # Cap the linguistic delta to ±0.15 so it can never dominate the LLM score
    delta = max(-0.15, min(+0.15, delta))

    # Transient detection (on content only — not source, to avoid false positives)
    transient_hits = []
    for pattern, label in _TRANSIENT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            transient_hits.append(label)

    return delta, emphasis_hits, hedging_hits, transient_hits




# ---------------------------------------------------------------------------
# Semantic dedup — keyword overlap check before writing
# ---------------------------------------------------------------------------

def _keyword_overlap(a: str, b: str) -> float:
    """
    Fast proxy for semantic similarity using keyword overlap.
    Strips stopwords and measures Jaccard similarity on content words.
    Used as a cheap pre-check before committing a write.

    Returns a score in [0, 1].
    """
    _STOPWORDS = frozenset({
        'a','an','the','is','are','was','were','be','been','being',
        'have','has','had','do','does','did','will','would','shall',
        'should','may','might','can','could','i','my','me','we','our',
        'user','their','they','it','its','this','that','these','those',
        'and','or','but','in','on','at','to','for','of','with','by',
        'from','as','into','about','always','never','not',
    })
    _PUNCT = re.compile(r'[^\w\s-]')
    def _tokens(s: str) -> set:
        return {
            _PUNCT.sub('', w.lower())
            for w in s.split()
            if len(w) > 2 and _PUNCT.sub('', w.lower()) not in _STOPWORDS
        }
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)   # Jaccard

KEYWORD_OVERLAP_THRESHOLD = 0.55   # above this = likely a rephrasing


def _is_rephrasing(
    candidate: str,
    existing_chunks,
    top_n: int = 15,
) -> tuple[bool, str]:
    """
    Check whether candidate is a rephrasing of something already stored.
    Uses fast keyword-overlap Jaccard similarity — no embedding call needed.
    Returns (is_rephrasing, matched_content).
    """
    # Only compare against chunks of the same broad type
    # (a rephrase of a skill fact won't match an event)
    for chunk in existing_chunks[:top_n]:
        if chunk.superseded_by:
            continue
        score = _keyword_overlap(candidate, chunk.content)
        if score >= KEYWORD_OVERLAP_THRESHOLD:
            return True, chunk.content
    return False, ""

# ---------------------------------------------------------------------------
# Pre-filter — fast check before calling the LLM
# ---------------------------------------------------------------------------

_FILLER_PATTERN = re.compile(
    r"^\s*(thanks?\.?|thank\s+you|ok(ay)?|sure|got\s+it|i\s+see|understood|"
    r"makes?\s+sense|great|perfect|good|nice|cool|wow|yes|no|yep|nope|right|"
    r"exactly|agreed|sounds?\s+good|alright|fine|noted|awesome|excellent|"
    r"i\s+understand|that\s+(makes?\s+sense|helps?|works?|is\s+(clear|helpful))"
    r")\s*[.!]?\s*$",
    re.IGNORECASE,
)

_QUESTION_PATTERN = re.compile(
    r"^\s*(what|how|why|when|where|who|which|can\s+you|could\s+you|would\s+you|how\s+do\s+i|how\s+can\s+i|what\s+should\s+i|"
    r"do\s+you|is\s+there|are\s+there|does\s+it|will\s+you|should\s+i)\b",
    re.IGNORECASE,
)

_IDENTITY_MARKERS = re.compile(
    r"\b(i\s+am|i\'m|my\s+(name|job|role|work|company|school|university|project|phd|advisor|"
    r"goal|preference|advisor|team|language|framework|tool|stack)|"
    r"i\s+(use|work|study|build|hate|love|always|never|prefer|switched|decided|"
    r"chose|graduated|live|moved))\b",
    re.IGNORECASE,
)

MIN_WORDS_FOR_EXTRACTION = 8


def _should_extract(user_message: str) -> tuple[bool, str]:
    """
    Fast pre-check before calling the LLM for extraction.
    Returns (should_extract, reason_if_skipped).

    Skips if:
      - Message is conversational filler or acknowledgement
      - Message is a pure question with no self-disclosure
      - Message is too short with no identity markers
    """
    msg = user_message.strip()
    if not msg:
        return False, "empty message"

    if _FILLER_PATTERN.match(msg):
        return False, "conversational filler"

    word_count = len(msg.split())

    if word_count < MIN_WORDS_FOR_EXTRACTION:
        if not _IDENTITY_MARKERS.search(msg):
            return False, f"too short ({word_count} words) with no identity marker"

    if _QUESTION_PATTERN.match(msg):
        if not _IDENTITY_MARKERS.search(msg):
            return False, "pure question with no self-disclosure"

    return True, ""

# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class MemoryExtractor:
    """
    Extracts and scores memorizable chunks from conversation turns.

    Scoring mode is controlled by the scorer parameter:

      scorer=None (default)
        Uses DeterministicScorer — fully model-agnostic, zero LLM calls,
        deterministic. Recommended for production and for weak models (LLaVA, etc.)

      scorer="llm" (legacy)
        Uses the LLM to score inline. Score changes if you change the model.
        Only useful if your LLM reliably returns JSON with score fields.

    The full scoring breakdown is stored in chunk.metadata["dfs"] (deterministic)
    or chunk.metadata["scoring"] (LLM) for inspection.

    Usage:
        # Deterministic (default, recommended)
        extractor = MemoryExtractor(llm_adapter)
        extractor = MemoryExtractor(llm_adapter, scorer=DeterministicScorer())

        # LLM-based (legacy)
        extractor = MemoryExtractor(llm_adapter, scorer="llm")

        chunks = extractor.extract(conversation, session_id="abc")
        for chunk in chunks:
            print(chunk.metadata.get("dfs", {}).get("summary", ""))
    """

    def __init__(self, llm_adapter: LLMAdapter, scorer=None):
        self._llm = llm_adapter
        # scorer=None or scorer=DeterministicScorer() -> use DFS
        # scorer="llm" -> use LLM scoring (legacy)
        if scorer == "llm":
            self._scorer = "llm"
        elif scorer is None:
            self._scorer = DeterministicScorer()
        else:
            self._scorer = scorer  # custom DeterministicScorer(weights=...)

    def extract(
        self,
        conversation: list[dict],
        session_id: Optional[str] = None,
        max_chunks: int = 10,
        existing_chunks: Optional[list] = None,
    ) -> list[MemoryChunk]:
        """
        Extract and score memories from a conversation slice.

        Args:
            conversation: list of {"role": "user"/"assistant", "content": "..."}
            session_id: identifier for the current session
            max_chunks: maximum memories to extract per call
            existing_chunks: current store contents for dedup check (optional)

        Returns:
            list of MemoryChunk objects with LLM+heuristic scores set
        """
        if not conversation:
            return []

        # Pre-filter: skip extraction for filler, questions, and short messages
        user_messages = [
            m.get("content", "") for m in conversation
            if m.get("role") == "user"
        ]
        last_user_msg = user_messages[-1] if user_messages else ""
        should, reason = _should_extract(last_user_msg)
        if not should:
            logger.debug("Skipping extraction: %s  msg=%s", reason, last_user_msg[:60])
            return []

        conv_text = self._format_for_extraction(conversation)

        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": f"Extract and score memories from:\n\n{conv_text}"}],
                system=EXTRACTION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1500,
            )
            raw = self._parse_json(response)

            # Fallback: if full prompt returned nothing, retry with simpler prompt
            # This handles weak models (llama3.2, LLaVA) that can't follow
            # the complex scoring format but can do basic extraction
            if not raw:
                logger.debug("Full extraction prompt returned []. Retrying with simple prompt.")
                simple_response = self._llm.chat(
                    messages=[{"role": "user", "content": f"Extract facts:\n\n{conv_text}"}],
                    system=EXTRACTION_SIMPLE_PROMPT,
                    temperature=0.1,
                    max_tokens=512,
                )
                raw = self._parse_json(simple_response)
                if raw:
                    logger.debug("Simple prompt recovered %d items.", len(raw))

        except Exception as e:
            logger.warning(f"Memory extraction failed: {e}")
            return []

        chunks = []
        for item in raw[:max_chunks]:
            if not isinstance(item, dict) or "content" not in item:
                continue

            content = item["content"].strip()
            if not content:
                continue

            # ── Content quality filters ───────────────────────────────────────

            # 1. Reject fragments — must be at least 5 words to be a standalone memory
            if len(content.split()) < 5:
                logger.debug("Rejecting fragment (too short): %s", content)
                continue

            # 2. Normalise to "User ..." form
            #    llama3.2 sometimes returns raw first-person ("I work at IISc")
            #    or bare phrases ("Video Analytics Lab") instead of "User ..."
            if not content.lower().startswith("user"):
                # First-person → third-person
                first_person = re.compile(
                    r"^(i am|i'm|i use|i work|i hate|i love|i always|i never|"
                    r"i prefer|i have|i study|i build|i switched|i decided|"
                    r"i graduated|i joined|i live|i moved|my name|my advisor|"
                    r"my goal|my phd|my role|my job|my team|my project)",
                    re.IGNORECASE,
                )
                if first_person.match(content):
                    # Ordered replacement: longest patterns first
                    content = re.sub(r"^I'm\b",  "User is", content, flags=re.IGNORECASE)
                    content = re.sub(r"^I am\b",  "User is", content, flags=re.IGNORECASE)
                    content = re.sub(r"^My\b",    "User's",  content, flags=re.IGNORECASE)
                    content = re.sub(r"^I\b",     "User",    content, flags=re.IGNORECASE)
                    logger.debug("Normalised to User-form: %s", content)
                elif len(content.split()) < 8:
                    logger.debug("Rejecting non-User fragment: %s", content)
                    continue
            # ── End content quality filters ───────────────────────────────────

            # Handle explicit forget requests
            if content.startswith("FORGET:"):
                topic = content[7:].strip()
                logger.info(f"Forget request: {topic}")
                continue

            mem_type_str = item.get("type", "generic").lower()
            if mem_type_str not in VALID_TYPES:
                mem_type_str = "generic"

            # --- Score computation ---
            mem_type = MemoryType(mem_type_str)

            if self._scorer == "llm":
                # Legacy LLM-based scoring
                scoring = self._compute_score(
                    content=content,
                    source_text=conv_text,
                    llm_score=item.get("score"),
                    llm_reasoning=item.get("reasoning", ""),
                    mem_type=mem_type,
                )
                final_score = scoring.final_score
                metadata = {"scoring": {
                    "llm_score": scoring.llm_score,
                    "llm_reasoning": scoring.llm_reasoning,
                    "linguistic_delta": scoring.linguistic_delta,
                    "temporal_cap": scoring.temporal_cap,
                    "final_score": scoring.final_score,
                    "emphasis_hits": scoring.emphasis_hits,
                    "hedging_hits": scoring.hedging_hits,
                    "transient_hits": scoring.transient_hits,
                    "summary": scoring.summary(),
                }}
                score_log = scoring.summary()
            else:
                # Deterministic scoring (default) — model-agnostic
                # Extract prior assistant turn for structural analysis
                prior_turn = None
                turns = conversation[-6:]
                for i, msg in enumerate(turns):
                    if msg.get("role") == "user" and msg.get("content", "")[:100] in conv_text:
                        if i > 0 and turns[i-1].get("role") == "assistant":
                            prior_turn = turns[i-1].get("content", "")
                        break

                final_score, fv = self._scorer.score(
                    content=content,
                    source_message=conv_text,
                    prior_assistant_turn=prior_turn,
                    turn_index=len(conversation),
                    total_turns=len(conversation),
                )
                metadata = {"dfs": fv.to_dict(self._scorer.weights)}
                score_log = fv.summary(self._scorer.weights)

            # Keyword-overlap dedup: skip if this is a rephrasing of something stored
            if existing_chunks:
                is_rephrase, matched = _is_rephrasing(content, existing_chunks)
                if is_rephrase:
                    logger.debug(
                        "Dedup: skipping rephrasing. new=%s  existing=%s",
                        content[:80], matched[:80],
                    )
                    continue

            chunk = MemoryChunk(
                content=content,
                layer=MemoryLayer.EPISODIC,
                memory_type=mem_type,
                score=final_score,
                session_id=session_id,
                source_text=conv_text[:500],
                created_at=time.time(),
                last_accessed=time.time(),
                metadata=metadata,
            )
            chunks.append(chunk)
            logger.debug(f"Extracted [{mem_type_str}] score={final_score:.2f}: {content[:80]}")
            logger.debug(f"  Scoring: {score_log}")

        return chunks

    def _compute_score(
        self,
        content: str,
        source_text: str,
        llm_score: Optional[float],
        llm_reasoning: str,
        mem_type: MemoryType,
    ) -> ScoringResult:
        """
        Combine LLM score + linguistic heuristics + temporal cap into a final score.

        If the LLM didn't return a score (malformed response), fall back to
        a type-based prior so we always have something reasonable.
        """
        # --- Layer A: LLM score ---
        if llm_score is not None:
            try:
                base = float(llm_score)
                base = max(0.0, min(1.0, base))  # Clamp to [0, 1]
            except (TypeError, ValueError):
                base = self._type_prior(mem_type)
                llm_reasoning = f"[fallback to type prior] {llm_reasoning}"
        else:
            base = self._type_prior(mem_type)
            llm_reasoning = f"[no LLM score, type prior used] {llm_reasoning}"

        # --- Layer B: Linguistic heuristics ---
        ling_delta, emphasis_hits, hedging_hits, transient_hits = _apply_linguistic_heuristics(
            content, source_text
        )
        adjusted = base + ling_delta
        adjusted = max(0.0, min(1.0, adjusted))

        # --- Layer C: Temporal stability cap ---
        temporal_cap = None
        if transient_hits:
            if adjusted > TRANSIENT_SCORE_CAP:
                temporal_cap = TRANSIENT_SCORE_CAP
                adjusted = TRANSIENT_SCORE_CAP

        return ScoringResult(
            llm_score=base,
            llm_reasoning=llm_reasoning,
            linguistic_delta=ling_delta,
            temporal_cap=temporal_cap,
            final_score=round(adjusted, 3),
            emphasis_hits=emphasis_hits,
            hedging_hits=hedging_hits,
            transient_hits=transient_hits,
        )

    def _type_prior(self, memory_type: MemoryType) -> float:
        """
        Fallback scores used only when the LLM fails to return a score.
        These are intentionally conservative — we'd rather under-score and
        let reinforcement lift important memories than over-score everything.
        """
        priors = {
            MemoryType.CORRECTION: 0.85,
            MemoryType.GOAL: 0.75,
            MemoryType.DECISION: 0.70,
            MemoryType.SKILL: 0.65,
            MemoryType.PREFERENCE: 0.62,
            MemoryType.FACT: 0.58,
            MemoryType.RELATIONSHIP: 0.58,
            MemoryType.EVENT: 0.42,
            MemoryType.GENERIC: 0.35,
        }
        return priors.get(memory_type, 0.50)

    def _format_for_extraction(self, conversation: list[dict]) -> str:
        """Format conversation for the extraction prompt."""
        lines = []
        for msg in conversation[-6:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            lines.append(f"{role}: {content[:500]}")
        return "\n".join(lines)

    def _parse_json(self, text: str) -> list:
        """
        Robustly parse JSON from LLM response.
        Tries multiple strategies before giving up.
        """
        text = text.strip()
        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*", "", text).strip()

        # Strategy 1: direct array parse
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 2: fix common issues and retry
        if match:
            fixed = match.group()
            # Remove trailing commas before ] or }
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            # Replace smart quotes
            fixed = fixed.replace("\u201c", "\"").replace("\u201d", "\"")
            fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        # Strategy 3: extract individual objects and build array manually
        # Handles cases where the model returns objects on separate lines
        objects = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
        if objects:
            parsed = []
            for obj in objects:
                obj = re.sub(r",\s*}", "}", obj)
                try:
                    parsed.append(json.loads(obj))
                except json.JSONDecodeError:
                    # Strategy 4: extract key-value pairs manually
                    item = {}
                    content_m = re.search(r'"content"\s*:\s*"([^"]+)"', obj)
                    type_m    = re.search(r'"type"\s*:\s*"([^"]+)"', obj)
                    score_m   = re.search(r'"score"\s*:\s*([0-9.]+)', obj)
                    if content_m:
                        item["content"] = content_m.group(1)
                        item["type"]    = type_m.group(1) if type_m else "fact"
                        if score_m:
                            item["score"] = float(score_m.group(1))
                        parsed.append(item)
            if parsed:
                return parsed

        logger.warning("JSON parse failed after all strategies. Response was: %s",
                       text[:200])
        return []