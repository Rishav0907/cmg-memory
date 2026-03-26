"""
CMG Memory System — Deterministic Feature Scorer (DFS)
========================================================
A fully model-agnostic importance scorer.
Produces the same score regardless of which LLM you use,
because it operates entirely on observable text features.

Five feature families:

  F1  Lexical salience      Emphasis/hedging/correction vocabulary
  F2  Temporal stability    State vs event, tense, time anchors
  F3  Entity density        Named tools, people, orgs, technical terms
  F4  Structural position   Volunteered vs elicited, focus ratio
  F5  Store novelty         Cosine distance from existing memories

Final score = w1·F1 + w2·F2 + w3·F3 + w4·F4 + w5·F5  (all in [0,1])

Default weights sum to 1.0 and are tunable without retraining anything.
An optional calibration pass can learn weights from labelled examples.

No LLM calls. No API keys. Fully deterministic.
Identical score on OpenAI GPT-4o, Claude, llama3, LLaVA, anything.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .types import MemoryChunk, MemoryType


# ─────────────────────────────────────────────────────────────────────────────
# Feature result — transparent and debuggable
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureVector:
    """
    The five feature scores for a single memory candidate.
    All values are in [0, 1].
    Stored in chunk.metadata["dfs"] for inspection.
    """
    f1_lexical:    float = 0.5   # Lexical salience
    f2_temporal:   float = 0.5   # Temporal stability
    f3_entity:     float = 0.5   # Entity density
    f4_structural: float = 0.5   # Structural position
    f5_novelty:    float = 0.5   # Store novelty

    # Signals found (for debugging)
    emphasis_signals:   list[str] = field(default_factory=list)
    hedging_signals:    list[str] = field(default_factory=list)
    correction_signals: list[str] = field(default_factory=list)
    transient_signals:  list[str] = field(default_factory=list)
    stable_signals:     list[str] = field(default_factory=list)
    entities_found:     list[str] = field(default_factory=list)

    def weighted_score(self, weights: "ScorerWeights") -> float:
        return (
            weights.w1 * self.f1_lexical
            + weights.w2 * self.f2_temporal
            + weights.w3 * self.f3_entity
            + weights.w4 * self.f4_structural
            + weights.w5 * self.f5_novelty
        )

    def summary(self, weights: "ScorerWeights") -> str:
        final = self.weighted_score(weights)
        return (
            f"F1(lex)={self.f1_lexical:.2f} "
            f"F2(tmp)={self.f2_temporal:.2f} "
            f"F3(ent)={self.f3_entity:.2f} "
            f"F4(str)={self.f4_structural:.2f} "
            f"F5(nov)={self.f5_novelty:.2f} "
            f"→ {final:.3f}"
        )

    def to_dict(self, weights: "ScorerWeights") -> dict:
        return {
            "f1_lexical":         round(self.f1_lexical, 3),
            "f2_temporal":        round(self.f2_temporal, 3),
            "f3_entity":          round(self.f3_entity, 3),
            "f4_structural":      round(self.f4_structural, 3),
            "f5_novelty":         round(self.f5_novelty, 3),
            "final_score":        round(self.weighted_score(weights), 3),
            "emphasis_signals":   self.emphasis_signals,
            "hedging_signals":    self.hedging_signals,
            "correction_signals": self.correction_signals,
            "transient_signals":  self.transient_signals,
            "stable_signals":     self.stable_signals,
            "entities_found":     self.entities_found,
            "summary":            self.summary(weights),
        }


@dataclass
class ScorerWeights:
    """
    Weights for combining the five features.
    Default values reflect the relative importance of each dimension.
    Sum does not need to equal 1.0 — output is normalised.
    """
    w1: float = 0.25   # Lexical salience      — strong direct signal
    w2: float = 0.30   # Temporal stability    — most predictive of long-term value
    w3: float = 0.20   # Entity density        — specificity proxy
    w4: float = 0.15   # Structural position   — conversational signal
    w5: float = 0.10   # Store novelty         — redundancy penalty

    def to_dict(self) -> dict:
        return {"w1": self.w1, "w2": self.w2, "w3": self.w3,
                "w4": self.w4, "w5": self.w5}

    @classmethod
    def from_dict(cls, d: dict) -> "ScorerWeights":
        return cls(**{k: v for k, v in d.items() if k in ("w1","w2","w3","w4","w5")})


# ─────────────────────────────────────────────────────────────────────────────
# F1 — Lexical salience
# ─────────────────────────────────────────────────────────────────────────────

# (pattern, delta, category, label)
_LEXICAL_RULES: list[tuple[str, float, str, str]] = [
    # Strong positive — user is signalling this matters
    (r"\bremember\s+this\b",                      +0.25, "emphasis", "explicit remember"),
    (r"\b(important|critical|crucial|essential)\b",+0.20, "emphasis", "importance marker"),
    (r"\b(always|never)\b",                        +0.15, "emphasis", "always/never"),
    (r"\b(hate|love|despise|adore)\b",             +0.12, "emphasis", "strong emotion"),
    (r"\b(absolutely|definitely|certainly)\b",     +0.10, "emphasis", "certainty"),
    (r"\bby\s+the\s+way\b",                        +0.08, "emphasis", "volunteered aside"),
    (r"!{1,3}",                                    +0.06, "emphasis", "exclamation"),

    # Corrections — explicit signal that old information was wrong
    (r"\b(correction|correcting|corrected)\b",     +0.20, "correction", "explicit correction"),
    (r"\bactually[,\s]",                            +0.15, "correction", "actually"),
    (r"\bno[,\s].{0,15}(i|my|the)\b",              +0.12, "correction", "direct negation"),
    (r"\bi\s+meant\b",                              +0.10, "correction", "self-correction"),
    (r"\bnot\s+(that|what\s+i\s+said)\b",           +0.10, "correction", "negation of prior"),

    # Hedging — user is uncertain, weaker signal
    (r"\b(maybe|perhaps|possibly|conceivably)\b",  -0.15, "hedging", "possibility hedge"),
    (r"\b(might|could\s+be|may\s+be)\b",           -0.10, "hedging", "modal hedge"),
    (r"\b(sometimes|occasionally|often|usually)\b",-0.08, "hedging", "frequency hedge"),
    (r"\b(i\s+think|i\s+guess|i\s+suppose)\b",    -0.08, "hedging", "belief hedge"),
    (r"\b(not\s+sure|unsure|unclear)\b",           -0.12, "hedging", "uncertainty"),
    (r"\b(sort\s+of|kind\s+of|somewhat)\b",        -0.06, "hedging", "vagueness"),
]

def _f1_lexical(content: str, source: str) -> tuple[float, list, list, list]:
    """
    Compute F1 from emphasis/hedging/correction signals.
    Scans both the extracted content and a window of the source text.
    Returns (score_0_to_1, emphasis_hits, hedging_hits, correction_hits).
    """
    combined = (content + " " + source[:300]).lower()
    delta = 0.0
    emphasis, hedging, corrections = [], [], []

    for pattern, adj, category, label in _LEXICAL_RULES:
        if re.search(pattern, combined, re.IGNORECASE):
            delta += adj
            if category == "emphasis":   emphasis.append(label)
            elif category == "hedging":  hedging.append(label)
            elif category == "correction": corrections.append(label)

    # Normalise delta [-0.5, +0.5] → [0, 1] with neutral point at 0.5
    delta = max(-0.5, min(+0.5, delta))
    score = 0.5 + delta
    return round(score, 3), emphasis, hedging, corrections


# ─────────────────────────────────────────────────────────────────────────────
# F2 — Temporal stability
# ─────────────────────────────────────────────────────────────────────────────

# High-stability signals: present simple state, long-term duration vocabulary
_STABLE_PATTERNS: list[tuple[str, float, str]] = [
    (r"\bfor\s+(years?|months?|decades?|a\s+long\s+time)\b", +0.30, "long duration"),
    (r"\b(since|from)\s+\d{4}\b",                             +0.25, "year anchor"),
    (r"\balways\s+(have|has|been|use[sd]?)\b",                +0.20, "habitual always"),
    (r"\b(i\s+am|i'm)\s+a\b",                                 +0.20, "identity statement"),
    (r"\b(my\s+job|my\s+role|my\s+position|my\s+title)\b",   +0.20, "job/role"),
    (r"\b(specialise|specialize|expert|expertise)\b",          +0.15, "expertise claim"),
    (r"\b(generally|in\s+general|as\s+a\s+rule)\b",           +0.12, "general habit"),
    (r"\bpermanently\b",                                        +0.20, "permanent"),
    # Employment and location — stable facts even with "currently"
    (r"\bwork(ing)?\s+(at|for|in)\s+\w",                     +0.25, "employment fact"),
    (r"\b(based\s+(in|out\s+of)|living\s+in|located\s+in)\s+\w", +0.25, "location fact"),
    (r"\bas\s+(a|an)\s+\w+\s+(scientist|engineer|researcher|developer|manager|analyst|designer)", +0.20, "job title"),
]

# Transient signals: present progressive, ephemeral time anchors
_TRANSIENT_PATTERNS: list[tuple[str, float, str]] = [
    (r"\bright\s+now\b",                                       -0.35, "right now"),
    (r"\b(currently|at\s+the\s+moment|at\s+present)\b",       -0.30, "currently"),
    (r"\btoday\b",                                             -0.25, "today"),
    (r"\bthis\s+(week|morning|afternoon|evening|hour|minute)\b",-0.25,"this week/time"),
    (r"\bjust\s+(started|began|finished|completed|realized)\b",-0.20, "just did"),
    (r"\btemporarily|for\s+now|for\s+the\s+time\s+being\b",   -0.20, "temporary"),
    (r"\bdebugging\b",                                         -0.15, "debugging (transient)"),
    (r"\bstuck\s+(on|at|with)\b",                             -0.15, "stuck on (transient)"),
    (r"\bwaiting\s+for\b",                                     -0.10, "waiting (transient)"),
    (r"\bgoing\s+to\s+(try|attempt|test)\b",                  -0.10, "future trial"),
]

# Patterns where "currently" signals a stable state (job, role, location)
# rather than a transient action (debugging, working on a bug)
_STABLE_CURRENTLY_PATTERNS = [
    re.compile(r"currently\s+work(ing)?\s+(at|for|in|as)", re.IGNORECASE),
    re.compile(r"currently\s+(based|living|located)\s+(in|at|out\s+of)", re.IGNORECASE),
    re.compile(r"currently\s+(a|an|the)\s+\w+\s+(at|for|in)", re.IGNORECASE),
    re.compile(r"working\s+at\s+\w", re.IGNORECASE),
    re.compile(r"(based\s+out\s+of|based\s+in)\s+\w", re.IGNORECASE),
]


def _f2_temporal(content: str, source: str) -> tuple[float, list, list]:
    """
    Compute F2 from temporal vocabulary.
    Returns (score_0_to_1, stable_signals, transient_signals).
    """
    combined = (content + " " + source[:300]).lower()
    delta = 0.0
    stable_hits, transient_hits = [], []

    # Check if "currently" is used in a stable employment/location context
    # If so, treat the whole message as stable (skip transient penalty for "currently")
    is_stable_currently = any(p.search(combined) for p in _STABLE_CURRENTLY_PATTERNS)

    for pattern, adj, label in _STABLE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            delta += adj
            stable_hits.append(label)

    for pattern, adj, label in _TRANSIENT_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            # Skip the "currently" penalty if it's used in a stable employment context
            if label == "currently" and is_stable_currently:
                stable_hits.append("currently (employment context)")
                delta += 0.15  # treat as stable signal instead
                continue
            delta += adj
            transient_hits.append(label)

    # Tense detection: present progressive (-) vs present simple (+)
    # "I am building" → transient; "I use" → stable
    prog_matches = len(re.findall(r"\bam\s+\w+ing\b|\bare\s+\w+ing\b|\bis\s+\w+ing\b",
                                  combined, re.IGNORECASE))
    simple_matches = len(re.findall(r"\b(i\s+use|i\s+work|i\s+build|i\s+do|i\s+have)\b",
                                    combined, re.IGNORECASE))
    if prog_matches > simple_matches:
        delta -= 0.10
        transient_hits.append("present progressive tense")
    elif simple_matches > prog_matches:
        delta += 0.08
        stable_hits.append("present simple (habitual)")

    delta = max(-0.5, min(+0.5, delta))
    score = 0.5 + delta
    return round(score, 3), stable_hits, transient_hits


# ─────────────────────────────────────────────────────────────────────────────
# F3 — Entity density
# ─────────────────────────────────────────────────────────────────────────────

# Technical domain vocabulary — tools, frameworks, datasets, methods
# Grouped by domain for extensibility
_TECH_VOCAB: set[str] = {
    # DL frameworks
    "pytorch", "tensorflow", "keras", "jax", "mxnet", "paddle", "onnx",
    # CV tools
    "opencv", "pillow", "albumentations", "torchvision",
    # XAI methods
    "grad-cam", "gradcam", "shap", "lime", "captum", "integrated gradients",
    "attention rollout", "rise", "scorecam", "eigencam",
    # Architectures
    "resnet", "vgg", "alexnet", "inception", "efficientnet", "vit", "swin",
    "mask2former", "maskformer", "detr", "sam", "clip", "dino",
    "transformer", "bert", "gpt", "llama", "mistral", "gemma",
    # Datasets
    "imagenet", "coco", "cityscapes", "bdd100k", "tusimple", "culane",
    "smiyc", "fishyscapes", "kitti", "nuscenes", "waymo",
    "mnist", "cifar", "ade20k",
    # Langs & tools
    "python", "rust", "go", "c++", "julia", "scala", "typescript",
    "docker", "kubernetes", "git", "vim", "tmux", "cuda", "tensorrt",
    # Cloud/infra
    "aws", "gcp", "azure", "wandb", "mlflow", "huggingface",
    # Orgs/institutions (common)
    "google", "anthropic", "openai", "deepmind", "meta", "microsoft",
    "iit", "iisc", "iitb", "iitm", "nit", "iiser",
    "stanford", "mit", "cmu", "oxford", "cambridge",
}

_ORG_PATTERNS = [
    r"\b[A-Z][a-z]+\s+(University|Institute|College|Lab|Research|Corp|Inc|Ltd)\b",
    r"\b(University|Institute)\s+of\s+[A-Z][a-z]+\b",
    r"\bPhD|M\.?S\.?|B\.?Tech|B\.?E\b",
]

_ROLE_PATTERNS = [
    r"\b(engineer|researcher|scientist|developer|architect|analyst|manager|lead|intern)\b",
    r"\b(professor|postdoc|grad\s*student|doctoral)\b",
]

def _f3_entity(content: str) -> tuple[float, list[str]]:
    """
    Compute F3 from named entity and technical term density.
    More specific, named content → higher score.
    """
    content_lower = content.lower()
    found: list[str] = []

    # Technical vocabulary match
    for term in _TECH_VOCAB:
        if term in content_lower:
            found.append(term)

    # Organisation patterns
    for pattern in _ORG_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for m in matches:
            label = m if isinstance(m, str) else " ".join(m)
            if label not in found:
                found.append(f"org:{label}")

    # Role/title patterns
    for pattern in _ROLE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            found.append("role_mention")
            break

    # Capitalised proper-noun runs (crude NER)
    cap_runs = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b", content)
    for run in cap_runs:
        if len(run) > 3 and run not in ("User", "The", "This", "That"):
            label = f"proper:{run}"
            if label not in found:
                found.append(label)

    # Numerical specifics (version numbers, quantities, years, percentages)
    numerics = re.findall(
        r"\b(\d+\.\d+|\d{4}|\d+[kKmMbB]|\d+%|\d+x)\b", content
    )
    for n in numerics[:3]:
        found.append(f"num:{n}")

    # Score: log-scale entity count, normalised
    unique = list(dict.fromkeys(found))  # deduplicate preserving order
    raw_count = len(unique)
    # 0 entities → 0.2, 3 entities → ~0.65, 6+ entities → ~0.9
    score = 0.2 + 0.7 * (1 - math.exp(-raw_count / 4.0))
    return round(score, 3), unique[:10]  # cap display at 10


# ─────────────────────────────────────────────────────────────────────────────
# F4 — Structural position
# ─────────────────────────────────────────────────────────────────────────────

def _f4_structural(
    content: str,
    source_message: str,
    prior_assistant_turn: Optional[str],
    turn_index: int,
    total_turns: int,
) -> float:
    """
    Compute F4 from conversational structure signals.

    Key signals:
    - Focus ratio: how much of the source message is about this memory?
      High ratio = user dedicated the message to this fact = volunteered
    - Elicitation: was the prior assistant turn a direct question?
      If yes, user was answering — lower score
    - Turn position: early turns often carry more identifying info
    - Message length: very short messages after questions = elicited answer
    """
    score = 0.5

    # 1. Focus ratio: extracted content length / source message length
    source_len  = max(len(source_message.split()), 1)
    content_len = max(len(content.split()), 1)
    focus_ratio = min(1.0, content_len / source_len)

    # High focus: most of the message was about this → volunteered
    if focus_ratio > 0.6:
        score += 0.15
    elif focus_ratio < 0.2:
        score -= 0.05  # Mentioned in passing

    # 2. Elicitation detection: was user answering a direct question?
    if prior_assistant_turn:
        prior_lower = prior_assistant_turn.strip().lower()
        is_question = (
            prior_lower.endswith("?") or
            re.search(r"\b(what|which|where|who|when|how|do you|are you|have you)\b",
                      prior_lower[-100:], re.IGNORECASE) is not None
        )
        if is_question:
            score -= 0.15  # Elicited answer
        # Explicit memory prompt from assistant boosts it
        if re.search(r"\b(tell\s+me\s+about\s+yourself|what\s+do\s+you\s+do|introduce)\b",
                     prior_lower, re.IGNORECASE):
            score -= 0.05  # Still elicited, just a broader question

    # 3. "By the way" / volunteered aside: already caught in F1 but structural bonus here
    if re.search(r"\bby\s+the\s+way\b|\boh\s+(and|also)\b|\bjust\s+so\s+you\s+know\b",
                 source_message, re.IGNORECASE):
        score += 0.12

    # 4. Correction structure: user contradicting a prior turn = strong signal
    if re.search(r"\b(no|wait|actually|correction)\b.{0,30}(i|my|the|it)\b",
                 source_message[:200], re.IGNORECASE):
        score += 0.15

    # 5. First-person strong declarations: "I am a X", "My Y is Z"
    strong_declaration = re.search(
        r"\b(i\s+am|i'm)\s+a\s+\w+|my\s+(name|job|role|title|company|university)\s+is\b",
        source_message, re.IGNORECASE
    )
    if strong_declaration:
        score += 0.10

    return round(max(0.0, min(1.0, score)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# F5 — Store novelty
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _f5_novelty(
    candidate_embedding: Optional[list[float]],
    existing_chunks: list[MemoryChunk],
    top_k: int = 5,
) -> float:
    """
    Compute F5: how novel is this memory relative to what's already stored?

    High similarity to existing chunks → low novelty → lower score
    (redundant information is less valuable to store again)

    Low similarity → high novelty → higher score
    (new information is more valuable)

    If no embedding available, returns neutral 0.5.
    """
    if candidate_embedding is None or not existing_chunks:
        return 0.5

    chunks_with_emb = [c for c in existing_chunks if c.embedding]
    if not chunks_with_emb:
        return 0.8  # Nothing stored yet → very novel

    similarities = []
    for chunk in chunks_with_emb:
        if chunk.embedding:
            sim = _cosine_sim(candidate_embedding, chunk.embedding)
            similarities.append(sim)

    if not similarities:
        return 0.8

    # Use max similarity (most similar existing chunk) as the novelty penalty
    max_sim = max(similarities)
    # 0.0 similarity → perfectly novel → score 0.9
    # 1.0 similarity → exact duplicate → score 0.1
    novelty = 0.9 - 0.8 * max_sim
    return round(max(0.0, min(1.0, novelty)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# DeterministicScorer — the public class
# ─────────────────────────────────────────────────────────────────────────────

class DeterministicScorer:
    """
    Computes importance scores using five observable text features.
    Zero LLM calls. Deterministic. Model-agnostic.

    Usage:
        scorer = DeterministicScorer()

        score, features = scorer.score(
            content="User uses Grad-CAM for explainability",
            source_message="I use Grad-CAM for explainability in my research",
            prior_assistant_turn=None,   # or the assistant's previous message
            turn_index=0,
            total_turns=1,
            existing_chunks=[],           # chunks already in the store
            candidate_embedding=None,     # embed(content) if available
        )

        print(f"Score: {score:.3f}")
        print(features.summary(scorer.weights))

    Tuning weights:
        scorer = DeterministicScorer(weights=ScorerWeights(w2=0.40, w1=0.20, ...))

    Calibration from labelled examples:
        scorer.calibrate(labelled_examples)  # see calibrate() docstring
    """

    def __init__(self, weights: Optional[ScorerWeights] = None):
        self.weights = weights or ScorerWeights()

    def score(
        self,
        content: str,
        source_message: str = "",
        prior_assistant_turn: Optional[str] = None,
        turn_index: int = 0,
        total_turns: int = 1,
        existing_chunks: Optional[list[MemoryChunk]] = None,
        candidate_embedding: Optional[list[float]] = None,
    ) -> tuple[float, FeatureVector]:
        """
        Compute importance score for a memory candidate.

        Args:
            content:               The extracted memory text
            source_message:        The user's original message it came from
            prior_assistant_turn:  The assistant's previous turn (for elicitation detection)
            turn_index:            0-indexed position in the conversation
            total_turns:           Total turns in the conversation
            existing_chunks:       Current store contents (for novelty computation)
            candidate_embedding:   Embedding of content (if already computed)

        Returns:
            (final_score: float [0,1], features: FeatureVector)
        """
        existing_chunks = existing_chunks or []

        # Compute each feature
        f1, emphasis, hedging, corrections = _f1_lexical(content, source_message)
        f2, stable, transient             = _f2_temporal(content, source_message)
        f3, entities                       = _f3_entity(content)
        f4                                 = _f4_structural(
            content, source_message, prior_assistant_turn,
            turn_index, total_turns
        )
        f5 = _f5_novelty(candidate_embedding, existing_chunks)

        fv = FeatureVector(
            f1_lexical    = f1,
            f2_temporal   = f2,
            f3_entity     = f3,
            f4_structural = f4,
            f5_novelty    = f5,
            emphasis_signals   = emphasis,
            hedging_signals    = hedging,
            correction_signals = corrections,
            transient_signals  = transient,
            stable_signals     = stable,
            entities_found     = entities,
        )

        final = fv.weighted_score(self.weights)
        final = round(max(0.0, min(1.0, final)), 3)
        return final, fv

    def score_chunk(
        self,
        chunk: MemoryChunk,
        source_message: str = "",
        prior_assistant_turn: Optional[str] = None,
        turn_index: int = 0,
        total_turns: int = 1,
        existing_chunks: Optional[list[MemoryChunk]] = None,
    ) -> MemoryChunk:
        """
        Score a MemoryChunk in-place, setting chunk.score and
        chunk.metadata["dfs"] with the full feature breakdown.
        Returns the chunk.
        """
        final, fv = self.score(
            content=chunk.content,
            source_message=source_message or chunk.source_text or "",
            prior_assistant_turn=prior_assistant_turn,
            turn_index=turn_index,
            total_turns=total_turns,
            existing_chunks=existing_chunks or [],
            candidate_embedding=chunk.embedding,
        )
        chunk.score = final
        chunk.metadata["dfs"] = fv.to_dict(self.weights)
        return chunk

    def calibrate(
        self,
        examples: list[dict],
        learning_rate: float = 0.01,
        epochs: int = 100,
    ) -> ScorerWeights:
        """
        Learn optimal weights from labelled examples using gradient-free
        coordinate descent (no ML framework needed).

        Each example is:
            {
              "content": str,
              "source": str,
              "label": float,   # human-assigned importance 0.0–1.0
              "prior": str,     # optional prior assistant turn
            }

        Example:
            examples = [
                {"content": "User is a senior ML engineer", "source": "I am a senior ML engineer",
                 "label": 0.85, "prior": None},
                {"content": "User is currently debugging", "source": "I am currently debugging",
                 "label": 0.35, "prior": None},
            ]
            new_weights = scorer.calibrate(examples)
        """
        if not examples:
            return self.weights

        import random

        def _mse(weights: ScorerWeights) -> float:
            total = 0.0
            for ex in examples:
                s, _ = self.score(
                    content=ex["content"],
                    source_message=ex.get("source", ""),
                    prior_assistant_turn=ex.get("prior"),
                )
                # Temporarily override weights
                fv_score = FeatureVector()
                _, fv = DeterministicScorer(weights).score(
                    content=ex["content"],
                    source_message=ex.get("source", ""),
                )
                pred = fv.weighted_score(weights)
                total += (pred - ex["label"]) ** 2
            return total / len(examples)

        best = ScorerWeights(
            w1=self.weights.w1, w2=self.weights.w2, w3=self.weights.w3,
            w4=self.weights.w4, w5=self.weights.w5,
        )
        best_loss = _mse(best)

        w_fields = ["w1", "w2", "w3", "w4", "w5"]
        for epoch in range(epochs):
            improved = False
            for field in w_fields:
                for delta in [+learning_rate, -learning_rate]:
                    candidate = ScorerWeights(**{
                        f: (getattr(best, f) + delta if f == field else getattr(best, f))
                        for f in w_fields
                    })
                    # Keep weights non-negative
                    if getattr(candidate, field) < 0:
                        continue
                    loss = _mse(candidate)
                    if loss < best_loss:
                        best = candidate
                        best_loss = loss
                        improved = True
            if not improved:
                break

        self.weights = best
        return best


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a type-prior baseline for comparison
# ─────────────────────────────────────────────────────────────────────────────

def type_prior_score(memory_type: MemoryType) -> float:
    """The old hardcoded type-based prior, kept for A/B comparison."""
    priors = {
        MemoryType.CORRECTION: 0.85,
        MemoryType.GOAL:       0.75,
        MemoryType.DECISION:   0.70,
        MemoryType.SKILL:      0.65,
        MemoryType.PREFERENCE: 0.62,
        MemoryType.FACT:       0.58,
        MemoryType.RELATIONSHIP: 0.58,
        MemoryType.EVENT:      0.50,
        MemoryType.GENERIC:    0.45,
    }
    return priors.get(memory_type, 0.50)