"""
Step 6 — Subjectivity Detection (Hybrid: Rule-Based + Transformer)

Workflow summary:
    0) Deterministic neutral overrides (moderation/rule posts)
    1) Rule-based scoring using VADER + regex heuristics
    2) Transformer inference for ambiguous cases
    3) Context override for borderline neutral-but-opinionated phrasing

Stage 1 rule band:
    score < 0.15  -> "neutral" (skip transformer)
    score > 0.85  -> "opinionated" (skip transformer)
    otherwise     -> pass to transformer stage

Transformer stage (0.15 <= score <= 0.85):
  Primary:   GroNLP/mdebertav3-subjectivity-english
  Fallback:  MoritzLaurer/deberta-v3-large-zeroshot-v2.0

Output:
    record["labels"]["subjectivity"] -> "opinionated" | "neutral"
    scores_buffer entries include source metadata for auditability:
        - label_source = "transformer"           : ambiguous case resolved by model
        - label_source = "context_override"      : model neutral flipped to opinionated
        - label_source = "flair_override"        : forced neutral by moderator-style flair
        - label_source = "mod_content_override"  : forced neutral by moderation regex
        - no label_source field                   : direct Stage 1 rule decision

Score semantics:
    - subjectivity_score is confidence for the final path decision.
    - For deterministic overrides, score is 1.0 to indicate high confidence in
        the override branch, not "opinionated probability".
    - model_subjectivity_score is only included for context_override rows to
        preserve the original transformer confidence before flipping.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

RULE_LOW  = 0.15   # below this → definitely neutral
RULE_HIGH = 0.85   # above this → definitely opinionated
TRANSFORMER_BATCH_SIZE = 64

# Primary model name
SUBJECTIVITY_MODEL = "GroNLP/mdebertav3-subjectivity-english"
ZSC_FALLBACK_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
MAX_TOKEN_LENGTH   = 512
NEUTRAL_FLIP_THRESHOLD = 0.65
MOD_FLAIR_KEYWORDS = ("rule", "announcement", "mod", "reminder")
NEUTRAL_CONTEXT_MAX_CONFIDENCE = 0.82
NEUTRAL_CONTEXT_MIN_RULE_SCORE = 0.35
_MOD_POST_RE = re.compile(
    r'\b(community rule|no unapproved|mod(?:erator)? reminder|subreddit rule)\b',
    re.IGNORECASE,
)

# ── VADER + Subjectivity Heuristics ──────────────────────────────────────────

# Strong subjectivity signals — presence of these pushes score toward opinionated
_STRONG_SUBJECTIVE_PATTERNS = [
    r'\b(i think|i feel|i believe|i find|i noticed|i prefer|imo|imho|ngl|tbh|in my opinion|to me|personally)\b',
    r'\b(love|hate|amazing|terrible|awful|fantastic|horrible|great|worst|best|annoying|frustrating|disappointed)\b',
    r'\b(would recommend|would not recommend|highly recommend|do not use|never again|totally worth|waste of)\b',
    r'[!]{2,}',                      # multiple exclamation marks
    r'\b(always|never|absolutely|definitely|obviously|clearly|honestly)\b',
]

# Strong objectivity signals — presence of these pushes score toward neutral
_STRONG_OBJECTIVE_PATTERNS = [
    r'\b(according to|research shows|studies show|it is known|the documentation|the changelog|release notes)\b',
    r'\b(version \d|v\d+\.\d+|released|announced|updated to|deprecated)\b',
    r'\b(how to|tutorial|steps to|guide for|example of|syntax for)\b',
]

_SUBJECTIVE_RE = [re.compile(p, re.IGNORECASE) for p in _STRONG_SUBJECTIVE_PATTERNS]
_OBJECTIVE_RE  = [re.compile(p, re.IGNORECASE) for p in _STRONG_OBJECTIVE_PATTERNS]
_FIRST_PERSON_RE = re.compile(r'\b(i|i\'m|i\'ve|i\'d|i\'ll|i am|i have|my|me|we|our|us)\b', re.IGNORECASE)
_STANCE_CUE_RE = re.compile(
    r'\b(honest question|not using|never using|what do you think|i need|i want|i am trying|i\'m trying|help me|is it reasonable)\b',
    re.IGNORECASE,
)
_QUESTION_CUE_RE = re.compile(r'\b(how|why|what|should|can|could|would|is it)\b', re.IGNORECASE)
_EMPHATIC_CUE_RE = re.compile(r'!\s*!|\b(ever|never)\b', re.IGNORECASE)


_vader_analyzer = None


def _get_vader_analyzer():
    global _vader_analyzer
    if _vader_analyzer is not None:
        return _vader_analyzer

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER loaded")
    except ImportError:
        logger.debug("vaderSentiment not installed — using heuristics only")
        _vader_analyzer = None

    return _vader_analyzer


def _rule_based_score(text: str) -> float:
    """
    Compute a subjectivity score in [0, 1] using VADER compound score + heuristics.
    Returns 0.5 if VADER is unavailable (triggers transformer for all posts).

    Score interpretation:
      0.0 → clearly objective/neutral
      1.0 → clearly opinionated/subjective
    """
    # Try VADER first
    vader_score = 0.5
    analyzer = _get_vader_analyzer()
    if analyzer is not None:
        compound = analyzer.polarity_scores(text)["compound"]
        # VADER compound is -1 to 1 — high absolute value = strong sentiment = opinionated
        vader_score = abs(compound)  # maps to 0–1

    # Count heuristic signals
    subj_hits = sum(1 for p in _SUBJECTIVE_RE if p.search(text))
    obj_hits  = sum(1 for p in _OBJECTIVE_RE  if p.search(text))

    # Combine: VADER base + heuristic adjustment
    # Each subjective hit adds 0.08, each objective hit subtracts 0.08
    heuristic_adjustment = (subj_hits - obj_hits) * 0.08
    score = vader_score + heuristic_adjustment

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def _should_flip_neutral_with_context(text: str, rule_score: float, confidence: float) -> bool:
    """
    Reduce false neutral predictions for informal Reddit stance-taking posts.
    """
    if confidence > NEUTRAL_CONTEXT_MAX_CONFIDENCE:
        return False
    if rule_score < NEUTRAL_CONTEXT_MIN_RULE_SCORE:
        return False

    has_first_person = _FIRST_PERSON_RE.search(text) is not None
    has_stance_cue = _STANCE_CUE_RE.search(text) is not None
    has_question_cue = _QUESTION_CUE_RE.search(text) is not None
    has_emphatic_cue = _EMPHATIC_CUE_RE.search(text) is not None

    return (has_first_person and has_stance_cue) or has_emphatic_cue or (has_stance_cue and has_question_cue)


# ── Transformer Setup ─────────────────────────────────────────────────────────

_classifier = None
_using_zsc  = False


def _get_classifier():
    global _classifier, _using_zsc
    if _classifier is not None:
        return _classifier, _using_zsc

    try:
        from transformers import pipeline as hf_pipeline

        try:
            _classifier = hf_pipeline(
                "text-classification",
                model=SUBJECTIVITY_MODEL,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                device=-1,  # CPU; change to 0 for GPU
            )
            _using_zsc = False
            logger.info("Loaded primary subjectivity model: %s", SUBJECTIVITY_MODEL)
        except Exception as e:
            logger.warning(
                "Primary model failed (%s), trying ZSC fallback: %s", e, ZSC_FALLBACK_MODEL
            )
            _classifier = hf_pipeline(
                "zero-shot-classification",
                model=ZSC_FALLBACK_MODEL,
                device=-1,
            )
            _using_zsc = True
            logger.info("Loaded fallback ZSC model: %s", ZSC_FALLBACK_MODEL)

    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers torch")
        raise

    return _classifier, _using_zsc


def _transformer_predict(texts: list[str]) -> list[tuple[str, float]]:
    """
    Run transformer on a batch of texts.
    Returns list of (label, confidence) tuples.
    label is "opinionated" or "neutral".
    """
    classifier, using_zsc = _get_classifier()
    results = []

    def _rebalance_label(label: str, score: float) -> tuple[str, float]:
        # GroNLP can over-predict neutral for informal Reddit phrasing.
        if label == "neutral" and score < NEUTRAL_FLIP_THRESHOLD:
            return "opinionated", 1.0 - score
        return label, score

    if using_zsc:
        # Zero-shot fallback: run one at a time (ZSC doesn't support clean batching here)
        for text in texts:
            try:
                out = classifier(
                    text[:MAX_TOKEN_LENGTH * 4],  # char limit approximation
                    candidate_labels=["personal opinion", "factual statement"],
                    multi_label=False,
                )
                top_label = out["labels"][0]
                score     = out["scores"][0]
                label = "opinionated" if "opinion" in top_label else "neutral"
                label, score = _rebalance_label(label, score)
                results.append((label, score))
            except Exception as e:
                logger.warning("ZSC error: %s — defaulting to neutral", e)
                results.append(("neutral", 0.5))
    else:
        # Primary model — batch inference
        try:
            batch_out = classifier(
                [t[:MAX_TOKEN_LENGTH * 4] for t in texts],
                batch_size=TRANSFORMER_BATCH_SIZE,
                truncation=True,
            )
            for out in batch_out:
                raw_label = out["label"].lower()
                score     = out["score"]
                # Model labels vary — normalise to opinionated/neutral
                # GroNLP model emits OBJ/SUBJ; keep broader fallbacks for other models.
                if any(k in raw_label for k in ("subj", "opinionated", "opinion")):
                    label = "opinionated"
                elif raw_label == "obj":
                    label = "neutral"
                else:
                    label = "neutral"
                label, score = _rebalance_label(label, score)
                results.append((label, score))
        except Exception as e:
            logger.warning("Batch transformer error: %s — falling back to neutral", e)
            results = [("neutral", 0.5)] * len(texts)

    return results


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run(
    records: list[dict],
    scores_buffer: list[dict] | None = None,
) -> list[dict]:
    """
    Hybrid subjectivity detection.

    scores_buffer: if provided, appends score rows for scores.jsonl.
    Rows may include optional audit metadata:
      - label_source
      - label_flipped
      - model_subjectivity_score
    """
    if scores_buffer is None:
        scores_buffer = []

    rule_neutral    = 0
    rule_opinionated = 0
    transformer_needed: list[tuple[int, str, float]] = []  # (index, doc_id, rule_score)

    # ── Stage 1: Rule-based ──────────────────────────────────────────────────
    for i, record in enumerate(records):
        text   = record.get("clean_text", "") or ""
        doc_id = record.get("doc_id", "UNKNOWN")
        platform_context = record.get("platform_context") or {}
        flair = ""
        if isinstance(platform_context, dict):
            flair = (platform_context.get("flair") or "")

        # Mod-style announcements/reminders are typically objective.
        if any(k in flair.lower() for k in MOD_FLAIR_KEYWORDS):
            record.setdefault("labels", {})["subjectivity"] = "neutral"
            # Deterministic override path: force neutral with high branch confidence.
            scores_buffer.append(
                {
                    "doc_id": doc_id,
                    "subjectivity_score": 1.0,
                    "label_source": "flair_override",
                }
            )
            rule_neutral += 1
            continue

        # Fallback for moderation/rule reminders when flair is missing.
        if _MOD_POST_RE.search(text):
            record.setdefault("labels", {})["subjectivity"] = "neutral"
            # Deterministic override path: force neutral with high branch confidence.
            scores_buffer.append(
                {
                    "doc_id": doc_id,
                    "subjectivity_score": 1.0,
                    "label_source": "mod_content_override",
                }
            )
            rule_neutral += 1
            continue

        if not text.strip():
            # Empty text → neutral, no score
            record.setdefault("labels", {})["subjectivity"] = "neutral"
            scores_buffer.append({"doc_id": doc_id, "subjectivity_score": 0.0})
            rule_neutral += 1
            continue

        score = _rule_based_score(text)

        if score < RULE_LOW:
            record.setdefault("labels", {})["subjectivity"] = "neutral"
            # Direct rule-path decision (no label_source field by design).
            scores_buffer.append({"doc_id": doc_id, "subjectivity_score": round(score, 4)})
            rule_neutral += 1

        elif score > RULE_HIGH:
            record.setdefault("labels", {})["subjectivity"] = "opinionated"
            # Direct rule-path decision (no label_source field by design).
            scores_buffer.append({"doc_id": doc_id, "subjectivity_score": round(score, 4)})
            rule_opinionated += 1

        else:
            # Ambiguous — queue for transformer
            transformer_needed.append((i, doc_id, score))
            record.setdefault("labels", {})["subjectivity"] = None  # placeholder

    logger.info(
        "Stage 1 (rule-based): %d neutral, %d opinionated, %d ambiguous → transformer",
        rule_neutral, rule_opinionated, len(transformer_needed),
    )

    # ── Stage 2: Transformer for ambiguous posts ─────────────────────────────
    if transformer_needed:
        indices  = [t[0]     for t in transformer_needed]
        doc_ids  = [t[1]     for t in transformer_needed]
        texts    = [records[i].get("clean_text", "") for i in indices]

        predictions = _transformer_predict(texts)

        for (idx, doc_id, rule_score), (label, confidence) in zip(transformer_needed, predictions):
            text = records[idx].get("clean_text", "") or ""
            original_model_confidence = confidence
            label_source = "transformer"
            label_flipped = False

            if label == "neutral" and _should_flip_neutral_with_context(text, rule_score, confidence):
                label = "opinionated"
                confidence = 1.0 - original_model_confidence
                label_source = "context_override"
                label_flipped = True

            records[idx]["labels"]["subjectivity"] = label
            # Ambiguous path: write confidence with explicit source metadata.
            score_entry = {
                "doc_id": doc_id,
                "subjectivity_score": round(confidence, 4),
                "label_source": label_source,
            }
            if label_flipped:
                score_entry["label_flipped"] = True
                score_entry["model_subjectivity_score"] = round(original_model_confidence, 4)
            scores_buffer.append(score_entry)

        trans_opinionated = sum(
            1
            for idx, _, _ in transformer_needed
            if records[idx]["labels"].get("subjectivity") == "opinionated"
        )
        trans_neutral = len(transformer_needed) - trans_opinionated
        logger.info(
            "Stage 2 (transformer): %d opinionated, %d neutral",
            trans_opinionated, trans_neutral,
        )

    # ── Ensure labels dict has all required null fields for downstream ────────
    for record in records:
        labels = record.setdefault("labels", {})
        # Force downstream fields to schema-compliant nulls.
        labels["polarity"] = None
        labels["aspects"] = None
        labels["agents"] = None
        labels["sarcasm"] = None

    logger.info(
        "s6_subjectivity complete: %d records | opinionated: %d | neutral: %d",
        len(records),
        sum(1 for r in records if r["labels"].get("subjectivity") == "opinionated"),
        sum(1 for r in records if r["labels"].get("subjectivity") == "neutral"),
    )

    return records
