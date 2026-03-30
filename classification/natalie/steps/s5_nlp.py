"""
Step 5 — spaCy NLP Pipeline
    5a. Sentence Boundary Disambiguation (SBD) using spaCy parser
  5b. POS Tagging
  5c. Text Chunking — noun phrases + verb phrases
  5d. Lemmatisation (stopwords, punctuation, whitespace excluded)

Input:  record["clean_text"]  (written by Step 4)
Output: record["nlp"] dict with keys: sentences, pos_tags, chunks, lemmas

The nlp dict is used downstream by:
  - Person 3 ABSA  → sentences (to localise aspect sentiment)
  - Person 2       → lemmas (optional boost signal)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_VERB_PHRASE_TOKENS = 10

# ── spaCy Setup ───────────────────────────────────────────────────────────────

_nlp = None  # Lazy-loaded


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError:
            logger.warning(
                "en_core_web_sm not found. Run: python -m spacy download en_core_web_sm"
            )
            raise

        # Keep parser-based sentence segmentation for better boundary accuracy.
        # Do not add sentencizer when using en_core_web_sm with parser enabled.

        logger.info(
            "spaCy pipeline loaded: %s | pipes: %s",
            _nlp.meta.get("name", "unknown"),
            _nlp.pipe_names,
        )
    except ImportError:
        logger.error("spaCy not installed. Run: pip install spacy")
        raise
    return _nlp


# ── NLP Helpers ───────────────────────────────────────────────────────────────

def _extract_verb_phrases(doc: Any) -> list[str]:
    """
    Extract verb phrases as the full subtree of each VERB or AUX head token.
    Filters out single-token 'phrases' and punctuation-only results.
    """
    phrases = []
    for token in doc:
        if token.pos_ in ("VERB", "AUX") and token == token.head:
            subtree_tokens = [
                t.text for t in token.subtree
                if not t.is_punct and not t.is_space
            ]
            phrase = " ".join(subtree_tokens).strip()
            if phrase and 1 < len(subtree_tokens) <= MAX_VERB_PHRASE_TOKENS:
                phrases.append(phrase)
    return phrases


def _extract_noun_chunks(doc: Any) -> list[str]:
    return [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip() and len(chunk) > 1]


def process_text(text: str) -> dict:
    """
    Run the full spaCy pipeline on a single text string.

    Returns a dict:
      sentences : list[str]          — sentence strings
      pos_tags  : list[(str, str)]   — (token, POS) pairs
      chunks    : list[str]          — noun + verb phrase strings
      lemmas    : list[str]          — lemmatised content words (no stops/punct)
    """
    if not text or not text.strip():
        return {"sentences": [], "pos_tags": [], "chunks": [], "lemmas": []}

    nlp = _get_nlp()

    # Truncate to spaCy's max length to avoid memory issues on very long posts
    max_chars = nlp.max_length - 1
    if len(text) > max_chars:
        logger.debug("Truncating text from %d to %d chars for spaCy", len(text), max_chars)
        text = text[:max_chars]

    doc = nlp(text)

    # 5a — Sentence Boundary Disambiguation
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # 5b — POS Tagging
    pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]

    # 5c — Chunking: noun phrases (built-in) + verb phrases (manual)
    noun_chunks = _extract_noun_chunks(doc)
    verb_phrases = _extract_verb_phrases(doc)
    chunks = noun_chunks + verb_phrases

    # 5d — Lemmatisation: exclude stopwords, punctuation, whitespace, numbers
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and not token.like_num
        and token.lemma_.strip()
        and len(token.lemma_) > 1  # skip single-char lemmas
    ]

    return {
        "sentences": sentences,
        "pos_tags":  pos_tags,
        "chunks":    chunks,
        "lemmas":    lemmas,
    }


# ── Batch Processing ──────────────────────────────────────────────────────────

def run(records: list[dict], batch_size: int = 256) -> list[dict]:
    """
    Run spaCy NLP pipeline on all records in batches.
    Reads clean_text, writes nlp dict.

    batch_size: number of texts passed to nlp.pipe() at once.
                Higher = faster but more memory. 256 is a throughput-focused default.
    """
    nlp = _get_nlp()

    texts = [record.get("clean_text", "") or "" for record in records]

    # Batch processing via nlp.pipe for efficiency
    processed = 0
    docs = []
    try:
        for doc in nlp.pipe(texts, batch_size=batch_size):
            docs.append(doc)
            processed += 1
    except Exception as e:
        logger.error("spaCy batch processing failed at doc %d: %s", processed, e)
        raise

    for record, doc in zip(records, docs):
        text = record.get("clean_text", "") or ""
        if not text.strip():
            record["nlp"] = {"sentences": [], "pos_tags": [], "chunks": [], "lemmas": []}
            continue

        sentences  = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        pos_tags   = [(token.text, token.pos_) for token in doc if not token.is_space]
        noun_chunks = _extract_noun_chunks(doc)
        verb_phrases = _extract_verb_phrases(doc)
        chunks     = noun_chunks + verb_phrases
        lemmas     = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and not token.like_num
            and token.lemma_.strip()
            and len(token.lemma_) > 1
        ]

        record["nlp"] = {
            "sentences": sentences,
            "pos_tags":  pos_tags,
            "chunks":    chunks,
            "lemmas":    lemmas,
        }

    logger.info(
        "s5_nlp: processed %d records | avg sentences: %.1f | avg lemmas: %.1f",
        len(records),
        sum(len(r["nlp"]["sentences"]) for r in records) / max(len(records), 1),
        sum(len(r["nlp"]["lemmas"])    for r in records) / max(len(records), 1),
    )
    return records
