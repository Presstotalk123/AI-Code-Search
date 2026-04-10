#!/usr/bin/env python3
"""
SC4021 — Person 2: Aspect Detection with Checkpointing
Saves progress after every batch — can be stopped and resumed anytime.
"""

import json
import os
import math
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURE HERE
# ─────────────────────────────────────────────
INPUT_FILE       = "natalie_final.jsonl"
OUTPUT_FILE      = "person2_sruthi_output_latest.jsonl"
SCORES_FILE      = "scores_sruthi_new_check_latest.jsonl"
CHECKPOINT_FILE  = "checkpoint.json"        # tracks progress
BATCH_SIZE       = 32
# ─────────────────────────────────────────────


# ── DIMENSIONS ────────────────────────────────────────────────────────────────
DIMENSIONS = [
    ("productivity",
     "This post is specifically about saving development time, faster code completion, "
     "or improved coding speed and workflow using AI tools"),

    ("trust_reliability",
     "This post is about AI giving wrong outputs, hallucinations, lying, performance "
     "degradation, the model getting dumber or worse over time, low trust, unreliable "
     "results, or not working as expected, catching AI mistakes"),

    ("code_quality",
     "This post is about code readability, bugs, best practices, technical debt or clean code"),

    ("control",
     "This post is specifically about an AI coding tool autonomously executing terminal "
     "commands, deleting files, or modifying code the developer explicitly did not request"),

    ("security_privacy",
     "This post is about confidentiality, compliance, data leakage, IP concerns, "
     "or sending code to external services"),

    ("learning_impact",
     "This post is specifically about a developer's programming skills improving or "
     "declining because of AI tool usage, or junior developers becoming over-reliant on AI"),

    ("job_security",
     "This post is about fear of replacement, layoffs, or AI replacing developers"),

    ("cost_value",
     "This post is about pricing, Return on Investment, subscriptions, comparing tiers "
     "and plans or whether AI tools are worth the cost"),

    ("token_usage",
     "This post is about token consumption, API credits, usage limits, optimizing prompt "
     "length or burning context window"),

    ("integration_ux",
     "This post is about VSCode plugins, editor support, or buggy AI extensions"),

    ("troubleshooting",
     "This post is specifically about debugging an error, fixing a broken feature, "
     "or resolving unexpected behavior"),
]

DIMENSION_NAMES = [d[0] for d in DIMENSIONS]
DIMENSION_DESCS = [d[1] for d in DIMENSIONS]


# ── KEYWORD BOOSTING ──────────────────────────────────────────────────────────
BOOST_KEYWORDS = {
    "cost_value":       ["cheap", "expensive", "price", "subscription", "worth",
                         "tier", "plan", "pricing", "cost", "paid", "free tier", "invoice"],
    "token_usage":      ["token", "context window", "api credits", "usage limit",
                         "premium requests", "rate limit", "free tier", "quota",
                         "burning", "context length"],
    "security_privacy": ["leak", "confidential", "compliance", "ip", "privacy",
                         "breach", "gdpr", "data", "send code", "external"],
}

_CLASSIFIER = None


# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_threshold(text: str) -> float:
    token_count = len(text.split())
    if token_count < 20:
        return 0.09
    elif token_count < 80:
        return 0.10
    else:
        return 0.12


def softmax(scores: list) -> list:
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def apply_keyword_boost(text: str, norm_scores: dict) -> dict:
    text_lower = text.lower()
    boosted = dict(norm_scores)
    for dim, keywords in BOOST_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            boosted[dim] = min(boosted.get(dim, 0) + 0.15, 1.0)
    return boosted


def detect_aspects(text: str, classifier) -> tuple:
    result = classifier(
        text,
        candidate_labels=DIMENSION_DESCS,
        multi_label=True
    )
    desc_to_name = {desc: name for name, desc in DIMENSIONS}
    raw_scores = {
        desc_to_name[label]: score
        for label, score in zip(result["labels"], result["scores"])
    }
    ordered_scores = [raw_scores[name] for name in DIMENSION_NAMES]
    normalised = softmax(ordered_scores)
    norm_scores = dict(zip(DIMENSION_NAMES, normalised))
    boosted_scores = apply_keyword_boost(text, norm_scores)
    threshold = get_threshold(text)
    matched = {
        name: score
        for name, score in boosted_scores.items()
        if score >= threshold
    }
    if len(matched) > 4:
        matched = dict(
            sorted(matched.items(), key=lambda x: x[1], reverse=True)[:4]
        )
    aspects_dict = {name: None for name in matched}
    return aspects_dict, boosted_scores


def _load_classifier(device_preference: str = "auto"):
    global _CLASSIFIER
    if _CLASSIFIER is not None:
        return _CLASSIFIER

    import torch
    from transformers import pipeline

    if device_preference == "cuda" and torch.cuda.is_available():
        device = 0
    elif device_preference == "cpu":
        device = -1
    elif torch.cuda.is_available():
        device = 0
    else:
        device = -1

    _CLASSIFIER = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        device=device,
    )
    logger.info("s6.5 classifier loaded (device=%s)", "cuda" if device == 0 else "cpu")
    return _CLASSIFIER


def run(
    records: list[dict],
    *,
    scores_buffer: list[dict[str, Any]] | None = None,
    device_preference: str = "auto",
) -> list[dict]:
    classifier = _load_classifier(device_preference=device_preference)
    errors = 0

    for record in records:
        text = (record.get("clean_text") or "").strip()
        labels = record.setdefault("labels", {})

        if not text:
            aspects_dict = {}
            scores_dict = {name: 0.0 for name in DIMENSION_NAMES}
        else:
            try:
                aspects_dict, scores_dict = detect_aspects(text, classifier)
            except Exception as e:
                logger.warning("s6.5 failed for doc_id=%s: %s", record.get("doc_id"), e)
                errors += 1
                aspects_dict = {}
                scores_dict = {name: 0.0 for name in DIMENSION_NAMES}

        labels["aspects"] = aspects_dict
        if scores_buffer is not None:
            scores_buffer.append(
                {
                    "doc_id": record.get("doc_id"),
                    "aspect_scores": scores_dict,
                }
            )

    logger.info("s6.5 complete: %d records, %d errors", len(records), errors)
    return records


# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────────
def load_checkpoint():
    """Returns the index to resume from (0 if no checkpoint)."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
            start_idx = data.get("last_completed_index", 0)
            print(f"Resuming from record {start_idx + 1}...")
            return start_idx
    print("No checkpoint found — starting from beginning.")
    return 0


def save_checkpoint(last_completed_index: int):
    """Saves current progress to checkpoint file."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_completed_index": last_completed_index}, f)


def delete_checkpoint():
    """Removes checkpoint file after successful completion."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== SC4021 Person 2 — Aspect Detection (with checkpointing) ===\n")

    # ── Load input ────────────────────────────────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(records):,} records from {INPUT_FILE}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    start_idx = load_checkpoint()
    total = len(records)

    if start_idx >= total:
        print("Already completed! Nothing to process.")
        return

    print(f"Records remaining: {total - start_idx:,}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading DeBERTa zero-shot classifier...")
    classifier = _load_classifier(device_preference="auto")
    print("Model loaded.\n")

    # ── Process with checkpointing ────────────────────────────────────────────
    # Open output files in APPEND mode so we don't overwrite previous progress
    out_mode   = "a" if start_idx > 0 else "w"
    score_mode = "a" if start_idx > 0 else "w"

    with open(OUTPUT_FILE, out_mode, encoding="utf-8") as out_f, \
         open(SCORES_FILE, score_mode, encoding="utf-8") as scores_f:

        for i in range(start_idx, total, BATCH_SIZE):
            batch = records[i: i + BATCH_SIZE]
            print(f"  Processing records {i+1}–{min(i+BATCH_SIZE, total)} of {total}...")

            for record in batch:
                text = record.get("clean_text", "")

                if not text or not text.strip():
                    aspects_dict = {}
                    scores_dict = {name: 0.0 for name in DIMENSION_NAMES}
                else:
                    try:
                        aspects_dict, scores_dict = detect_aspects(text, classifier)
                    except Exception as e:
                        print(f"    Warning: failed on doc_id={record.get('doc_id')} — {e}")
                        aspects_dict = {}
                        scores_dict = {name: 0.0 for name in DIMENSION_NAMES}

                if "labels" not in record:
                    record["labels"] = {
                        "subjectivity": None,
                        "polarity": None,
                        "aspects": {},
                        "agents": None,
                        "sarcasm": None
                    }
                record["labels"]["aspects"] = aspects_dict

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                scores_f.write(json.dumps({
                    "doc_id": record.get("doc_id"),
                    "aspect_scores": scores_dict
                }, ensure_ascii=False) + "\n")

            # Save checkpoint after every batch
            last_completed = min(i + BATCH_SIZE, total)
            save_checkpoint(last_completed)
            out_f.flush()
            scores_f.flush()

    # ── Done ──────────────────────────────────────────────────────────────────
    delete_checkpoint()
    print(f"\n✅ Done!")
    print(f"   Main output : {OUTPUT_FILE}  ({total:,} records)")
    print(f"   Scores file : {SCORES_FILE}")

    # Count aspects for validation
    with open(OUTPUT_FILE) as f:
        out_records = [json.loads(l) for l in f if l.strip()]

    empty   = sum(1 for r in out_records if r["labels"]["aspects"] == {})
    single  = sum(1 for r in out_records if len(r["labels"]["aspects"]) == 1)
    multi   = sum(1 for r in out_records if len(r["labels"]["aspects"]) > 1)
    over4   = sum(1 for r in out_records if len(r["labels"]["aspects"]) > 4)

    print(f"\n── Validation Summary ──────────────────────────────────")
    print(f"   Records with 0 aspects : {empty:,}")
    print(f"   Records with 1 aspect  : {single:,}")
    print(f"   Records with 2+ aspects: {multi:,}")
    print(f"   Records with >4 aspects: {over4:,}  (should be 0)")
    print(f"────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
