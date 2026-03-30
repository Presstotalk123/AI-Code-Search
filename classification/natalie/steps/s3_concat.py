"""
Step 3 — Text Concatenation
  3a. Build a lookup of thread_id → thread_title from post records
  3b. Enrich comment records with parent thread_title (comments have null thread_title)
  3c. Concatenate thread_title + main_text + reply_text into raw_text per record
"""

import logging
from .utils import get_text_fields, is_empty

logger = logging.getLogger(__name__)


def _build_thread_title_lookup(records: list[dict]) -> dict[str, str]:
    """
    Build a map of thread_id → thread_title from post-type records.
    Used to enrich comment records that have no thread_title of their own.
    """
    lookup: dict[str, str] = {}
    for record in records:
        content_type = record.get("content_type", "")
        if content_type != "post":
            continue
        ctx = record.get("platform_context", {}) or {}
        thread_id = ctx.get("thread_id")
        thread_title, _, _ = get_text_fields(record)
        if thread_id and thread_title and not is_empty(thread_title):
            lookup[thread_id] = thread_title.strip()
    logger.info("Built thread title lookup with %d entries", len(lookup))
    return lookup


def concatenate_text(
    records: list[dict],
    thread_lookup: dict[str, str] | None = None,
) -> list[dict]:
    """
    Adds raw_text to each record by concatenating available text fields.
    Comment records are enriched with the parent post's thread_title if available.
    """
    if thread_lookup is None:
        thread_lookup = {}

    enriched = 0
    for record in records:
        thread_title, main_text, reply_text = get_text_fields(record)

        # 3b — Enrich comment records with parent thread_title
        content_type = record.get("content_type", "")
        if content_type == "comment" and is_empty(thread_title):
            ctx = record.get("platform_context", {}) or {}
            thread_id = ctx.get("thread_id")
            if thread_id and thread_id in thread_lookup:
                thread_title = thread_lookup[thread_id]
                enriched += 1

        # 3c — Concatenate non-empty parts
        parts = [
            thread_title or "",
            main_text    or "",
            reply_text   or "",
        ]
        raw_text = " ".join(p.strip() for p in parts if p and p.strip())
        record["raw_text"] = raw_text if raw_text else ""

    logger.info(
        "s3_concat: %d records processed, %d comment records enriched with thread title",
        len(records), enriched,
    )
    return records


def run(records: list[dict]) -> list[dict]:
    """
    Full Step 3 pipeline:
      - Build thread title lookup from post records
      - Concatenate text fields for all records
    """
    thread_lookup = _build_thread_title_lookup(records)
    return concatenate_text(records, thread_lookup=thread_lookup)
