"""
Step 1 — Deduplication & Record Filtering
  1a. Deduplicate by doc_id (keep first occurrence)
  1b. Drop records where thread_title + main_text + reply_text are ALL empty
  1c. Normalise engagement field casing (Downvotes → downvotes, Awards → awards)
"""

import logging
from .utils import is_empty, get_text_fields, normalise_engagement, write_skip_log

logger = logging.getLogger(__name__)


def filter_records(
    records: list[dict],
    log_path: str = "logs/skipped_docids.txt",
) -> list[dict]:
    seen_ids: set[str] = set()
    skipped:  list[tuple[str, str]] = []
    cleaned:  list[dict] = []

    for record in records:
        doc_id = record.get("doc_id") or record.get("docid", "UNKNOWN")
        thread_title, main_text, reply_text = get_text_fields(record)

        # 1a — Deduplication
        if doc_id in seen_ids:
            skipped.append((doc_id, "duplicate_doc_id"))
            continue
        seen_ids.add(doc_id)

        # 1b — All-null text check
        if is_empty(thread_title) and is_empty(main_text) and is_empty(reply_text):
            skipped.append((doc_id, "all_text_fields_empty"))
            continue

        # 1c — Normalise engagement casing
        record = normalise_engagement(record)

        cleaned.append(record)

    write_skip_log(skipped, log_path)

    logger.info(
        "s1_filter: %d kept, %d skipped (dupes: %d, empty: %d)",
        len(cleaned), len(skipped),
        sum(1 for _, r in skipped if r == "duplicate_doc_id"),
        sum(1 for _, r in skipped if r == "all_text_fields_empty"),
    )
    return cleaned