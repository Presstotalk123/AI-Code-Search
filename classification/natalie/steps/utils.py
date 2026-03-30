"""Shared helpers used across multiple steps."""

from typing import Any

_EMPTY_SENTINELS  = {"", "[deleted]", "[removed]", "null", "none"}
_DELETED_SENTINELS = {"[deleted]", "[removed]"}

_ENGAGEMENT_REMAP = {
    "Downvotes": "downvotes",
    "Awards":    "awards",
    "Upvotes":   "upvotes",
    "Comments":  "comments",
    "Score":     "score",
}


def is_empty(value: Any) -> bool:
    """True if value is None or a known empty/deleted/removed sentinel."""
    if value is None:
        return True
    return str(value).strip().lower() in _EMPTY_SENTINELS


def is_deleted(value: Any) -> bool:
    """True if value is None or specifically [deleted]/[removed]."""
    if value is None:
        return True
    return str(value).strip().lower() in _DELETED_SENTINELS


def get_text_fields(record: dict) -> tuple[str | None, str | None, str | None]:
    """Extract (thread_title, main_text, reply_text) from the nested content block."""
    content = record.get("content", {}) or {}
    thread_title = content.get("thread_title") or content.get("threadtitle")
    main_text    = content.get("main_text")    or content.get("maintext")
    reply_text   = content.get("reply_text")   or content.get("replytext")
    return thread_title, main_text, reply_text


def normalise_engagement(record: dict) -> dict:
    """Fix engagement field casing in-place (Downvotes → downvotes, etc.)."""
    engagement = record.get("engagement", {})
    if not isinstance(engagement, dict):
        return record
    record["engagement"] = {
        _ENGAGEMENT_REMAP.get(k, k): v
        for k, v in engagement.items()
    }
    return record


def write_skip_log(skipped: list[tuple[str, str]], log_path: str) -> None:
    if not skipped:
        return
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for doc_id, reason in skipped:
            f.write(f"{doc_id}\t{reason}\n")