"""
Step 2 — Bot Filtering + Deleted/Removed Record Filtering
  2a. Skip records where author.username is in the bot list
  2b. Skip records where BOTH main_text AND reply_text are null/[deleted]/[removed]
      (thread_title alone is not sufficient to keep the record)
"""

import logging
from .utils import is_deleted, get_text_fields, write_skip_log

logger = logging.getLogger(__name__)

_BOT_LIST = {
    "AutoModerator",
    "AutoModerator-ModTeam",
    "ModTeam",
    "RemindMeBot",
    "UpdateMeBot",
    "RepostSleuthBot",
    "WikiSummarizerBot",
    "ExperiencedDevs-ModTeam",
    "Reddit-ModTeam",
    "RedditBot",
    "reddit",
    "sneakpeek_bot",
    "timezone_bot", 
    "convertbot",
    "MAGIC_EYE_BOT",
    "image_linker_bot",
}


def filter_records(
    records: list[dict],
    log_path: str = "logs/skipped_docids.txt",
) -> list[dict]:
    skipped: list[tuple[str, str]] = []
    cleaned: list[dict] = []

    for record in records:
        doc_id   = record.get("doc_id") or record.get("docid", "UNKNOWN")
        author   = record.get("author", {}) or {}
        username = author.get("username", "")
        _, main_text, reply_text = get_text_fields(record)

        # 2a — Bot author check
        if username in _BOT_LIST:
            skipped.append((doc_id, f"bot_author:{username}"))
            continue

        # 2b — Both main_text and reply_text deleted/removed/null
        if is_deleted(main_text) and is_deleted(reply_text):
            skipped.append((doc_id, "both_texts_deleted_or_null"))
            continue

        cleaned.append(record)

    write_skip_log(skipped, log_path)

    logger.info(
        "s2_bot_filter: %d kept, %d skipped (bots: %d, deleted: %d)",
        len(cleaned), len(skipped),
        sum(1 for _, r in skipped if r.startswith("bot_author:")),
        sum(1 for _, r in skipped if r == "both_texts_deleted_or_null"),
    )
    return cleaned