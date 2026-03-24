"""Data loader for JSONL Reddit dataset"""
import json
import logging
from typing import Dict, Iterator

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> Iterator[Dict]:
    """
    Load JSONL file line by line (memory efficient for large files)

    Args:
        file_path: Path to JSONL file

    Yields:
        Parsed JSON records
    """
    logger.info(f"Loading data from {file_path}")
    count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    count += 1
                    yield record
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                    continue

        logger.info(f"Successfully loaded {count} records")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def combine_text_content(record: Dict) -> str:
    """
    Merge thread_title + main_text + reply_text for comprehensive search

    Args:
        record: JSON record from dataset

    Returns:
        Combined text content
    """
    parts = []

    # Thread title (for posts)
    if record['content'].get('thread_title'):
        parts.append(record['content']['thread_title'])

    # Main text (for posts)
    if record['content'].get('main_text'):
        parts.append(record['content']['main_text'])

    # Reply text (for comments)
    if record['content'].get('reply_text'):
        parts.append(record['content']['reply_text'])

    combined = " ".join(parts).strip()
    return combined if combined else "[No text content]"


def extract_title(record: Dict) -> str:
    """
    Extract title from record

    Args:
        record: JSON record

    Returns:
        Title string (thread_title or empty)
    """
    return record['content'].get('thread_title', '')
