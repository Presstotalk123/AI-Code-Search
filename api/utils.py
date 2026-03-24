"""Utility functions for API"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """
    Parse ISO8601 date string

    Args:
        date_str: Date string (e.g., "2025-01-15" or "2025-01-15T10:30:00Z")

    Returns:
        datetime object
    """
    try:
        # Handle both date and datetime formats
        if 'T' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            return datetime.fromisoformat(date_str)
    except ValueError as e:
        logger.warning(f"Date parsing error: {e}")
        raise ValueError(f"Invalid date format: {date_str}")


def format_response_error(error_message: str, status_code: int = 400) -> tuple:
    """
    Format error response

    Args:
        error_message: Error message
        status_code: HTTP status code

    Returns:
        Tuple of (response_dict, status_code)
    """
    return {'error': error_message, 'status': 'error'}, status_code


def validate_search_params(query: str, mode: str, page: int, page_size: int) -> tuple:
    """
    Validate search parameters

    Args:
        query: Search query
        mode: Search mode
        page: Page number
        page_size: Results per page

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query parameter 'q' is required and cannot be empty"

    if mode not in ['keyword', 'semantic', 'hybrid']:
        return False, "Invalid mode. Use: keyword, semantic, or hybrid"

    if page < 1:
        return False, "Page must be >= 1"

    if page_size < 1 or page_size > 100:
        return False, "Page size must be between 1 and 100"

    return True, None
