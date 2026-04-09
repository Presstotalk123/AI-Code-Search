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


# Domain-specific words to protect from spell correction
_DOMAIN_WORDS = [
    # AI coding tools
    'copilot', 'claude code', 'codex', 'kline', 'llama', 'gemma', 'gpt', 'haiku', 'sonnet', 'opus','cursor', 'windsurf', 'codewhisperer', 'tabnine', 'codeium',
    'supermaven', 'devin', 'chatgpt', 'gemini', 'replit', 'jetbrains',
    'claude', 'kimi', 'grok', 'cline', 'lovable', 'antigravity',
    'bolt', 'roo', 'kilo', 'openai', 'mistral', 'gemini', 'anthropic', 'deepseek', 'moonshot',
    # AI / ML terminology
    'ai', 'llm', 'llms', 'agentic', 'finetune', 'finetuning', 'finetuned',
    'hallucinate', 'hallucination', 'hallucinations', 'hallucinating', 'hallucinated',
    # Editors / IDEs
    'vscode', 'ide', 'neovim', 'nvim', 'pycharm', 'intellij',
    # Dev tools / ecosystems
    'github', 'gitlab', 'stackoverflow', 'reddit', 'subreddit',
    'docker', 'dockerfile', 'kubectl', 'localhost',
    'npm', 'pip', 'sdk', 'cli', 'cors', 'json', 'yaml', 'toml', 'huggingface', 'openrouter',
    'eslint', 'linter', 'linting',
    'pytorch', 'tensorflow',
    # Programming concepts
    'autocomplete', 'autocompletion', 'autoformat',
    'refactor', 'refactoring', 'plugin', 'snippet', 'workflow', 'bugfix',
    'codebase', 'boilerplate', 'typescript', 'javascript', 'golang',
    'async', 'await', 'regex', 'config',
    # Reddit-specific
    'upvote', 'upvotes', 'downvote', 'downvotes',
    # Misc
    'v0',
]

_spell_checker = None


def _get_spell_checker():
    global _spell_checker
    if _spell_checker is None:
        from spellchecker import SpellChecker
        _spell_checker = SpellChecker()
        # Give domain words a high frequency (50000) so they outrank common English
        # words when both are at the same edit distance from a misspelled token.
        # e.g. "cursr" → "cursor" (domain) beats "curse" (common English, low dist)
        high_freq = {word: 1_000_000 for word in _DOMAIN_WORDS}
        _spell_checker.word_frequency._dictionary.update(high_freq)
        # Remove 'claud' so it doesn't block correction to 'claude'
        _spell_checker.word_frequency._dictionary.pop('claud', None)
    return _spell_checker


def suggest_spell_correction(query: str):
    """
    Returns a corrected query string if misspellings are found, else None.
    Skips tokens that are very short (<= 2 chars), contain ':', or are digits.
    """
    try:
        spell = _get_spell_checker()
        tokens = query.split()
        corrected_tokens = []
        changed = False
        for token in tokens:
            if len(token) <= 2 or ':' in token or token.isdigit():
                corrected_tokens.append(token)
                continue
            correction = spell.correction(token)
            if correction and correction != token.lower():
                corrected_tokens.append(correction)
                changed = True
            else:
                corrected_tokens.append(token)
        return ' '.join(corrected_tokens) if changed else None
    except Exception as e:
        logger.warning(f"Spell check error: {e}")
        return None
