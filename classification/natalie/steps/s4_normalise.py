"""
Step 4 — Microtext Normalisation
  4a. Protect technical terms and agent names from spell correction
  4b. Normalise using ekphrasis (URLs, slang, hashtags, contractions, emoticons)
  4c. Clean up ekphrasis annotation tags
  4d. Collapse excess whitespace

Produces: record["clean_text"]
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Protected Terms Whitelist ─────────────────────────────────────────────────
# These are replaced with placeholder tokens before ekphrasis runs,
# then restored after. This prevents spell correction from mangling them.

PROTECTED_TERMS: list[tuple[str, str]] = [
    # .NET ecosystem
    (".NET", "__DOTNET__"),
    (".net", "__DOTNET__"),
    # Editors / IDEs
    ("VSCode", "__VSCODE__"),
    ("vscode", "__VSCODE__"),
    ("VS Code", "__VSCODE__"),
    # AI / ML terms
    ("LangChain", "__LANGCHAIN__"),
    ("langchain", "__LANGCHAIN__"),
    ("DeBERTa", "__DEBERTA__"),
    ("deberta", "__DEBERTA__"),
    ("RoBERTa", "__ROBERTA__"),
    ("roberta", "__ROBERTA__"),
    ("GPT-4o", "__GPT4O__"),
    ("GPT-4", "__GPT4__"),
    ("GPT-3.5", "__GPT35__"),
    ("Gemini-1.5", "__GEMINI15__"),
    ("Gemini-2.0", "__GEMINI20__"),
    ("Claude-3", "__CLAUDE3__"),
    ("gpt-4o", "__GPT4O__"),
    ("gpt-4", "__GPT4__"),
    ("gpt-3.5", "__GPT35__"),
    ("gemini-1.5", "__GEMINI15__"),
    ("gemini-2.0", "__GEMINI20__"),
    ("claude-3", "__CLAUDE3__"),
    ("HuatuoGPT-o1", "__HUATUOGPT_O1__"),
    ("huatuogpt-o1", "__HUATUOGPT_O1__"),
    # Common technical abbreviations
    ("API", "__API__"),
    ("CLI", "__CLI__"),
    ("SDK", "__SDK__"),
    ("LLM", "__LLM__"),
    ("RAG", "__RAG__"),
    ("NLP", "__NLP__"),
    ("ABSA", "__ABSA__"),
    # Python / JS special dunder-like names used in code snippets
    ("__name__", "__PYNAME__"),
    ("__main__", "__PYMAIN__"),
    ("__init__", "__PYINIT__"),
    ("__call__", "__PYCALL__"),
    ("__dirname", "__JSDIRNAME__"),
    ("__filename", "__JSFILENAME__"),
    # Agent names (from AGENTS list)
    ("GitHub Copilot", "__COPILOT__"),
    ("github copilot", "__COPILOT__"),
    ("Cursor AI", "__CURSOR__"),
    ("cursor ai", "__CURSOR__"),
    ("Claude Code", "__CLAUDECODE__"),
    ("claude code", "__CLAUDECODE__"),
    ("CodeWhisperer", "__CODEWHISPERER__"),
    ("codewhisperer", "__CODEWHISPERER__"),
    ("TabNine", "__TABNINE__"),
    ("tabnine", "__TABNINE__"),
    ("Windsurf", "__WINDSURF__"),
    ("windsurf", "__WINDSURF__"),
    ("Codeium", "__CODEIUM__"),
    ("codeium", "__CODEIUM__"),
    ("Supermaven", "__SUPERMAVEN__"),
    ("supermaven", "__SUPERMAVEN__"),
    ("continue.dev", "__CONTINUEDEV__"),
    ("Sourcegraph", "__SOURCEGRAPH__"),
    ("sourcegraph", "__SOURCEGRAPH__"),
    ("JetBrains", "__JETBRAINS__"),
    ("jetbrains", "__JETBRAINS__"),
    ("bolt.new", "__BOLTNEW__"),
    ("lovable.dev", "__LOVABLEDEV__"),
    ("v0.dev", "__V0DEV__"),
    ("Kilo Code", "__KILOCODE__"),
    ("kilo code", "__KILOCODE__"),
    ("Roo Code", "__ROOCODE__"),
    ("roo code", "__ROOCODE__"),
    ("Kimi Code", "__KIMICODE__"),
    ("kimi code", "__KIMICODE__"),
    ("Grok Code", "__GROKCODE__"),
    ("grok code", "__GROKCODE__"),
]

# Reverse lookup: placeholder → original canonical form
_RESTORE_MAP: dict[str, str] = {
    placeholder: canonical
    for canonical, placeholder in PROTECTED_TERMS
    # Keep only one canonical per placeholder (first occurrence wins)
    if placeholder not in {p for _, p in PROTECTED_TERMS[:PROTECTED_TERMS.index((canonical, placeholder))]}
}
# Build clean restore map (deduplicated, first canonical per placeholder)
_RESTORE_MAP = {}
for canonical, placeholder in PROTECTED_TERMS:
    if placeholder not in _RESTORE_MAP:
        _RESTORE_MAP[placeholder] = canonical


# ── Emoticon / Slang Dictionaries ─────────────────────────────────────────────
# Extends ekphrasis's built-in emoticon dict with developer-relevant shorthand

EXTRA_EMOTICONS: dict[str, str] = {
    # Positive
    ":)":  "happy",
    ":-)": "happy",
    ":D":  "very happy",
    "=D":  "very happy",
    ";)":  "winking",
    "<3":  "love",
    "❤️":  "love",
    "👍":  "thumbs up",
    "🔥":  "fire",
    "✅":  "check",
    "🚀":  "rocket",
    # Negative
    ":(" :  "sad",
    ":-(" : "sad",
    ">:(":  "angry",
    "😤":  "frustrated",
    "😡":  "angry",
    "💀":  "dead",
    "🤮":  "disgusted",
    "❌":  "wrong",
    "🗑️":  "trash",
    # Sarcasm / irony signals (keep as-is for sarcasm detection later)
    "🙄":  "eye roll",
    "😒":  "unimpressed",
    "🤡":  "clown",
    # Neutral / technical
    "🤔":  "thinking",
    "💡":  "idea",
    "⚠️":  "warning",
    "🐛":  "bug",
    "🔧":  "fix",
}

SLANG_MAP: dict[str, str] = {
    # General internet slang
    "tbh":   "to be honest",
    "imo":   "in my opinion",
    "imho":  "in my humble opinion",
    "ngl":   "not gonna lie",
    "fr":    "for real",
    "nah":   "no",
    "yea":   "yes",
    "yep":   "yes",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "lol":   "laughing",
    "lmao":  "laughing",
    "rofl":  "laughing",
    "omg":   "oh my god",
    "wtf":   "what the",
    "smh":   "shaking my head",
    "idk":   "i do not know",
    "iirc":  "if i recall correctly",
    "afaik": "as far as i know",
    "afaict":"as far as i can tell",
    "atm":   "at the moment",
    "rn":    "right now",
    "irl":   "in real life",
    "btw":   "by the way",
    "fwiw":  "for what it is worth",
    "tldr":  "summary",
    "tl;dr": "summary",
    # Developer-specific slang
    "pls":   "please",
    "plz":   "please",
    "thx":   "thanks",
    "ty":    "thank you",
    "np":    "no problem",
    "ugh":   "frustrated",
    "meh":   "unimpressed",
    "lgtm":  "looks good to me",
    "wip":   "work in progress",
    "iiuc":  "if i understand correctly",
    "ymmv":  "your experience may vary",
    "yolo":  "doing it anyway",
    "fomo":  "fear of missing out",
    "brb":   "be right back",
    "afk":   "away from keyboard",
}

# Pre-compiled patterns to avoid recompiling on every record.
_SIZE_NOTATION_RE = re.compile(r'(?<!\w)(\d+(?:\.\d+)?(?:gb|mb|tb|b)|t\d+)(?!\w)', re.IGNORECASE)

_SLANG_RE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\b' + re.escape(slang) + r'\b', re.IGNORECASE), expansion)
    for slang, expansion in SLANG_MAP.items()
]

_PROTECT_RE: list[tuple[str, re.Pattern[str], str]] = []
for canonical, placeholder in PROTECTED_TERMS:
    if any(c in canonical for c in "./-"):
        pat = re.compile(r'(?<!\w)' + re.escape(canonical) + r'(?!\w)', re.IGNORECASE)
    else:
        pat = re.compile(r'\b' + re.escape(canonical) + r'\b', re.IGNORECASE)
    _PROTECT_RE.append((canonical.lower(), pat, placeholder))

# ── Ekphrasis Initialisation ──────────────────────────────────────────────────

_processor = None  # Lazy-loaded

def _get_processor():
    global _processor
    if _processor is not None:
        return _processor
    try:
        from ekphrasis.classes.preprocessor import TextPreProcessor
        from ekphrasis.classes.tokenizer import SocialTokenizer
        from ekphrasis.dicts.emoticons import emoticons as ekphrasis_emoticons

        combined_emoticons = {**ekphrasis_emoticons, **EXTRA_EMOTICONS}

        _processor = TextPreProcessor(
            normalize=["url", "email", "percent", "money", "phone",
                       "user", "time", "date", "number"],
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                      "emphasis", "censored"},
            fix_html=True,
            segmenter="twitter",
            corrector="twitter",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_aggressiveness=0,  # conservative — avoids over-correcting technical text
            tokenizer=SocialTokenizer(lowercase=False).tokenize,
            dicts=[combined_emoticons],
        )
        logger.info("ekphrasis TextPreProcessor loaded")
    except ImportError:
        logger.warning(
            "ekphrasis not installed — falling back to regex-only normalisation. "
            "Install with: pip install ekphrasis"
        )
        _processor = None
    return _processor


# ── Normalisation Helpers ─────────────────────────────────────────────────────

def _protect(text: str) -> tuple[str, dict[str, str]]:
    """Replace protected terms with placeholder tokens. Returns modified text and restore map."""
    active_restores: dict[str, str] = {}

    # Protect model/hardware size notation before general term replacement.
    # Examples: 70B, 34B, 128GB, 2.5B, T4.
    size_placeholders: dict[str, str] = {}
    size_counter = 0

    def _replace_size(m: re.Match) -> str:
        nonlocal size_counter
        token = m.group(1)
        key = token.lower()
        if key not in size_placeholders:
            placeholder = f"__SIZE_{size_counter}__"
            size_counter += 1
            size_placeholders[key] = placeholder
            active_restores[placeholder] = token
        return size_placeholders[key]

    text = _SIZE_NOTATION_RE.sub(_replace_size, text)

    lowered = text.lower()
    for canonical_lower, pattern, placeholder in _PROTECT_RE:
        if canonical_lower not in lowered:
            continue

        updated = pattern.sub(placeholder, text)
        if updated != text:
            text = updated
            active_restores[placeholder] = _RESTORE_MAP[placeholder]
            lowered = text.lower()
    return text, active_restores


def _restore(text: str, active_restores: dict[str, str]) -> str:
    """Restore placeholder tokens back to canonical forms."""
    for placeholder, canonical in active_restores.items():
        text = re.sub(re.escape(placeholder), canonical, text, flags=re.IGNORECASE)
    return text


def _apply_slang(text: str) -> str:
    """Replace known slang terms with expanded forms (word-boundary aware)."""
    for pattern, expansion in _SLANG_RE:
        text = pattern.sub(expansion, text)
    return text


def _clean_tags(text: str) -> str:
    """Remove ekphrasis annotation tags like <hashtag>, <elongated>, <allcaps>."""
    text = re.sub(r'<[^>]+>', '', text)
    return text


def _strip_markdown(text: str) -> str:
    """Remove common markdown markers that distort sentence splitting."""
    text = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', text)  # bold/italic
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)  # headers
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # links
    text = re.sub(r'`{1,3}([^`]+)`{1,3}', r'\1', text)  # inline/fenced code markers
    return text


def _collapse_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _regex_fallback_normalise(text: str) -> str:
    """
    Basic normalisation when ekphrasis is not available.
    Handles the most impactful cases: URLs, user mentions, excess punctuation.
    """
    text = re.sub(r'http\S+|www\.\S+', '', text)          # URLs
    text = re.sub(r'@\w+', '', text)                        # user mentions
    text = re.sub(r'#(\w+)', r'\1', text)                   # hashtags → word
    text = re.sub(r'([!?.]){2,}', r'\1', text)             # repeated punctuation
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)             # elongated chars (heeyyy → hey)
    return text


# ── Main Entry Point ──────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """
    Full microtext normalisation pipeline for a single string.
    1. Protect technical terms
    2. Apply slang expansion
    3. ekphrasis normalisation (or regex fallback)
    4. Clean annotation tags
    5. Restore protected terms
    6. Collapse whitespace
    """
    if not text or not text.strip():
        return ""

    # 0 — Strip markdown formatting
    text = _strip_markdown(text)

    # 1 — Protect terms
    text, active_restores = _protect(text)

    # 2 — Slang expansion
    text = _apply_slang(text)

    # 3 — ekphrasis (or fallback)
    processor = _get_processor()
    if processor is not None:
        try:
            tokens = processor.pre_process_doc(text)
            text = " ".join(tokens)
        except Exception as e:
            logger.warning("ekphrasis error, falling back to regex: %s", e)
            text = _regex_fallback_normalise(text)
    else:
        text = _regex_fallback_normalise(text)

    # 4 — Clean tags
    text = _clean_tags(text)

    # 5 — Restore protected terms
    text = _restore(text, active_restores)

    # 6 — Collapse whitespace
    text = _collapse_whitespace(text)

    return text


def _finalise_normalised_text(text: str, active_restores: dict[str, str]) -> str:
    """Apply post-ekphrasis cleanup and placeholder restoration."""
    text = _clean_tags(text)
    text = _restore(text, active_restores)
    text = _collapse_whitespace(text)
    return text


def run(records: list[dict]) -> list[dict]:
    """
    Applies microtext normalisation to record["raw_text"] → record["clean_text"].
    raw_text is written by Step 3 (s3_concat).
    """
    processor = _get_processor()

    # Fast path: batch ekphrasis for non-empty records.
    if processor is not None:
        prepared_texts: list[str] = []
        restore_maps: list[dict[str, str]] = []
        record_indices: list[int] = []

        for idx, record in enumerate(records):
            raw = record.get("raw_text", "") or ""
            if not raw.strip():
                record["clean_text"] = ""
                continue

            text = _strip_markdown(raw)
            text, active_restores = _protect(text)
            text = _apply_slang(text)

            prepared_texts.append(text)
            restore_maps.append(active_restores)
            record_indices.append(idx)

        if prepared_texts:
            try:
                token_batches = processor.pre_process_docs(prepared_texts)
                for idx, tokens, active_restores in zip(record_indices, token_batches, restore_maps):
                    text = " ".join(tokens)
                    records[idx]["clean_text"] = _finalise_normalised_text(text, active_restores)
            except Exception as e:
                logger.warning("ekphrasis batch error, falling back per-record: %s", e)
                for idx in record_indices:
                    raw = records[idx].get("raw_text", "")
                    records[idx]["clean_text"] = normalise(raw)
    else:
        for record in records:
            raw = record.get("raw_text", "")
            record["clean_text"] = normalise(raw)

    logger.info("s4_normalise: normalised %d records", len(records))
    return records
