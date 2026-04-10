"""
Step 7 - Agent Labeling (Keyword Matching)

Uses deterministic keyword matching to populate:
    record["labels"]["agents"] -> list[str] | None

If no agent keyword is found, sets labels["agents"] = None.

This module can also run as a standalone CLI for full JSONL processing.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)
_RUNNER_LOGGER = logging.getLogger("natalie.step7_runner")


# (id, canonical_label, keyword_variants)
AGENTS: list[tuple[str, str, list[str]]] = [
    ("1", "copilot", ["copilot", "github copilot", "GithubCopilot", "gh copilot", "co-pilot"]),
    ("2", "cursor", ["cursor", "cursor ai", "cursor editor"]),
    ("3", "claude_code", ["claude code", "claude", "claude cli", "anthropic claude", "ClaudeCode", "claude sonnet", "claude opus", "claude haiku", "claude api"]),
    ("4", "devin", ["devin", "cognition devin", "cognition ai"]),
    ("5", "codewhisperer", ["codewhisperer", "code whisperer", "aws codewhisperer", "amazon codewhisperer"]),
    ("6", "tabnine", ["tabnine", "tab nine"]),
    ("7", "windsurf", ["windsurf", "windsurf ai", "codeium windsurf"]),
    ("8", "chatgpt", ["chatgpt", "gpt", "gpt-5", "gpt-o3", "gpt-4", "gpt-4o", "chat gpt", "gpt-4", "gpt4", "gpt-3.5", "openai chat", "openai api", "openai"]),
    ("9", "gemini", ["gemini", "gemini code", "google gemini", "gemini pro", "gemini ultra"]),
    ("10", "codeium", ["codeium"]),
    ("11", "supermaven", ["supermaven"]),
    ("12", "replit_ai", ["replit ai", "replit ghostwriter", "ghostwriter"]),
    ("13", "aider", ["aider", "aider chat", "aider ai"]),
    ("14", "continue", ["continue.dev", "continue dev", "continuedev"]),
    ("15", "sourcegraph", ["sourcegraph", "cody", "cody ai"]),
    ("16", "amazon_q", ["amazon q", "aws q", "amazon q developer"]),
    ("17", "jetbrains_ai", ["jetbrains ai", "jb ai", "intellij ai", "ai assistant jetbrains"]),
    ("18", "gpt_engineer", ["gpt engineer", "gpt-engineer", "gptengineer"]),
    ("19", "v0", ["v0", "v0.dev", "vercel v0"]),
    ("20", "bolt", ["bolt.new", "bolt ai", "stackblitz bolt"]),
    ("21", "antigravity", ["antigravity", "google_antigravity", "google antigravity", "antigravity ide", "antigravity ai"]),
    ("22", "lovable", ["lovable", "lovable.dev", "lovable ai"]),
    ("24", "kimi_code", ["kimi code", "kimi cli", "kimi k2", "kimi k2.5", "moonshot ai", "moonshot kimi"]),
    ("25", "grok_code", ["grok code", "grok-code", "grok code fast", "xai grok", "grok studio"]),
    ("26", "cline", ["cline", "cline ai", "cline bot", "cline vscode"]),
    ("27", "roo_code", ["roo code", "roocode", "roo coder", "roo ai"]),
    ("28", "kilo_code", ["kilo code", "kilocode", "kilo ai"]),
    ("0", "general", ["ai coding", "llm", "ai tool", "coding assistant", "ai assistant"]),
]


def _compile_keyword_pattern(keyword: str) -> re.Pattern[str]:
    # Allow flexible whitespace while keeping strict token boundaries.
    escaped = re.escape(keyword).replace(r"\ ", r"\s+")

    # If punctuation exists (e.g., continue.dev, co-pilot), avoid word-boundary pitfalls.
    if any(ch in keyword for ch in ".-_"):
        return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)

    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


_AGENT_PATTERNS: list[tuple[str, str, list[re.Pattern[str]]]] = [
    (agent_id, canonical, [_compile_keyword_pattern(k) for k in keywords])
    for agent_id, canonical, keywords in AGENTS
]


def _get_match_text(record: dict) -> str:
    text = (record.get("clean_text") or record.get("raw_text") or "").strip()
    if text:
        return text

    content = record.get("content") or {}
    if not isinstance(content, dict):
        return ""

    parts = [
        content.get("thread_title"),
        content.get("main_text"),
        content.get("reply_text"),
    ]
    return " ".join(str(p).strip() for p in parts if p and str(p).strip())


def _match_agents(text: str) -> list[str] | None:
    if not text:
        return None

    matched: list[str] = []
    for _agent_id, canonical, patterns in _AGENT_PATTERNS:
        if any(p.search(text) for p in patterns):
            matched.append(canonical)

    return matched or None


def run(records: list[dict]) -> list[dict]:
    matched_count = 0
    total_agent_hits = 0

    for record in records:
        labels = record.setdefault("labels", {})
        text = _get_match_text(record)
        agents = _match_agents(text)
        labels["agents"] = agents

        if agents:
            matched_count += 1
            total_agent_hits += len(agents)

    logger.info(
        "s7_agents: %d records processed, %d with agent matches, %d total matches",
        len(records),
        matched_count,
        total_agent_hits,
    )
    return records


def _resolve_input_path(input_path: str) -> str:
    """Resolve input path with a helpful fallback to data/<input_path>."""
    primary = Path(input_path)
    tried: list[Path] = [primary]

    if primary.exists():
        return str(primary)

    if not primary.is_absolute():
        fallback = Path("data") / primary
        tried.append(fallback)
        if fallback.exists():
            _RUNNER_LOGGER.info("Input not found at %s; using %s", primary, fallback)
            return str(fallback)

    tried_str = ", ".join(str(p) for p in tried)
    raise FileNotFoundError(f"Input file not found. Tried: {tried_str}")


def run_step7_only(
    input_path: str,
    output_path: str,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, int | bool]:
    # Local import avoids circular import when pipeline imports this step module.
    from classification.natalie.pipeline import load_records, save_records

    resolved_input_path = _resolve_input_path(input_path)
    records, load_meta = load_records(resolved_input_path, limit=limit, offset=offset)
    records = run(records)

    matched_records = 0
    total_agent_hits = 0
    for record in records:
        agents = (record.get("labels") or {}).get("agents")
        if isinstance(agents, list) and agents:
            matched_records += 1
            total_agent_hits += len(agents)

    save_records(records, output_path)

    stats = {
        "loaded_count": int(load_meta["loaded_count"]),
        "processed_count": len(records),
        "matched_records": matched_records,
        "total_agent_hits": total_agent_hits,
        "next_offset": int(load_meta["next_offset"]),
        "reached_eof": bool(load_meta["reached_eof"]),
    }
    _RUNNER_LOGGER.info(
        "Step 7 complete: processed=%d matched_records=%d total_agent_hits=%d",
        stats["processed_count"],
        stats["matched_records"],
        stats["total_agent_hits"],
    )
    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run only Step 7 agent matching")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N records after --offset",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N records before processing",
    )

    args = parser.parse_args()
    run_step7_only(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        offset=args.offset,
    )


if __name__ == "__main__":
    main()
