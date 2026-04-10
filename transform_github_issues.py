"""
Transforms data/anthropics_claude-code_issues.jsonl into a format
compatible with the app's solr_indexer.py transform_record() method.

Output: data/anthropics_claude-code_issues_compatible.jsonl
"""
import json
from pathlib import Path

INPUT_FILE = Path("data/anthropics_claude-code_issues.jsonl")
OUTPUT_FILE = Path("data/anthropics_claude-code_issues_compatible.jsonl")

DEFAULT_LABELS = {
    "subjectivity": None,
    "polarity": None,
    "aspects": [],
    "agents": [],
    "sarcasm": None,
    "annotator": None,
    "labeled_at": None,
}


def transform(record: dict) -> dict:
    record["content_type"] = "post"
    record["platform_context"]["subreddit"] = record["platform_context"]["repo"]
    record["engagement"]["upvotes"] = record["engagement"]["reactions_total"]
    record["engagement"]["num_replies"] = record["engagement"]["num_comments"]
    record["labels"] = DEFAULT_LABELS.copy()
    return record


def main():
    count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                transformed = transform(record)
                fout.write(json.dumps(transformed, ensure_ascii=False) + "\n")
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line {line_num}: {e}")

    print(f"Done. {count} records written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
