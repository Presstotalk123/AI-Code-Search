# classification/natalie/pipeline.py
import argparse
import json
import logging
import os
import re
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("natalie.pipeline")

from classification.natalie.steps.s1_filter      import filter_records as s1_filter
from classification.natalie.steps.s2_bot_filter  import filter_records as s2_filter
from classification.natalie.steps.s3_concat      import run as s3_concat
from classification.natalie.steps.s4_normalise   import run as s4_normalise
from classification.natalie.steps.s5_nlp         import run as s5_nlp
from classification.natalie.steps.s6_subjectivity import run as s6_subjectivity


def _timestamped_path(base_dir: str, stem: str, ext: str = ".jsonl") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{stem}_{ts}{ext}")


def _chunked_path(base_dir: str, stem: str, offset: int, limit: int, ext: str = ".jsonl") -> str:
    start = max(0, offset)
    end = max(start, start + max(1, limit) - 1)
    return os.path.join(base_dir, f"{stem}_{start:07d}_{end:07d}{ext}")


def _read_progress(progress_path: str) -> dict:
    default = {"next_offset": None, "completed": False}
    if not os.path.exists(progress_path):
        return default
    try:
        with open(progress_path, encoding="utf-8") as f:
            data = json.load(f)
        value = data.get("next_offset")
        completed = bool(data.get("completed", False))
        if isinstance(value, int) and value >= 0:
            return {"next_offset": value, "completed": completed}
    except Exception as e:
        logger.warning("Could not read progress file %s: %s", progress_path, e)
    return default


def _write_progress(progress_path: str, next_offset: int, completed: bool) -> None:
    os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
    payload = {
        "next_offset": next_offset,
        "completed": completed,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_records(
    input_path: str,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[dict], dict[str, int | bool]]:
    if offset < 0:
        raise ValueError("offset must be >= 0")

    records = []
    upper_bound = offset + limit if limit is not None else None
    next_offset = offset
    reached_eof = True

    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                next_offset = i + 1
                continue
            if upper_bound is not None and i >= upper_bound:
                reached_eof = False
                next_offset = upper_bound
                break

            next_offset = i + 1
            line = line.strip()
            if not line:
                continue
            # Fix bare null values: "key": , → "key": null,
            line = re.sub(r':\s*,', ': null,', line)
            line = re.sub(r':\s*}', ': null}', line)
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d: %s", i + 1, e)
    logger.info(
        "Loaded %d records from %s (offset=%d, limit=%s)",
        len(records),
        input_path,
        offset,
        "None" if limit is None else limit,
    )
    meta = {
        "loaded_count": len(records),
        "next_offset": next_offset,
        "reached_eof": reached_eof,
    }
    return records, meta


def save_records(records: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Keep all fields in final output (including raw_text and nlp).
    _INTERNAL_FIELDS = set()
    _INTERNAL_LABEL_FIELDS = {"annotator", "labeled_at", "agreement"}

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            out = {k: v for k, v in r.items() if k not in _INTERNAL_FIELDS}
            labels = out.get("labels")
            if isinstance(labels, dict):
                out["labels"] = {
                    k: v for k, v in labels.items()
                    if k not in _INTERNAL_LABEL_FIELDS
                }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), output_path)


def save_scores(scores_buffer: list[dict], scores_path: str) -> None:
    """
    Write intermediate scores file (scores.jsonl).
    Used by the indexer for confidence-weighted ranking.
    Not part of the final label schema.
    """
    if not scores_buffer:
        return
    os.makedirs(os.path.dirname(scores_path) or ".", exist_ok=True)

    # Merge score entries by doc_id
    merged: dict[str, dict] = {}
    for entry in scores_buffer:
        doc_id = entry.get("doc_id", "UNKNOWN")
        if doc_id not in merged:
            merged[doc_id] = {"doc_id": doc_id}
        merged[doc_id].update({k: v for k, v in entry.items() if k != "doc_id"})

    with open(scores_path, "w", encoding="utf-8") as f:
        for entry in merged.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Saved %d score entries → %s", len(merged), scores_path)


def run(
    input_path:  str,
    output_path: str,
    scores_path: str,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, int | bool]:

    records, load_meta = load_records(input_path, limit=limit, offset=offset)

    # ── Step 1: Deduplication + field normalisation ──────────────────────────
    records = s1_filter(records, log_path="logs/skipped_docids.txt")

    # ── Step 2: Bot + deleted record filtering ───────────────────────────────
    records = s2_filter(records, log_path="logs/skipped_docids.txt")

    if not records:
        logger.warning("All records filtered out — skipping remaining steps and writing empty output")
        save_records([], output_path)
        save_scores([], scores_path)
        logger.info("Pipeline complete.")
        return {
            "loaded_count": int(load_meta["loaded_count"]),
            "processed_count": 0,
            "next_offset": int(load_meta["next_offset"]),
            "reached_eof": bool(load_meta["reached_eof"]),
        }

    # ── Step 3: Text concatenation (with comment enrichment) ─────────────────
    records = s3_concat(records)

    # ── Step 4: Microtext normalisation ──────────────────────────────────────
    records = s4_normalise(records)

    # ── Step 5: spaCy NLP (SBD, POS, chunking, lemmatisation) ───────────────
    records = s5_nlp(records)

    # ── Step 6: Hybrid subjectivity detection ────────────────────────────────
    scores_buffer: list[dict] = []
    records = s6_subjectivity(records, scores_buffer=scores_buffer)

    # ── Output ────────────────────────────────────────────────────────────────
    save_records(records, output_path)
    save_scores(scores_buffer, scores_path)

    logger.info("Pipeline complete.")
    return {
        "loaded_count": int(load_meta["loaded_count"]),
        "processed_count": len(records),
        "next_offset": int(load_meta["next_offset"]),
        "reached_eof": bool(load_meta["reached_eof"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Natalie's classification pipeline — Person 1")
    parser.add_argument("--input",   required=True,  help="Path to raw JSONL input")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write Person 1 output JSONL (default: timestamped under classification/natalie/output)"
    )
    parser.add_argument(
        "--scores",
        default=None,
        help="Path to write intermediate scores JSONL (default: timestamped under classification/natalie/output)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N records after --offset"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N input lines before processing"
    )
    parser.add_argument(
        "--auto-next",
        action="store_true",
        help="Use and update a progress file to auto-advance offset between chunk runs"
    )
    parser.add_argument(
        "--progress-file",
        default="classification/natalie/output/chunks/progress.json",
        help="Path to progress file used by --auto-next"
    )
    args = parser.parse_args()

    effective_offset = args.offset
    if args.auto_next:
        progress = _read_progress(args.progress_file)
        if progress.get("completed"):
            logger.info("Progress indicates all chunks are complete. Nothing left to process.")
            raise SystemExit(0)
        saved_offset = progress.get("next_offset")
        if isinstance(saved_offset, int):
            effective_offset = saved_offset
        logger.info("Using offset %d (auto-next=%s)", effective_offset, args.auto_next)

    if args.output:
        output_path = args.output
    elif args.limit is not None:
        output_path = _chunked_path(
            "classification/natalie/output/chunks",
            "natalie_output",
            effective_offset,
            args.limit,
        )
    else:
        output_path = _timestamped_path("classification/natalie/output", "natalie_output")

    if args.scores:
        scores_path = args.scores
    elif args.limit is not None:
        scores_path = _chunked_path(
            "classification/natalie/output/chunks",
            "scores",
            effective_offset,
            args.limit,
        )
    else:
        scores_path = _timestamped_path("classification/natalie/output", "scores")

    logger.info("Output path: %s", output_path)
    logger.info("Scores path: %s", scores_path)

    run_stats = run(
        input_path=args.input,
        output_path=output_path,
        scores_path=scores_path,
        limit=args.limit,
        offset=effective_offset,
    )

    if args.auto_next and args.limit is not None:
        next_offset = int(run_stats["next_offset"])
        reached_eof = bool(run_stats["reached_eof"])
        completed = reached_eof

        _write_progress(args.progress_file, next_offset, completed=completed)
        logger.info(
            "Updated progress file: %s (next_offset=%d, completed=%s)",
            args.progress_file,
            next_offset,
            completed,
        )
        if completed:
            logger.info("Reached end of file at offset %d — all records processed", next_offset)