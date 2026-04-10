# classification/natalie/pipeline.py
import argparse
import json
import logging
import os
import re
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import perf_counter
from typing import Any

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
from classification.natalie.steps.s6_7_sarcasm_placeholder import run as s6_7_sarcasm
from classification.natalie.steps.s7_agents      import run as s7_agents

try:
    import psutil
except Exception:
    psutil = None


_STEP_MODULE_CACHE: dict[str, Any] = {}


def _load_step_module_from_file(filename: str, module_name: str):
    if filename in _STEP_MODULE_CACHE:
        return _STEP_MODULE_CACHE[filename]

    file_path = Path(__file__).resolve().parent / "steps" / filename
    spec = spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load step module from {file_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _STEP_MODULE_CACHE[filename] = module
    return module


def _run_s6_5_original(records: list[dict], scores_buffer: list[dict], stage_device: str) -> list[dict]:
    module = _load_step_module_from_file(
        "s6.5_aspect_detection.py",
        "classification.natalie.steps.s6_5_aspect_original",
    )
    return module.run(records, scores_buffer=scores_buffer, device_preference=stage_device)


def _run_s6_6_original(records: list[dict], scores_buffer: list[dict], stage_device: str) -> list[dict]:
    module = _load_step_module_from_file(
        "s6.6_absa_polarity.py",
        "classification.natalie.steps.s6_6_absa_original",
    )
    return module.run(records, scores_buffer=scores_buffer, device_preference=stage_device)


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


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _process_rss_mb() -> float | None:
    if psutil is None:
        return None
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except Exception:
        return None


def _count_nonempty_lines(path: str) -> int | None:
    try:
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    except Exception as e:
        logger.warning("Could not count records in %s: %s", path, e)
        return None


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", tag).strip("_")
    return cleaned or "run"


def _write_benchmark_artifacts(summary: dict[str, Any], benchmark_dir: str, tag: str | None) -> dict[str, str]:
    os.makedirs(benchmark_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"perf_{_sanitize_tag(tag)}_{ts}" if tag else f"perf_{ts}"
    json_path = os.path.join(benchmark_dir, f"{prefix}.json")
    md_path = os.path.join(benchmark_dir, f"{prefix}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    stage_rows = summary.get("stage_metrics", [])
    lines = [
        "# Classification Pipeline Benchmark",
        "",
        f"- Timestamp: {summary.get('timestamp', '')}",
        f"- Input: {summary.get('input_path', '')}",
        f"- Loaded records: {summary.get('loaded_count', 0)}",
        f"- Processed records: {summary.get('processed_count', 0)}",
        f"- Total elapsed (s): {summary.get('elapsed_seconds', 0.0):.4f}",
        f"- Throughput loaded (records/s): {summary.get('records_per_second_loaded', 0.0):.4f}",
        f"- Throughput processed (records/s): {summary.get('records_per_second_processed', 0.0):.4f}",
        f"- RSS start/end/peak (MB): {summary.get('rss_start_mb')} / {summary.get('rss_end_mb')} / {summary.get('rss_peak_mb')}",
        f"- Stage device: {summary.get('stage_device', 'auto')}",
        "",
        "## Stage Breakdown",
        "",
        "| Stage | In | Out | Elapsed (s) | Records/s (out) | Runtime Share | RSS After (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for stage in stage_rows:
        lines.append(
            "| {name} | {records_in} | {records_out} | {elapsed_seconds:.4f} | {records_per_second_out:.4f} | {runtime_share_pct:.2f}% | {rss_after_mb} |".format(
                name=stage.get("name", ""),
                records_in=stage.get("records_in", 0),
                records_out=stage.get("records_out", 0),
                elapsed_seconds=float(stage.get("elapsed_seconds", 0.0)),
                records_per_second_out=float(stage.get("records_per_second_out", 0.0)),
                runtime_share_pct=float(stage.get("runtime_share_pct", 0.0)),
                rss_after_mb=stage.get("rss_after_mb"),
            )
        )

    estimates = summary.get("estimates", {})
    lines.extend(
        [
            "",
            "## Full Dataset Estimate",
            "",
            f"- Total input records: {estimates.get('full_dataset_records')}",
            f"- Estimate from loaded throughput (s): {estimates.get('estimate_seconds_from_loaded_throughput')}",
            f"- Estimate from processed throughput (s): {estimates.get('estimate_seconds_from_processed_throughput')}",
            "",
            "## Reliability",
            "",
            f"- Error count: {summary.get('error_count', 0)}",
            f"- Success rate: {summary.get('success_rate', 0.0):.4f}",
            "",
        ]
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"benchmark_json": json_path, "benchmark_md": md_path}


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
    enable_step_6_5: bool = True,
    enable_step_6_6: bool = True,
    enable_step_6_7: bool = True,
    stage_device: str = "auto",
    benchmark: bool = False,
    benchmark_dir: str = "classification/natalie/output/perf",
    benchmark_tag: str | None = None,
) -> dict[str, Any]:
    start_time = perf_counter()
    rss_start_mb = _process_rss_mb()
    rss_peak_mb = rss_start_mb

    records, load_meta = load_records(input_path, limit=limit, offset=offset)
    scores_buffer: list[dict] = []
    stage_metrics: list[dict[str, Any]] = []
    error_count = 0

    def _run_stage(name: str, fn) -> None:
        nonlocal records
        nonlocal rss_peak_mb

        in_count = len(records)
        t0 = perf_counter()
        records = fn(records)
        elapsed = perf_counter() - t0
        out_count = len(records)
        rss_after = _process_rss_mb()
        if rss_after is not None and (rss_peak_mb is None or rss_after > rss_peak_mb):
            rss_peak_mb = rss_after
        stage_metrics.append(
            {
                "name": name,
                "records_in": in_count,
                "records_out": out_count,
                "elapsed_seconds": round(elapsed, 6),
                "records_per_second_out": round(_safe_div(out_count, elapsed), 6),
                "rss_after_mb": round(rss_after, 3) if rss_after is not None else None,
            }
        )

    # ── Step 1: Deduplication + field normalisation ──────────────────────────
    _run_stage("s1_filter", lambda rs: s1_filter(rs, log_path="logs/skipped_docids.txt"))

    # ── Step 2: Bot + deleted record filtering ───────────────────────────────
    _run_stage("s2_filter", lambda rs: s2_filter(rs, log_path="logs/skipped_docids.txt"))

    if not records:
        logger.warning("All records filtered out — skipping remaining steps and writing empty output")
        save_records([], output_path)
        save_scores([], scores_path)

        elapsed = perf_counter() - start_time
        rss_end_mb = _process_rss_mb()
        if rss_end_mb is not None and (rss_peak_mb is None or rss_end_mb > rss_peak_mb):
            rss_peak_mb = rss_end_mb

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_path": input_path,
            "output_path": output_path,
            "scores_path": scores_path,
            "limit": limit,
            "offset": offset,
            "loaded_count": int(load_meta["loaded_count"]),
            "processed_count": 0,
            "filtered_count": int(load_meta["loaded_count"]),
            "elapsed_seconds": round(elapsed, 6),
            "records_per_second_loaded": round(_safe_div(int(load_meta["loaded_count"]), elapsed), 6),
            "records_per_second_processed": 0.0,
            "rss_start_mb": round(rss_start_mb, 3) if rss_start_mb is not None else None,
            "rss_end_mb": round(rss_end_mb, 3) if rss_end_mb is not None else None,
            "rss_peak_mb": round(rss_peak_mb, 3) if rss_peak_mb is not None else None,
            "memory_per_record_mb": round(_safe_div(float(rss_peak_mb or 0.0), max(1, int(load_meta["loaded_count"]))), 6),
            "stage_metrics": stage_metrics,
            "stage_device": stage_device,
            "enabled_steps": {
                "s6_5": enable_step_6_5,
                "s6_6": enable_step_6_6,
                "s6_7": enable_step_6_7,
            },
            "error_count": error_count,
            "success_rate": 0.0,
            "estimates": {
                "full_dataset_records": _count_nonempty_lines(input_path),
                "estimate_seconds_from_loaded_throughput": None,
                "estimate_seconds_from_processed_throughput": None,
            },
        }
        artifact_paths: dict[str, str] = {}
        if benchmark:
            artifact_paths = _write_benchmark_artifacts(summary, benchmark_dir, benchmark_tag)

        logger.info("Pipeline complete.")
        return {
            "loaded_count": int(load_meta["loaded_count"]),
            "processed_count": 0,
            "next_offset": int(load_meta["next_offset"]),
            "reached_eof": bool(load_meta["reached_eof"]),
            "benchmark": summary,
            **artifact_paths,
        }

    # ── Step 3: Text concatenation (with comment enrichment) ─────────────────
    _run_stage("s3_concat", s3_concat)

    # ── Step 4: Microtext normalisation ──────────────────────────────────────
    _run_stage("s4_normalise", s4_normalise)

    # ── Step 5: spaCy NLP (SBD, POS, chunking, lemmatisation) ───────────────
    _run_stage("s5_nlp", s5_nlp)

    # ── Step 6: Hybrid subjectivity detection ────────────────────────────────
    _run_stage("s6_subjectivity", lambda rs: s6_subjectivity(rs, scores_buffer=scores_buffer, device_preference=stage_device))

    # ── Step 6.5: Aspect detection (before final agents stage) ───────────────
    if enable_step_6_5:
        _run_stage(
            "s6_5_aspect_detection",
            lambda rs: _run_s6_5_original(rs, scores_buffer=scores_buffer, stage_device=stage_device),
        )

    # ── Step 6.6: ABSA + document polarity (before final agents stage) ───────
    if enable_step_6_6:
        _run_stage(
            "s6_6_absa_polarity",
            lambda rs: _run_s6_6_original(rs, scores_buffer=scores_buffer, stage_device=stage_device),
        )

    # ── Step 6.7: Sarcasm placeholder slot (before final agents stage) ───────
    if enable_step_6_7:
        _run_stage("s6_7_sarcasm", s6_7_sarcasm)

    # ── Step 7: Final keyword-based agent detection ──────────────────────────
    _run_stage("s7_agents", s7_agents)

    # ── Output ────────────────────────────────────────────────────────────────
    save_records(records, output_path)
    save_scores(scores_buffer, scores_path)

    elapsed = perf_counter() - start_time
    rss_end_mb = _process_rss_mb()
    if rss_end_mb is not None and (rss_peak_mb is None or rss_end_mb > rss_peak_mb):
        rss_peak_mb = rss_end_mb

    for stage in stage_metrics:
        stage["runtime_share_pct"] = round(_safe_div(float(stage["elapsed_seconds"]), elapsed) * 100.0, 4)

    loaded_count = int(load_meta["loaded_count"])
    processed_count = len(records)
    records_per_second_loaded = _safe_div(loaded_count, elapsed)
    records_per_second_processed = _safe_div(processed_count, elapsed)
    full_dataset_records = _count_nonempty_lines(input_path)

    estimates = {
        "full_dataset_records": full_dataset_records,
        "estimate_seconds_from_loaded_throughput": None,
        "estimate_seconds_from_processed_throughput": None,
    }
    if isinstance(full_dataset_records, int) and full_dataset_records > 0:
        if records_per_second_loaded > 0:
            estimates["estimate_seconds_from_loaded_throughput"] = round(
                full_dataset_records / records_per_second_loaded,
                3,
            )
        if records_per_second_processed > 0:
            estimates["estimate_seconds_from_processed_throughput"] = round(
                full_dataset_records / records_per_second_processed,
                3,
            )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_path": input_path,
        "output_path": output_path,
        "scores_path": scores_path,
        "limit": limit,
        "offset": offset,
        "loaded_count": loaded_count,
        "processed_count": processed_count,
        "filtered_count": max(0, loaded_count - processed_count),
        "elapsed_seconds": round(elapsed, 6),
        "records_per_second_loaded": round(records_per_second_loaded, 6),
        "records_per_second_processed": round(records_per_second_processed, 6),
        "rss_start_mb": round(rss_start_mb, 3) if rss_start_mb is not None else None,
        "rss_end_mb": round(rss_end_mb, 3) if rss_end_mb is not None else None,
        "rss_peak_mb": round(rss_peak_mb, 3) if rss_peak_mb is not None else None,
        "memory_per_record_mb": round(_safe_div(float(rss_peak_mb or 0.0), max(1, loaded_count)), 6),
        "stage_metrics": stage_metrics,
        "stage_device": stage_device,
        "enabled_steps": {
            "s6_5": enable_step_6_5,
            "s6_6": enable_step_6_6,
            "s6_7": enable_step_6_7,
        },
        "error_count": error_count,
        "success_rate": round(_safe_div(processed_count, loaded_count), 6),
        "estimates": estimates,
    }

    artifact_paths: dict[str, str] = {}
    if benchmark:
        artifact_paths = _write_benchmark_artifacts(summary, benchmark_dir, benchmark_tag)
        logger.info("Benchmark JSON: %s", artifact_paths.get("benchmark_json"))
        logger.info("Benchmark report: %s", artifact_paths.get("benchmark_md"))

    logger.info("Pipeline complete.")
    return {
        "loaded_count": loaded_count,
        "processed_count": processed_count,
        "next_offset": int(load_meta["next_offset"]),
        "reached_eof": bool(load_meta["reached_eof"]),
        "benchmark": summary,
        **artifact_paths,
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
    parser.add_argument(
        "--disable-6-5",
        action="store_true",
        help="Disable step 6.5 aspect detection"
    )
    parser.add_argument(
        "--disable-6-6",
        action="store_true",
        help="Disable step 6.6 ABSA + polarity"
    )
    parser.add_argument(
        "--disable-6-7",
        action="store_true",
        help="Disable step 6.7 placeholder"
    )
    parser.add_argument(
        "--stage-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device preference for heavy model stages (6.5, 6.6)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Write benchmark JSON and Markdown artifacts"
    )
    parser.add_argument(
        "--benchmark-dir",
        default="classification/natalie/output/perf",
        help="Directory for benchmark artifacts"
    )
    parser.add_argument(
        "--benchmark-tag",
        default=None,
        help="Optional short tag included in benchmark filenames"
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
        enable_step_6_5=not args.disable_6_5,
        enable_step_6_6=not args.disable_6_6,
        enable_step_6_7=not args.disable_6_7,
        stage_device=args.stage_device,
        benchmark=args.benchmark,
        benchmark_dir=args.benchmark_dir,
        benchmark_tag=args.benchmark_tag,
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

    if args.benchmark:
        if run_stats.get("benchmark_json"):
            logger.info("Benchmark JSON written: %s", run_stats["benchmark_json"])
        if run_stats.get("benchmark_md"):
            logger.info("Benchmark Markdown written: %s", run_stats["benchmark_md"])