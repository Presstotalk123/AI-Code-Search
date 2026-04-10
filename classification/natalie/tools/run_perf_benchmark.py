from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from classification.natalie import pipeline as natalie_pipeline


def _read_nonempty_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)
    return lines


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_manifest(path: Path, selected_lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in selected_lines:
            try:
                record = json.loads(line)
                doc_id = str(record.get("doc_id", ""))
            except Exception:
                doc_id = ""
            f.write(doc_id + "\n")


def _default_sample_path(sample_size: int, seed: int, output_dir: Path) -> Path:
    return output_dir / f"sample_{sample_size}_seed{seed}.jsonl"


def _default_manifest_path(sample_size: int, seed: int, output_dir: Path) -> Path:
    return output_dir / f"sample_{sample_size}_seed{seed}_docids.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic benchmark sample and run pipeline")
    parser.add_argument("--input", default="data/merged_output_raw.jsonl", help="Input JSONL file")
    parser.add_argument("--sample-size", type=int, default=300, help="Sample size for benchmark run")
    parser.add_argument("--sample-mode", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for random mode")
    parser.add_argument(
        "--output-dir",
        default="classification/natalie/output/perf",
        help="Directory for sampled input and benchmark artifacts",
    )
    parser.add_argument("--sample-path", default=None, help="Optional explicit sample JSONL output path")
    parser.add_argument("--manifest-path", default=None, help="Optional explicit doc_id manifest path")
    parser.add_argument("--pipeline-output", default=None, help="Optional pipeline labelled output JSONL path")
    parser.add_argument("--pipeline-scores", default=None, help="Optional pipeline scores JSONL path")
    parser.add_argument("--stage-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-6-5", action="store_true", help="Disable 6.5 aspect stage")
    parser.add_argument("--disable-6-6", action="store_true", help="Disable 6.6 ABSA stage")
    parser.add_argument("--disable-6-7", action="store_true", help="Disable 6.7 placeholder stage")
    parser.add_argument(
        "--benchmark-tag",
        default=None,
        help="Optional benchmark tag. Default uses mode, size, seed, and device",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only create sample and manifest; do not run the pipeline",
    )

    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be > 0")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = Path(args.sample_path) if args.sample_path else _default_sample_path(args.sample_size, args.seed, output_dir)
    manifest_path = Path(args.manifest_path) if args.manifest_path else _default_manifest_path(args.sample_size, args.seed, output_dir)

    all_lines = _read_nonempty_lines(input_path)
    total = len(all_lines)
    if args.sample_size > total:
        raise ValueError(f"Requested {args.sample_size} records but input has only {total}")

    if args.sample_mode == "first":
        selected_lines = all_lines[: args.sample_size]
    else:
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(total), args.sample_size))
        selected_lines = [all_lines[i] for i in indices]

    _write_lines(sample_path, selected_lines)
    _write_manifest(manifest_path, selected_lines)

    print(f"Input records: {total}")
    print(f"Sample records: {len(selected_lines)}")
    print(f"Sample written: {sample_path}")
    print(f"Manifest written: {manifest_path}")

    if args.skip_run:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_output = args.pipeline_output or str(output_dir / f"benchmark_output_{ts}.jsonl")
    pipeline_scores = args.pipeline_scores or str(output_dir / f"benchmark_scores_{ts}.jsonl")

    benchmark_tag = args.benchmark_tag
    if not benchmark_tag:
        benchmark_tag = f"{args.sample_mode}_{args.sample_size}_seed{args.seed}_{args.stage_device}"

    run_stats = natalie_pipeline.run(
        input_path=str(sample_path),
        output_path=pipeline_output,
        scores_path=pipeline_scores,
        enable_step_6_5=not args.disable_6_5,
        enable_step_6_6=not args.disable_6_6,
        enable_step_6_7=not args.disable_6_7,
        stage_device=args.stage_device,
        benchmark=True,
        benchmark_dir=str(output_dir),
        benchmark_tag=benchmark_tag,
    )

    print("Pipeline run complete")
    print(f"Pipeline output: {pipeline_output}")
    print(f"Pipeline scores: {pipeline_scores}")
    if run_stats.get("benchmark_json"):
        print(f"Benchmark JSON: {run_stats['benchmark_json']}")
    if run_stats.get("benchmark_md"):
        print(f"Benchmark Markdown: {run_stats['benchmark_md']}")


if __name__ == "__main__":
    main()
