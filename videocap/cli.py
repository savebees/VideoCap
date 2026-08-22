"""Command-line entry points for VideoCap production and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from videocap.builtins import load_builtin_components
from videocap.registry import DATASETS, registry_snapshot
from videocap.runner import run_dense_caption


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videocap", description="VideoCap")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("components", help="List registered VideoCap components")

    validate = sub.add_parser(
        "activitynet-validate",
        help="Validate official ActivityNet Captions annotations and report coverage",
    )
    validate.add_argument("annotations", nargs="+", help="One or more annotation JSON files")
    validate.add_argument("--video-manifest")
    validate.add_argument("--video-root")
    validate.add_argument("--split")
    validate.add_argument("--max-samples", type=int)
    validate.add_argument("--check-video-files", action="store_true")

    run = sub.add_parser("run", help="Run VideoCap on a local JSONL video manifest")
    run.add_argument("dataset", help="Local JSONL video manifest")
    run.add_argument("--output-root", default="runs")
    run.add_argument(
        "--metrics",
        default="temporal,lexical-caption,exact-structure",
        help="Comma-separated registered metrics",
    )
    run.add_argument("--config", help="JSON runtime config, including pipeline_config")
    run.add_argument("--run-id")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_builtin_components()
    if args.command == "components":
        print(json.dumps(registry_snapshot(), indent=2))
        return 0
    if args.command == "activitynet-validate":
        dataset = DATASETS.create(
            "activitynet-captions",
            args.annotations,
            video_manifest=args.video_manifest,
            video_root=args.video_root,
            split=args.split,
            max_samples=args.max_samples,
            require_video_files=args.check_video_files,
        )
        samples = list(dataset)
        print(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "version": dataset.version,
                    "fingerprint": dataset.fingerprint(),
                    "samples": len(samples),
                    "references": sum(len(sample.references) for sample in samples),
                    "captions": sum(
                        len(reference.captions)
                        for sample in samples
                        for reference in sample.references
                    ),
                    "duration_hours": round(
                        sum(sample.duration_ms for sample in samples) / 3_600_000,
                        6,
                    ),
                    "video_files_checked": args.check_video_files,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "run":
        run_config = {}
        if args.config:
            config_path = Path(args.config).expanduser().resolve()
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(run_config, dict):
                raise ValueError("--config JSON must contain an object")
        dataset = DATASETS.create("local-jsonl", args.dataset, require_video_files=True)
        result = run_dense_caption(
            dataset,
            args.output_root,
            metric_names=tuple(name.strip() for name in args.metrics.split(",") if name.strip()),
            config=run_config,
            run_id=args.run_id,
        )
        print(result.run_dir)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
