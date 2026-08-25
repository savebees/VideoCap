"""VideoCap command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from videocap.adapters.llm import LLM
from videocap.adapters.vlm import VLM
from videocap.config import Config
from videocap.dataset import load_manifest
from videocap.eval import evaluate_dataset
from videocap.io import atomic_write_json, atomic_write_jsonl
from videocap.pipeline import VideoCap
from videocap.qa import derive_qa
from videocap.runner import run


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"record at line {line_number} must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"JSONL file contains no records: {path}")
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videocap")
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("run", help="Annotate a local JSONL video manifest")
    command.add_argument("manifest")
    command.add_argument("--config", required=True)
    command.add_argument("--output-root", default="runs")
    command.add_argument("--run-id")
    evaluate = subcommands.add_parser("eval", help="Evaluate annotations against references")
    evaluate.add_argument("candidates")
    evaluate.add_argument("references")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", default="unspecified")
    qa = subcommands.add_parser("qa", help="Derive grounded QA examples from annotations")
    qa.add_argument("annotations")
    qa.add_argument("--output", required=True)
    qa.add_argument("--split", default="unspecified")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        config = Config.load(args.config)
        samples = load_manifest(args.manifest)
        pipeline = VideoCap(VLM(config.vlm), LLM(config.llm), config.pipeline)
        result = run(
            pipeline,
            samples,
            config,
            args.manifest,
            args.output_root,
            run_id=args.run_id,
        )
        print(result.run_dir)
        return 0
    if args.command == "eval":
        report = evaluate_dataset(
            _read_jsonl(args.candidates),
            _read_jsonl(args.references),
            split=args.split,
        )
        atomic_write_json(Path(args.output).expanduser(), report)
        print(args.output)
        return 0
    if args.command == "qa":
        records = _read_jsonl(args.annotations)
        atomic_write_jsonl(
            Path(args.output).expanduser(),
            (derive_qa(record, split=args.split) for record in records),
        )
        print(args.output)
        return 0
    raise AssertionError(f"unknown command: {args.command}")


__all__ = ["main"]
