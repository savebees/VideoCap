"""VideoCap command-line interface."""

from __future__ import annotations

import argparse

from videocap.adapters.llm import LLM
from videocap.adapters.vlm import VLM
from videocap.config import Config
from videocap.dataset import load_manifest
from videocap.pipeline import VideoCap
from videocap.runner import run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videocap")
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("run", help="Annotate a local JSONL video manifest")
    command.add_argument("manifest")
    command.add_argument("--config", required=True)
    command.add_argument("--output-root", default="runs")
    command.add_argument("--run-id")
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
    raise AssertionError(f"unknown command: {args.command}")


__all__ = ["main"]
