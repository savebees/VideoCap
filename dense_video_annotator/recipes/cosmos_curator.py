"""Adapter for NVIDIA Cosmos Curator's batch video-caption pipeline.

Cosmos Curator remains an external dependency. This module invokes its official
configuration entry point and converts per-clip metadata into the package's
stable temporal dense-caption contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dense_video_annotator.contracts import DenseCaptionPrediction, TemporalCaption
from dense_video_annotator.environment import resolve_cosmos_command
from dense_video_annotator.io import atomic_write_json
from dense_video_annotator.protocols import (
    DenseCaptionOutcome,
    DenseCaptionRecipe,
    DenseCaptionSample,
)
from dense_video_annotator.registry import RECIPES


_TESTED_UPSTREAM_REVISION = "975b68910f23067fb2391e68368d5f7cd8cf64ce"
_USABLE_STATUSES = frozenset({"success", "truncated"})
_QUALITY_FLAGS = (
    "flag_length_outlier",
    "flag_repetition",
    "flag_near_duplicate",
)
_RESERVED_COSMOS_ARGS = frozenset(
    {
        "pipeline",
        "input_video_path",
        "input_video_list_json_path",
        "output_clip_path",
        "upload_clip_info_in_chunks",
        "upload_clip_info_in_lance",
        "dry_run",
        "generate_captions",
    }
)


def _recipe_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("recipe_config", {})
    if not isinstance(value, Mapping):
        raise TypeError("recipe_config must be an object")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _write_json_array(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(list(values), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _select_caption(window: Mapping[str, Any], caption_field: str | None) -> tuple[str, str]:
    if caption_field:
        text = window.get(caption_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"usable window has no text in caption_field {caption_field!r}")
        return caption_field, text.strip()

    enhanced = sorted(
        key
        for key, value in window.items()
        if key.endswith("_enhanced_caption") and isinstance(value, str) and value.strip()
    )
    if len(enhanced) == 1:
        key = enhanced[0]
        return key, str(window[key]).strip()
    if len(enhanced) > 1:
        raise ValueError(
            "multiple enhanced caption fields found; set recipe_config.caption_field"
        )

    raw = sorted(
        key
        for key, value in window.items()
        if key.endswith("_caption")
        and not key.endswith("_enhanced_caption")
        and isinstance(value, str)
        and value.strip()
    )
    if len(raw) == 1:
        key = raw[0]
        return key, str(window[key]).strip()
    if len(raw) > 1:
        raise ValueError("multiple caption fields found; set recipe_config.caption_field")
    raise ValueError("usable window contains no caption text")


def _source_key(source: str, input_root: Path | None) -> Path:
    path = Path(source).expanduser()
    if not path.is_absolute() and input_root is not None:
        path = input_root / path
    return path.resolve()


@RECIPES.register("cosmos-curator")
class CosmosCuratorRecipe(DenseCaptionRecipe):
    """Run or import one local Cosmos Curator split-and-caption batch."""

    name = "cosmos-curator"
    version = "0.1"

    def __init__(
        self,
        *,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self._command_runner = command_runner

    def produce(
        self,
        samples: Sequence[DenseCaptionSample],
        work_dir: Path,
        config: Mapping[str, Any],
    ) -> Sequence[DenseCaptionOutcome]:
        if not samples:
            return ()
        recipe_config = _recipe_config(config)
        mode = recipe_config.get("mode", "run")
        if mode not in {"run", "import"}:
            raise ValueError("recipe_config.mode must be 'run' or 'import'")

        caption_field_value = recipe_config.get("caption_field")
        if caption_field_value is not None and (
            not isinstance(caption_field_value, str) or not caption_field_value.strip()
        ):
            raise ValueError("recipe_config.caption_field must be a non-empty string")
        caption_field = caption_field_value.strip() if caption_field_value else None
        declared_revision = recipe_config.get("upstream_revision")
        if declared_revision is not None and (
            not isinstance(declared_revision, str) or not declared_revision.strip()
        ):
            raise ValueError("recipe_config.upstream_revision must be a non-empty string")

        if mode == "run":
            output_path, input_root = self._run_batch(samples, work_dir, recipe_config)
        else:
            output_value = recipe_config.get("output_path")
            if not isinstance(output_value, str) or not output_value.strip():
                raise ValueError("import mode requires recipe_config.output_path")
            if "://" in output_value:
                raise ValueError("import mode currently requires a local output_path")
            output_path = Path(output_value).expanduser().resolve()
            root_value = recipe_config.get("input_video_path")
            input_root = (
                Path(root_value).expanduser().resolve()
                if isinstance(root_value, str) and root_value.strip()
                else None
            )

        return self._convert_outputs(
            samples,
            output_path=output_path,
            input_root=input_root,
            caption_field=caption_field,
            mode=str(mode),
            declared_revision=declared_revision,
        )

    def _run_batch(
        self,
        samples: Sequence[DenseCaptionSample],
        work_dir: Path,
        config: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        paths = [sample.video_path.expanduser().resolve() for sample in samples]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(f"input videos do not exist: {preview}")

        configured_root = config.get("input_video_path")
        if configured_root is not None:
            if not isinstance(configured_root, str) or not configured_root.strip():
                raise ValueError("recipe_config.input_video_path must be a non-empty string")
            input_root = Path(configured_root).expanduser().resolve()
        else:
            common = Path(os.path.commonpath([str(path) for path in paths]))
            input_root = common if common.is_dir() else common.parent
        if input_root == Path(input_root.anchor):
            raise ValueError(
                "refusing to use a filesystem root as input_video_path; configure a narrower root"
            )
        if not input_root.is_dir():
            raise NotADirectoryError(f"input_video_path is not a directory: {input_root}")
        for path in paths:
            try:
                path.relative_to(input_root)
            except ValueError as exc:
                raise ValueError(f"video is outside input_video_path: {path}") from exc

        work_dir.mkdir(parents=True, exist_ok=False)
        input_manifest = work_dir / "input_videos.json"
        output_path = work_dir / "cosmos_output"
        cosmos_config_path = work_dir / "cosmos_config.json"
        _write_json_array(input_manifest, [str(path) for path in paths])

        extra_args = config.get("cosmos_args", {})
        if not isinstance(extra_args, Mapping):
            raise TypeError("recipe_config.cosmos_args must be an object")
        reserved = sorted(_RESERVED_COSMOS_ARGS.intersection(extra_args))
        if reserved:
            raise ValueError(
                "recipe_config.cosmos_args cannot override adapter-owned fields: "
                + ", ".join(reserved)
            )
        cosmos_config = {
            "pipeline": "split",
            "args": {
                "input_video_path": str(input_root),
                "input_video_list_json_path": str(input_manifest),
                "output_clip_path": str(output_path),
                "upload_clip_info_in_chunks": False,
                "upload_clip_info_in_lance": False,
                "dry_run": False,
                "generate_captions": True,
                "generate_embeddings": False,
                **dict(extra_args),
            },
        }
        atomic_write_json(cosmos_config_path, cosmos_config)

        explicit_command = config.get("command")
        if explicit_command is not None and (
            not isinstance(explicit_command, Sequence)
            or isinstance(explicit_command, (str, bytes))
        ):
            raise TypeError("recipe_config.command must be an array of command tokens")
        environment_path = config.get("environment_path")
        if environment_path is not None and (
            not isinstance(environment_path, str) or not environment_path.strip()
        ):
            raise TypeError("recipe_config.environment_path must be a non-empty string")
        command_value = resolve_cosmos_command(
            cosmos_config_path,
            explicit_command,
            environment_path=environment_path,
        )
        if command_value is None:
            command_value = [
                sys.executable,
                "-m",
                "cosmos_curator.pipelines.video.run_pipeline",
                "{config_path}",
            ]
        if not isinstance(command_value, Sequence) or isinstance(command_value, (str, bytes)):
            raise TypeError("recipe_config.command must be an array of command tokens")
        replacements = {
            "config_path": str(cosmos_config_path),
            "input_manifest_path": str(input_manifest),
            "input_video_path": str(input_root),
            "output_path": str(output_path),
        }
        command: list[str] = []
        for token in command_value:
            if not isinstance(token, str) or not token:
                raise ValueError("recipe_config.command tokens must be non-empty strings")
            try:
                command.append(token.format_map(replacements))
            except KeyError as exc:
                raise ValueError(f"unknown command placeholder: {exc.args[0]}") from exc
        if not command:
            raise ValueError("recipe_config.command must not be empty")

        timeout_value = config.get("timeout_sec")
        if timeout_value is not None and (
            isinstance(timeout_value, bool)
            or not isinstance(timeout_value, (int, float))
            or timeout_value <= 0
        ):
            raise ValueError("recipe_config.timeout_sec must be a positive number")
        try:
            result = self._command_runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_value,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Cosmos Curator command is unavailable. Install this package inside "
                "the official Cosmos Curator Pixi/container environment, or use import mode."
            ) from exc
        (work_dir / "cosmos_stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (work_dir / "cosmos_stderr.log").write_text(result.stderr or "", encoding="utf-8")
        if result.returncode != 0:
            combined_output = result.stderr or result.stdout or ""
            if "No module named" in combined_output and "cosmos_curator" in combined_output:
                raise RuntimeError(
                    "Cosmos Curator is not installed in this environment. Install this "
                    "package inside the official Cosmos Curator Pixi/container environment, "
                    "or use import mode."
                )
            tail = combined_output.strip()[-1000:]
            raise RuntimeError(
                f"Cosmos Curator exited with code {result.returncode}"
                + (f": {tail}" if tail else "")
            )
        return output_path, input_root

    def _convert_outputs(
        self,
        samples: Sequence[DenseCaptionSample],
        *,
        output_path: Path,
        input_root: Path | None,
        caption_field: str | None,
        mode: str,
        declared_revision: Any,
    ) -> Sequence[DenseCaptionOutcome]:
        metadata_root = output_path / "metas" / "v0"
        if not metadata_root.is_dir():
            raise FileNotFoundError(f"Cosmos Curator metadata directory not found: {metadata_root}")

        samples_by_path = {
            sample.video_path.expanduser().resolve(): sample for sample in samples
        }
        if len(samples_by_path) != len(samples):
            raise ValueError("Cosmos Curator recipe requires unique video_path values")
        records: dict[str, list[tuple[int, int, str, str, Mapping[str, Any]]]] = defaultdict(list)
        status_counts: dict[str, Counter[str]] = defaultdict(Counter)
        clip_counts: Counter[str] = Counter()
        errors: dict[str, list[str]] = defaultdict(list)

        for metadata_path in sorted(metadata_root.glob("*.json")):
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(raw, Mapping):
                    raise TypeError("clip metadata must be an object")
                source = raw.get("source_video")
                if not isinstance(source, str) or not source.strip():
                    raise ValueError("clip metadata has no source_video")
                sample = samples_by_path.get(_source_key(source, input_root))
                if sample is None:
                    continue
                video_id = sample.video_id
                clip_counts[video_id] += 1
                clip_start_ns = _require_int(raw.get("start_ns"), "clip start_ns")
                windows = raw.get("windows", [])
                if not isinstance(windows, list):
                    raise TypeError("clip windows must be an array")
                for window_index, window in enumerate(windows):
                    if not isinstance(window, Mapping):
                        errors[video_id].append(
                            f"{metadata_path.name} window {window_index}: window must be an object"
                        )
                        continue
                    status = window.get("caption_status", "unknown")
                    status_key = str(status)
                    status_counts[video_id][status_key] += 1
                    if status_key not in _USABLE_STATUSES:
                        continue
                    try:
                        field, text = _select_caption(window, caption_field)
                        relative_start = _require_int(
                            window.get("start_ns"), "window start_ns"
                        )
                        relative_end = _require_int(window.get("end_ns"), "window end_ns")
                        global_start = clip_start_ns + relative_start
                        global_end = clip_start_ns + relative_end
                        start_ms = max(0, global_start // 1_000_000)
                        end_ms = min(
                            sample.duration_ms,
                            max(0, (global_end + 999_999) // 1_000_000),
                        )
                        if end_ms <= start_ms:
                            raise ValueError("caption interval is empty or outside video duration")
                        quality = {
                            flag: window.get(flag)
                            for flag in _QUALITY_FLAGS
                            if window.get(flag) is not None
                        }
                        records[video_id].append(
                            (start_ms, end_ms, text, field, quality)
                        )
                    except (TypeError, ValueError) as exc:
                        errors[video_id].append(
                            f"{metadata_path.name} window {window_index}: {exc}"
                        )
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid Cosmos metadata {metadata_path}: {exc}") from exc

        outcomes: list[DenseCaptionOutcome] = []
        for sample in samples:
            video_records = sorted(
                records.get(sample.video_id, ()),
                key=lambda item: (item[0], item[1], item[2], item[3]),
            )
            production = {
                "recipe": self.name,
                "mode": mode,
                "adapter_tested_upstream_revision": _TESTED_UPSTREAM_REVISION,
                "declared_upstream_revision": declared_revision,
                "clip_count": clip_counts[sample.video_id],
                "caption_status_counts": dict(sorted(status_counts[sample.video_id].items())),
                "conversion_errors": errors[sample.video_id],
            }
            if not video_records:
                message = "Cosmos Curator produced no usable caption windows"
                if errors[sample.video_id]:
                    message += ": " + "; ".join(errors[sample.video_id][:3])
                outcomes.append(
                    DenseCaptionOutcome(
                        video_id=sample.video_id,
                        error_type="NoUsableCaptions",
                        message=message,
                        metadata=production,
                    )
                )
                continue

            captions: list[TemporalCaption] = []
            caption_details: list[Mapping[str, Any]] = []
            for index, (start_ms, end_ms, text, field, quality) in enumerate(
                video_records, start=1
            ):
                caption_id = f"c_{index:04d}"
                captions.append(
                    TemporalCaption(
                        caption_id=caption_id,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=text,
                    )
                )
                caption_details.append(
                    {"caption_id": caption_id, "source_field": field, "quality_flags": quality}
                )
            production["captions"] = caption_details
            outcomes.append(
                DenseCaptionOutcome(
                    video_id=sample.video_id,
                    prediction=DenseCaptionPrediction(
                        video_id=sample.video_id,
                        duration_ms=sample.duration_ms,
                        captions=tuple(captions),
                    ),
                    metadata=production,
                )
            )
        return tuple(outcomes)
