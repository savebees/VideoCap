"""JSON Schema validation for final VideoCap records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

import jsonschema


def load_schema() -> dict[str, Any]:
    resource = files("videocap.schemas").joinpath("videocap.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_record(document: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(load_schema()).validate(dict(document))


__all__ = ["load_schema", "validate_record"]
