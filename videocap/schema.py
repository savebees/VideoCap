"""JSON Schema validation for serialized public artifacts."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Mapping

import jsonschema


_SCHEMAS = {
    "dense_caption": "dense_caption.schema.json",
    "run_manifest": "run_manifest.schema.json",
    "videocap": "videocap.schema.json",
}


def load_schema(name: str) -> dict[str, Any]:
    try:
        filename = _SCHEMAS[name]
    except KeyError as exc:
        raise ValueError(f"unknown schema {name!r}; available: {sorted(_SCHEMAS)}") from exc
    resource = files("videocap.schemas").joinpath(filename)
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_document(document: Mapping[str, Any], schema_name: str) -> None:
    validator = jsonschema.Draft202012Validator(
        load_schema(schema_name),
        format_checker=jsonschema.FormatChecker(),
    )
    validator.validate(dict(document))
