"""Prompt registry.

Prompts are organized per dataset. ``general`` holds the dataset-agnostic
defaults; each dataset module (e.g. ``nwpu_campus``) defines only the prompts it
needs to override, and anything it omits falls back to ``general``.

Select a set with ``get_prompts(config.get("dataset_type"))`` and read prompts as
attributes, e.g. ``prompts.PROMPT_CAPTION_SURVEILLANCE``.
"""

import importlib

from prompts import general

# dataset_type -> module name inside this package
_MODULES = {
    "default": "general",
    "general": "general",
    "nwpu_campus": "nwpu_campus",
}

# dataset_types that are fixed-camera surveillance footage: the pipeline forces
# surveillance mode (fixed-chunk segmentation + surveillance prompts) regardless
# of clip duration, since even short clips are still single-camera surveillance.
_FORCE_SURVEILLANCE = {"nwpu_campus"}


def _normalize(dataset_type: str | None) -> str:
    return (dataset_type or "default").lower()


def is_surveillance_dataset(dataset_type: str | None) -> bool:
    return _normalize(dataset_type) in _FORCE_SURVEILLANCE


class PromptSet:
    """Resolves prompt constants from a dataset module, falling back to general."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name: str):
        if hasattr(self._module, name):
            return getattr(self._module, name)
        return getattr(general, name)


def get_prompts(dataset_type: str | None = None) -> PromptSet:
    key = _normalize(dataset_type)
    if key not in _MODULES:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. Known: {sorted(_MODULES)}"
        )
    module = importlib.import_module(f"prompts.{_MODULES[key]}")
    return PromptSet(module)


# Backward-compat: keep `from prompts import PROMPT_X` working (defaults to general).
from prompts.general import (  # noqa: E402
    PROMPT_SEGMENT,
    PROMPT_CAPTION,
    PROMPT_CAPTION_SURVEILLANCE,
    PROMPT_CAPTION_RETRY,
    PROMPT_ACTIONS,
    PROMPT_ACTIONS_SURVEILLANCE,
    PROMPT_DETECT_OBJECTS,
)
