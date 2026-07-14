"""Per-dataset prompt registry. ``general`` holds defaults; a dataset module
overrides only what it needs. Select with ``get_prompts(config["dataset_type"])``.
"""

import importlib

from prompts import general

_MODULES = {
    "default": "general",
    "general": "general",
    "nwpu_campus": "nwpu_campus",
    "physics_iq": "physics_iq",
}

# Datasets forced into surveillance mode (fixed chunks + surveillance prompts)
# regardless of duration. physics_iq is fixed-camera too but event-dense, so it
# takes the normal short-video path.
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


# Backward-compat: `from prompts import PROMPT_X` resolves to general.
from prompts.general import (  # noqa: E402
    PROMPT_SEGMENT,
    PROMPT_CAPTION,
    PROMPT_CAPTION_SURVEILLANCE,
    PROMPT_CAPTION_RETRY,
    PROMPT_ACTIONS,
    PROMPT_ACTIONS_SURVEILLANCE,
    PROMPT_DETECT_OBJECTS,
)
