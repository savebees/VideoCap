"""Optional SODA adapter.

SODA implementations vary by release. This adapter keeps the dependency external
and provides a stable integration point without copying third-party code into the
system. Configure a command that accepts a JSON request on stdin and emits a JSON
object on stdout.
"""

from __future__ import annotations

import json
import subprocess
from typing import Mapping, Sequence

from videocap.contracts import DenseCaptionPrediction
from videocap.protocols import DenseCaptionMetric
from videocap.registry import METRICS


@METRICS.register("soda")
class SODA(DenseCaptionMetric):
    name = "soda"
    version = "0.1"

    def __init__(self, command: Sequence[str] | None = None) -> None:
        self.command = tuple(command or ())

    def score(self, prediction: DenseCaptionPrediction, reference: DenseCaptionPrediction) -> Mapping[str, float]:
        if not self.command:
            raise RuntimeError("SODA is optional; configure an external SODA command")
        payload = {"prediction": prediction.to_dict(), "reference": reference.to_dict()}
        completed = subprocess.run(self.command, input=json.dumps(payload), text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"SODA command failed ({completed.returncode}): {completed.stderr.strip()}")
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or not all(isinstance(value, (int, float)) for value in result.values()):
            raise ValueError("SODA command must emit a JSON object of numeric scores")
        return {str(key): float(value) for key, value in result.items()}
