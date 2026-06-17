from __future__ import annotations

import numpy as np


class CountingDistance:

    __slots__ = ("count", "_metric")

    def __init__(self, metric: str = "l2"):
        self.count = 0
        self._metric = metric

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        self.count += 1
        return float(np.linalg.norm(a - b))

    def snapshot(self) -> int:
        return self.count

    def reset(self) -> None:
        self.count = 0
