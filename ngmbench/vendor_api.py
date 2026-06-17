from __future__ import annotations

import os
import sys

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor_repo")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from hnsw import HNSW, heuristic, k_closest, l2_distance  # noqa: E402
from merge_hnsw import (  # noqa: E402
    CGTM,
    IGTM,
    insertion_merge,
    merge_naive,  # NGM
)

# Map the paper's algorithm names to the vendored merge callables.
# Each takes (hnsw_a, hnsw_b, merged_data, **params) and returns a merged HNSW.
MERGE_FUNCS = {
    "NGM": merge_naive,
    "IGTM": IGTM,
    "CGTM": CGTM,
}

__all__ = [
    "HNSW",
    "heuristic",
    "k_closest",
    "l2_distance",
    "CGTM",
    "IGTM",
    "merge_naive",
    "insertion_merge",
    "MERGE_FUNCS",
]
