from __future__ import annotations

from typing import Tuple

import numpy as np

from .distance import CountingDistance
from .index.base import SubIndex


def recall_at_k(
    sub: SubIndex,
    queries: np.ndarray,
    groundtruth: np.ndarray,
    dist: CountingDistance,
    k: int = 10,
    ef: int = 64,
) -> dict:
    """Return mean recall@k and mean search distance-comps/query for an HNSW."""
    local_to_global = sub.global_ids
    recalls = np.empty(len(queries), dtype=np.float64)
    search_calc = np.empty(len(queries), dtype=np.int64)
    for i, q in enumerate(queries):
        before = dist.snapshot()
        observed = sub.hnsw.search(q=q.astype(np.float32), k=k, ef=ef, return_observed=True)
        search_calc[i] = dist.snapshot() - before
        top = [local_to_global[local_id] for local_id, _ in observed[:k]]
        truth = set(int(t) for t in groundtruth[i, :k])
        recalls[i] = len(truth.intersection(top)) / k
    return {
        f"recall@{k}": float(recalls.mean()),
        "search_calc_per_query": float(search_calc.mean()),
        "eval_ef": ef,
        "eval_k": k,
    }
