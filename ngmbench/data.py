from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# TEXMEX .fvecs / .ivecs
def read_fvecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    d = int(raw[0])
    raw = raw.reshape(-1, d + 1)
    return raw[:, 1:].copy().view(np.float32)


def read_ivecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy()


# Bundle
@dataclass
class Dataset:
    name: str
    base: np.ndarray                    # (n, d) float32
    queries: Optional[np.ndarray] = None        # (nq, d) float32
    groundtruth: Optional[np.ndarray] = None     # (nq, k) int32, ids into base

    @property
    def n(self) -> int:
        return self.base.shape[0]

    @property
    def dim(self) -> int:
        return self.base.shape[1]


def load_sift_like(
    name: str,
    base_path: str,
    query_path: Optional[str] = None,
    gt_path: Optional[str] = None,
    base_limit: Optional[int] = None,
) -> Dataset:
    base = read_fvecs(base_path).astype(np.float32)
    if base_limit is not None:
        base = base[:base_limit]
    queries = read_fvecs(query_path).astype(np.float32) if query_path else None
    gt = read_ivecs(gt_path) if gt_path else None
    if gt is not None and base_limit is not None:
        gt = None
    return Dataset(name=name, base=base, queries=queries, groundtruth=gt)


def make_synthetic(
    n: int = 2000,
    dim: int = 16,
    n_clusters: int = 8,
    n_queries: int = 100,
    gt_k: int = 10,
    seed: int = 0,
) -> Dataset:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 10, size=(n_clusters, dim)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n)
    base = (centers[assign] + rng.normal(0, 1, size=(n, dim))).astype(np.float32)
    qassign = rng.integers(0, n_clusters, size=n_queries)
    queries = (centers[qassign] + rng.normal(0, 1, size=(n_queries, dim))).astype(np.float32)
    gt = compute_groundtruth(base, queries, k=gt_k)
    return Dataset(name=f"synthetic_n{n}_d{dim}", base=base, queries=queries, groundtruth=gt)


def compute_groundtruth(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    nq = queries.shape[0]
    gt = np.empty((nq, k), dtype=np.int32)
    b2 = (base * base).sum(axis=1)            # (n,)
    chunk = 256
    for s in range(0, nq, chunk):
        q = queries[s:s + chunk]
        d2 = b2[None, :] - 2.0 * q @ base.T + (q * q).sum(axis=1)[:, None]
        gt[s:s + chunk] = np.argpartition(d2, k, axis=1)[:, :k]
        for i in range(gt[s:s + chunk].shape[0]):
            idx = gt[s + i]
            gt[s + i] = idx[np.argsort(d2[i, idx])]
    return gt


# Partitioning                                                                 #
def partition(
    X: np.ndarray, n_parts: int, method: str = "random", seed: int = 0
) -> List[np.ndarray]:
    n = X.shape[0]
    if n_parts == 1:
        return [np.arange(n)]
    if method == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        return [np.sort(p) for p in np.array_split(perm, n_parts)]
    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=n_parts, random_state=seed, n_init=3, batch_size=2048)
        labels = km.fit_predict(X)
        return [np.where(labels == c)[0] for c in range(n_parts)]
    raise ValueError(f"unknown partition method: {method!r}")
