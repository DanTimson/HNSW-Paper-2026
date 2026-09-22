# Constructor comparison — monolithic HNSW vs. layerwise NN-Descent vs. FastHNSW

This summarizes the current experimental result of the constructor-comparison
track: whether building an HNSW graph by running NN-Descent independently on
each preassigned HNSW layer reduces construction cost relative to
conventional monolithic HNSW insertion, and what it costs in search
efficiency. FastHNSW/FastKCNA pg2 is included as a second, more aggressively
refined NN-Descent-derived construction path.

The comparison is made in dataset-distance evaluations (completed squared-L2
metric calls), not total CPU work — see "Reading these numbers" below for
what that does and doesn't support.

## Methods compared

**Monolithic HNSW** — conventional sequential HNSW construction with `M =
16`, `ef_construction = 200`, single-threaded, squared-L2 metric.
Construction distance counts come from HNSWMerger's metric wrapper; its
counter semantics are verified against the pinned upstream HNSWMerger source
(`cpp/HNSWMERGER_PROVENANCE.md`).

**Layerwise NN-Descent HNSW** — the literal per-layer baseline: `K = 500`, `L
= 500`, `S = 12`, `R = 100`, 6 iterations, seed 2024, delta 0.002, controls
100, recall-stop 0.98, `M = 16`, single-threaded. Each nontrivial HNSW layer
is built independently on the pinned FastKCNA/KGraph backend, followed by a
minimal HNSW-compatible diversification/reciprocity conversion.
Construction-search cost is exactly zero by design (see
`cpp/LAYERWISE_NND_HNSW.md`).

**FastHNSW pg2** — the pinned FastKCNA/FastHNSW constructor using the
canonical pg2 path and canonical metric-boundary construction accounting.
Not the same algorithm as the layerwise baseline.

## Main result at 1M vectors

| Method | Construction dataset-distance evaluations | d_s@0.95 |
|---|---:|---:|
| Monolithic HNSW | 3,722,580,816 | 1218.35 |
| Layerwise NN-Descent HNSW | 3,314,236,807 | 1389.41 |
| FastHNSW pg2 | 9,838,110,472 | 1200.04 |

Relative to monolithic HNSW, Layerwise construction uses **10.97% fewer
dataset-distance evaluations** but requires **14.04% more query distance
evaluations at 95% Recall@10**. FastHNSW pg2 requires substantially more
construction distance evaluations than either method but recovers search
efficiency close to monolithic HNSW. The correct reading is a
construction/search-efficiency trade-off, not a universal ordering of total
cost.

## Counter comparability

The Layerwise and HNSWMerger counters instrument different backend-specific
metric choke-points but count the same physical unit: one invocation of the
underlying squared-L2 dataset metric. HNSWMerger increments immediately
before delegating once to its real metric function; the Layerwise/FastKCNA
oracle increments after the corresponding metric evaluation completes. For
successfully completed runs these events are in one-to-one correspondence,
so the construction distance totals are directly comparable as counts of the
same unit — this does not imply equal total CPU, memory, allocation, or
synchronization cost between the implementations.

## Layerwise scaling over the measured range

| N | Total distance evaluations | Evaluations/vector |
|---:|---:|---:|
| 10K | 45,296,135 | 4,529.6 |
| 100K | 381,770,489 | 3,817.7 |
| 1M | 3,314,236,807 | 3,314.2 |

Observed decade ratios: 10K→100K is 8.43x work for 10x more points; 100K→1M
is 8.68x. These three points correspond to a finite-range log-log slope of
about 0.93 — a description of the measured range only, not an asymptotic
complexity estimate; three points don't support one.

The same per-vector decline already appears in the base layer alone (100K:
~352.8M, 1M: ~3.069B, ~8.70x for 10x more vectors, with base-layer `K=L=500`
and six iterations fixed at both scales), so it isn't explained by
upper-layer size or iteration clamping. The exact mechanism behind the
decline hasn't been identified from the existing per-iteration/candidate
counters; it isn't worth a new experiment campaign on its own, but would be
worth a quick look at those counters if it comes up.

## Layerwise 1M construction decomposition

| Phase | Distance evaluations |
|---|---:|
| NN-Descent candidate evaluation | 2,175,905,937 |
| Construction search | 0 |
| Neighbour prune/diversification | 1,115,701,223 |
| Reverse repair | 22,629,647 |
| Other construction | 0 |
| **Total** | **3,314,236,807** |

Candidate generation dominates throughout the measured range; neighbour
pruning/diversification's share grows somewhat with scale; reverse repair
stays below 1% even at 1M.

## Generated figures and tables

`scripts/make_constructor_figures.py --results-dir results --out
docs/figures/constructors` produces:

- `constructor_build_scaling.{png,pdf}` — measured construction distance
  calls versus N (points connected only, no fitted power-law trend).
- `constructor_build_per_vector.{png,pdf}` — construction distance calls per
  input vector.
- `layerwise_phase_per_vector.{png,pdf}` — candidate/prune/repair
  contributions per vector across 10K/100K/1M.
- `constructor_tradeoff_1m.{png,pdf}` — construction cost against
  matched-recall search cost (`d_s@0.95`) for all three methods.
- `recall_vs_ds_1m.{png,pdf}` — the underlying Recall@10/search-distance
  curves.
- `constructor_summary.csv`, `layerwise_scaling.csv`,
  `quality_curves_1m.csv`, and `constructor_figures_manifest.json` (exact
  selected run identities and SHA-256 hashes of every input evidence file).

Evidence selection is explicit rather than "cheapest" or "latest matching
row": each figure/table cross-checks dataset, scale, frozen parameters, and
(where applicable) the exact construction run key behind a quality record,
and fails rather than guessing on ambiguous evidence.

## Reading these numbers

A few things worth being precise about when quoting these results:

- These are dataset-distance-evaluation counts, not total CPU or memory
  cost. The three backends spend that unit through structurally different
  code paths (Layerwise on NN-Descent candidate generation and pruning with
  zero construction-search; monolithic HNSW through sequential
  insertion/traversal/neighbour-selection), so equal counts don't imply
  equal wall-clock cost, and no phase decomposition equivalent to the
  Layerwise table has been established for monolithic HNSW.
- The 10.97% construction saving is a count of fewer distance evaluations,
  not "10.97% less total construction work."
- The 10K–1M scaling range describes three measured points; it doesn't
  establish an asymptotic complexity exponent for NN-Descent.
- FastHNSW's larger construction cost doesn't make it a worse method
  overall — it buys back search efficiency close to monolithic HNSW, which
  is the actual trade-off being measured.

## See also

- `cpp/HNSWMERGER_PROVENANCE.md` — HNSWMerger upstream identity, retained
  diff, and license status.
- `cpp/FASTKCNA_DISTANCE_ACCOUNTING.md` — full source audit of the FastKCNA
  distance-counting patch.
- `cpp/LAYERWISE_NND_HNSW.md` — implementation detail for the layerwise
  baseline.
- `cpp/FASTHNSW_QUALITY_EVALUATION.md` — the shared stock-index quality
  evaluator used for both FastHNSW and the layerwise baseline.
