# CoursePaper2026 — Fast Navigable Graph Construction by Merge Operation

Experiment harness and evidence base for the HSE Nizhny Novgorod master's
course paper *«Быстрое построение навигационных графов, используя операцию
объединения»* (advisor A. A. Ponomarenko). It measures the computational
cost of obtaining an HNSW navigable graph three different ways and how that
cost trades off against search quality.

The Python package is `ngmbench` (see `pyproject.toml`): an experiment
harness for navigable-graph construction by merge (NGM/IGTM/CGTM/SIGM/
HNSW-Merger) versus NN-Descent-based construction (Layerwise NN-Descent,
FastHNSW).

The full results and methodology summary is [`docs/RESULTS.md`](docs/RESULTS.md).

## What is being compared

Three ways of obtaining an HNSW index, all measured by the same metric — a
counter incremented on every squared-L2 distance computation, so results are
comparable across methods and independent of hardware:

1. **Monolithic HNSW** — ordinary sequential insertion (`ngmbench/cli_build_budget.py`).
2. **Partition + merge ("divide and conquer")** — split the data into `P`
   parts, build an HNSW subindex over each part independently, then merge
   pairwise using one of five algorithms: NGM, IGTM, CGTM, SIGM, or
   HNSW-Merger (`ngmbench/cli_cpp.py`, backed by a patched external
   [HNSWMerger](https://github.com/Kimchuls/HNSWMerger)).
3. **Layerwise NN-Descent / FastHNSW** — replace sequential insertion with
   iterative neighbor refinement (NN-Descent) per HNSW layer, either as a
   single candidate-generation pass (Layerwise) or with full KCNA refinement
   at every layer (FastHNSW), both backed by an external
   [FastKCNA](https://github.com/xdyangsh/FastKCNA)/KGraph build
   (`ngmbench/cli_layerwise_nnd.py`, `ngmbench/cli_fasthnsw.py`,
   `ngmbench/cli_fastkcna.py`).

Cost is reported as raw distance-evaluation counts, and quality as Recall@10
/ `d_s@0.95` (distance evaluations per query needed to reach recall 0.95).
Wall-clock time is recorded but treated as a secondary metric, since the
three backends allocate memory and do bookkeeping differently.

## Repository layout

- **`ngmbench/`** — the installable Python package: CLI entry points
  (`cli_cpp`, `cli_build_budget`, `cli_fastkcna`, `cli_fasthnsw`,
  `cli_layerwise_nnd`, `cli_layerwise_nnd_quality`), index-builder wrappers
  under `ngmbench/index/` (one module per backend), plus `distance.py`,
  `quality.py`, `cache.py`, `config.py`, `prepare_bigann.py`.
- **`cpp/`** — the C++ side: patches/instrumentation applied on top of the
  external HNSWMerger checkout (`experiment.cpp`, `dump_graph_level0.cpp`),
  the standalone Layerwise NN-Descent builder and validator, the FastHNSW
  quality evaluator, a `Makefile`, and per-backend provenance/accounting
  notes (`FASTKCNA.md`, `FASTKCNA_DISTANCE_ACCOUNTING.md`,
  `FASTHNSW_QUALITY_EVALUATION.md`, `HNSWMERGER_PROVENANCE.md`,
  `LAYERWISE_NND_HNSW.md`) plus its own `README.md` with build instructions.
- **`scripts/`** — dataset prep (`get_data.sh`, `prepare_dataset.py`),
  figure generation (`make_figures.py`, `make_overall_figures.py`,
  `make_constructor_figures.py`), analysis (`analyse_trends.py`,
  `graph_structure.py`, `xval_python_ref.py` — cross-checks against
  Ponomarenko's reference `merge_hnsw.py`), and smoke/validation scripts for
  the FastKCNA backend.
- **`config/`** — 51 JSON run configs: one per dataset/scale sweep
  (`bigann*`, `deep*`, `gist*`, `turing*` at 10K–10M), build-budget configs,
  and per-backend canonical/quality configs for FastKCNA, FastHNSW, and
  layerwise NN-Descent.
- **`results/`** — 42 JSONL evidence files, one family per experiment
  (partition sweeps, total-cost sweeps, layerwise/FastHNSW canonical and
  quality runs, monolithic quality baselines, etc.). This is the raw
  evidence all figures and tables are generated from.
- **`docs/`** — results, methodology, and reproduction notes (see Further reading below)
  plus generated figures under `docs/figures/`. Some subdirectories there
  (`_components`, `_cross`, `_scale`, `structure`, `synthetic_n1500_d16`,
  `trends`) are exploratory/historical and not part of the current reported
  result set — `sift1m/`, `bigann1m/`, `gist1m/`, `partition/`, `overall/`,
  and `constructors/` are the current ones.
- **`tests/`** — pytest suite (10 files) covering the CLIs, index wrappers,
  quality evaluation, and figure-generation scripts.
- Top-level files: `pyproject.toml` / `requirements.txt` (see Setup),
  `.env.example` (copy to `.env`), `c_leaf.csv` (an example leaf-graph
  structure CSV consumed by `scripts/graph_structure.py`), and
  `hnswmerger.patch` (an already-applied historical diff, kept for the
  record rather than as something to re-apply).

## External dependencies (not vendored)

Two C++ backends are cloned and built outside this repository:

- **HNSWMerger** (Jin et al., *Efficient Vector Index Merging in Vector
  Databases*, PACMMOD 4(1) art. 31), licensed Apache-2.0 upstream — build via
  [`cpp/README.md`](cpp/README.md). It is not vendored here even so: the
  pinned base commit, the exact diff, and the rebuild instructions in
  [`cpp/HNSWMERGER_PROVENANCE.md`](cpp/HNSWMERGER_PROVENANCE.md) are enough
  for reproducibility.
- **FastKCNA/KGraph** — build via [`cpp/FASTKCNA.md`](cpp/FASTKCNA.md).
  Backs both FastHNSW and the layerwise NN-Descent constructor.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # installs ngmbench per pyproject.toml
# or, for the lighter analysis-only path with no C++ backend work:
pip install -r requirements.txt

cp .env.example .env      # then fill in HNSWMERGER_BIN / FASTKCNA_ROOT
                           # once the two backends above are built
```

Datasets: `scripts/get_data.sh` fetches and extracts SIFT1M and GIST1M from
[TEXMEX](http://corpus-texmex.irisa.fr/) (`--sift`, `--gist`, or `--all`).
Deep1M and Turing1M are 1M-vector prefixes cut from the Big-ANN-Benchmarks
Deep1B / MSTuring1B base files via `scripts/prepare_dataset.py --dataset
deep|turing ...` — those base files are not fetched by `get_data.sh` and
need to be obtained separately first.

## Running experiments

Full copy-paste commands (including the total-cost merge experiment and the
layerwise NN-Descent baseline runs) live in
[`docs/REPRODUCING_EXPERIMENTS.md`](docs/REPRODUCING_EXPERIMENTS.md). In
short:

- **Monolithic build budget**: `python -m ngmbench.cli_build_budget --config config/build_budget_*.json`
- **Merge algorithms**: `python -m ngmbench.cli_cpp --config config/<dataset>_sweep.json` (or `config/total_cost_*.json` for the full build+merge+quality pipeline)
- **Layerwise NN-Descent**: `python -m ngmbench.cli_layerwise_nnd --config config/layerwise_nnd_hnsw_canonical_<dataset>.json`, then `ngmbench.cli_layerwise_nnd_quality` against the resulting run key
- **FastHNSW / FastKCNA**: `python -m ngmbench.cli_fasthnsw` / `ngmbench.cli_fastkcna --config config/fasthnsw_quality_<dataset>.json` etc.

## Figures

- `scripts/make_figures.py` — merge cost by algorithm, cross-dataset
  generalization, partition scaling (`docs/figures/<dataset>/`).
- `scripts/make_overall_figures.py` — recursive end-to-end merge-tree
  scaling (`docs/figures/overall/`).
- `scripts/make_constructor_figures.py` — monolithic vs. Layerwise
  NN-Descent vs. FastHNSW comparison (`docs/figures/constructors/`);
  canonical-evidence-only selection, see
  `docs/RESULTS.md`.

## Tests

```bash
pytest -q
```

63 passed, 2 skipped.

## Current headline results

All figures are distance-evaluation counts (squared L2), not wall-clock.

**Merge cost, SIFT1M, P=2** (one pairwise merge, no accumulated tree effect):

| Algorithm   | Merge cost, B distance evals | Share of monolithic build |
|-------------|------------------------------:|---------------------------:|
| HNSW-Merger | 0.194                         | 5.2%                        |
| IGTM        | 0.447                         | 12.0%                       |
| CGTM        | 0.575                         | 15.4%                       |
| NGM         | 0.578                         | 15.5%                       |
| SIGM        | 1.991                         | 53.5%                       |

IGTM is cheapest of NGM/IGTM/CGTM on all four datasets (SIFT1M, Deep1M,
Turing1M, GIST1M); SIGM is always most expensive. HNSW-Merger is tracked
only as an external reference point — it reuses existing edges rather than
merging graph structure, so its low cost isn't directly comparable.

**Partition scaling, SIFT1M**: leaf-build savings grow from 7.4% (P=2) to
28.7% (P=16), but for NGM/IGTM/CGTM the merge overhead already exceeds that
saving at P=2. Only HNSW-Merger stays cheaper than monolithic construction
across the full P range (2.2%–9.0% advantage) — the others may still win on
wall-clock via parallel leaf construction, but not on total operation count.

**Constructor comparison, 1M vectors**:

| Method                    | Construction, B distance evals | d_s@0.95 |
|---------------------------|--------------------------------:|---------:|
| Monolithic HNSW           | 3.7226                          | 1218.35  |
| Layerwise NN-Descent HNSW | 3.3142 (−10.97%)                | 1389.41 (+14.04%) |
| FastHNSW pg2              | 9.8381 (+164.3%)                | 1200.04 (≈ monolithic) |

Layerwise NN-Descent trades cheaper construction for costlier search;
FastHNSW trades much costlier construction for search efficiency close to
monolithic. Neither is a universal win.

## Known limitations

- Distance-evaluation counts are comparable as *the same physical unit*
  across backends (one squared-L2 call), but that is not equivalence of
  total CPU/memory/wall-clock work — the three backends spend that unit
  through structurally different code paths.
- The four-dataset generalization claim (SIFT1M/Deep1M/Turing1M/GIST1M)
  covers the merge-algorithm family only. Layerwise NN-Descent and FastHNSW
  have only ever been run on SIFT, at 10K/100K/1M — never on the other three
  datasets.
- The 10K→100K→1M layerwise scaling trend (~8.4–8.7× per decade) is reported
  as a finite-range measurement, not an asymptotic complexity claim — three
  points don't support one.
- SIGM is intentionally excluded from the P>2 total-cost matrix; see
  `docs/REPRODUCING_EXPERIMENTS.md` for why.
- The real wall-clock benefit of parallel leaf construction in the
  divide-and-conquer scheme has not actually been measured, only the
  operation count.
## Further reading

- [`docs/RESULTS.md`](docs/RESULTS.md) — full results and methodology
  summary: methods, headline numbers, generated figures/tables, and what
  the numbers do and don't support.
- [`cpp/HNSWMERGER_PROVENANCE.md`](cpp/HNSWMERGER_PROVENANCE.md),
  [`cpp/FASTKCNA_DISTANCE_ACCOUNTING.md`](cpp/FASTKCNA_DISTANCE_ACCOUNTING.md),
  [`cpp/LAYERWISE_NND_HNSW.md`](cpp/LAYERWISE_NND_HNSW.md),
  [`cpp/FASTHNSW_QUALITY_EVALUATION.md`](cpp/FASTHNSW_QUALITY_EVALUATION.md)
  — backend-specific provenance and accounting notes.
- [`docs/REPRODUCING_EXPERIMENTS.md`](docs/REPRODUCING_EXPERIMENTS.md) —
  exact, copy-paste commands for the total-cost experiment and the
  layerwise NN-Descent canonical runs.
