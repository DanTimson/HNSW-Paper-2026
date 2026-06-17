# Fast Navigable Graph Construction by Merge Operation — reproduction code

Code, configs, and figure pipeline behind the course paper *"Fast Navigable Graph
Construction by Merge Operation"* (RU: «Быстрое построение навигационных графов используя операцию объединения»). It reproduces the SIFT1M / GIST1M comparison of HNSW construction by graph **merge** — divide-and-conquer over the NGM / IGTM / CGTM / ES family, plus a `TWO_MERGE` variant and an insertion baseline — against **NN-Descent**.

## What this reproduces


- **distance-computation counts** (`build_calc` / `merge_calc` / `total_calc`) for the   merge family on full SIFT1M and GIST1M — the language-independent cost metric;
- **recall@10** and the **recall-vs-`ef` curve** under greedy search;
- **build / merge wall-clock**;
- the **NN-Descent baseline** (recall + wall-clock; its distance count is `null` by
  design — see *Interpreting the logs*).

Results accumulate in JSONL logs; `make_figures.py` turns them into the per-dataset figure sets under `docs/figures/{sift1m,gist1m}/`.

## Setup

- Python ≥ 3.10
- A C++ toolchain (`g++` with OpenMP) for the merge backend
- ~4 GB free disk for the datasets (more for GIST)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[playground]"      # core deps + the Streamlit results browser
```

### Datasets

```bash
scripts/get_data.sh --all           # SIFT ~168 MB, GIST ~2.6 GB; or --sift / --gist
```

### Build the merge backend 

The merge-family numbers come from the [HNSWMerger](https://github.com/Kimchuls/HNSWMerger) C++ tool, which implements NGM / IGTM / CGTM, the Elasticsearch-style merge, and the rebuild/insert baselines in one codebase.

```bash
git clone https://github.com/Kimchuls/HNSWMerger.git
cd HNSWMerger/HNSW-Merger
make build && make exp              # produces ./builds and ./exps

# GIST1M is not a built-in workload — patch the config and force a clean rebuild:
python /path/to/this-repo/scripts/patch_hnswmerger_gist.py test_config.h
make clean && make build && make exp
```

### Merge family, full scale (C++)

Edit the `binaries` and `dataset` paths in `config/sift1m_cpp.json` and
`config/gist1m_cpp.json` to point at your `./builds`, `./exps`, and the data files,
then:

```bash
python -m ngmbench.cli_cpp --config config/sift1m_cpp.json    # -> results_cpp.jsonl
python -m ngmbench.cli_cpp --config config/gist1m_cpp.json    # -> results_gist_cpp.jsonl
```

Each runs the sweep `NGM/IGTM/CGTM × {2,4,8} partitions`, `ES`/`TWO_MERGE × {2}`,
`INSERT × {1}`, at `M = 16`, `ef_construction = 200`, evaluated at `k = 10` over the
`ef` sweep `[10, 50, 100, 200, 400]`.

### NN-Descent baseline (Python / pynndescent)

`config/sift1m.json` ships with `base_limit = 100000` for a quick subset — **set it to
`1000000` (or remove it) to reproduce the full-scale baseline**.

```bash
python -m ngmbench.cli --config config/sift1m.json            # -> results_sift.jsonl
python -m ngmbench.cli --config config/gist1m.json            # -> results_gist.jsonl  (dim 960; RAM-heavy, ~tens of min)
```

### Figures

```bash
python scripts/make_figures.py \
  --results results_cpp.jsonl results_sift.jsonl results_gist_cpp.jsonl results_gist.jsonl \
  --out docs/figures
```

Writes, per dataset, `merge_cost`, `partition_scaling`, `recall_vs_qps`,
`construction_time`, `recall_vs_buildtime` (PNG + PDF) and `summary.csv` to
`docs/figures/{sift1m,gist1m}/`. Rows are grouped by their `dataset` field, so SIFT
and GIST never mix.

### Browse results 

```bash
cat results*.jsonl > results_all.jsonl                        # combine the logs (de-duped on run_key)
streamlit run app/playground.py -- --results results_all.jsonl
```

Use the sidebar **Dataset** filter to view one dataset at a time.

## Settings for reproducibility

- HNSW `M = 16`, `ef_construction = 200`; eval `k = 10`; `ef` sweep `[10,50,100,200,400]`.
- Query-set size `nq`: 10000 (SIFT), 1000 (GIST) — read per row from the log.
- C++ partitions are **contiguous id-ranges** (HNSWMerger's own scheme), not the Python side's random/k-means splits.

## Interpreting the logs

Key fields: `builder`, `algo`, `dataset`, `n_parts`, `dim`, `n`, `m`, `ef_construction`, `build_calc`, `merge_calc`, `total_calc`, `build_seconds`, `merge_seconds`, `recall@10`, `recall_curve[]`, `run_key`.

- **Compare merge algorithms by `merge_calc`, not `total_calc`** — the build cost is shared across algorithms at a given partition count and would swamp the signal.
- **`TWO_MERGE` reports `merge_calc = 0`** (it is not routed through the distance counter) — it is excluded from distance plots but kept for time and recall.
- **NN-Descent has `build_calc` / `merge_calc` / `total_calc` = `null`** by design:   `pynndescent` is Numba-compiled and exposes no count comparable to the merge   family, so its honest axis is recall vs wall-clock.
- On a `run_key` collision the **last** occurrence wins, so a re-run supersedes an
  earlier row without hand-pruning.

## Sources

Merge backend: [HNSWMerger](https://github.com/Kimchuls/HNSWMerger) (Kimchuls,
Apache-2.0). Merge algorithms: A. Ponomarenko, *Three Algorithms for Merging
Hierarchical Navigable Small World Graphs*, arXiv:2505.16064. NN-Descent via
`pynndescent`. Datasets: TEXMEX SIFT1M / GIST1M
