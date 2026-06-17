#!/usr/bin/env python3
"""
Measure graph structure of merged / built HNSW graphs:
  - out-degree distribution at level 0 (mean / median / max / fraction at the M0 cap)
  - connected components (union-find over level-0 edges, treated as undirected)

INPUT: one adjacency dump per graph, produced by scripts/dump_graph_level0.cpp
(dropped into the HNSWMerger backend). Format, one node per line:
    # cap=<maxM0>            (optional comment; default 32 for M=16)
    <node_id> <deg> <nbr1> <nbr2> ...
Node ids are the backend's internal ids (0..n-1). Lines starting with '#' are comments.

USAGE:
    python scripts/graph_structure.py \
        --dumps IGTM-2:igtm.txt TWO_MERGE-2:twomerge.txt INSERT:insert.txt \
        --out docs/figures/sift1m --dataset sift1m --cap 32

Produces  <out>/degree_distribution.png  and  <out>/graph_structure.csv .
"""
import argparse, csv, os, sys
from collections import defaultdict


def load_dump(path):
    """Return (adjacency dict id->list[int], cap_or_None)."""
    adj = {}
    cap = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "cap=" in line:
                    try:
                        cap = int(line.split("cap=")[1].split()[0])
                    except ValueError:
                        pass
                continue
            parts = line.split()
            nid = int(parts[0])
            # tolerate both "<id> <deg> <nbrs...>" and "<id> <nbrs...>"
            rest = parts[1:]
            if rest and len(rest) >= 1:
                # detect whether parts[1] is a degree count matching remaining length
                maybe_deg = int(rest[0])
                if maybe_deg == len(rest) - 1:
                    nbrs = [int(x) for x in rest[1:]]
                else:
                    nbrs = [int(x) for x in rest]
            else:
                nbrs = []
            adj[nid] = nbrs
    return adj, cap


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def analyze(adj, cap):
    n = (max(adj) + 1) if adj else 0
    degs = [len(adj.get(i, [])) for i in range(n)]
    mean = sum(degs) / n if n else 0.0
    sdegs = sorted(degs)
    median = sdegs[n // 2] if n else 0
    mx = max(degs) if degs else 0
    cap = cap or (mx if mx else 1)
    frac_at_cap = sum(1 for d in degs if d >= cap) / n if n else 0.0
    # undirected connectivity over level-0 edges
    uf = UnionFind(n)
    for i in range(n):
        for j in adj.get(i, []):
            if 0 <= j < n:
                uf.union(i, j)
    comp = defaultdict(int)
    for i in range(n):
        comp[uf.find(i)] += 1
    sizes = sorted(comp.values(), reverse=True)
    return {
        "n": n,
        "mean_degree": round(mean, 3),
        "median_degree": median,
        "max_degree": mx,
        "cap": cap,
        "frac_at_cap": round(frac_at_cap, 4),
        "n_components": len(sizes),
        "largest_cc_frac": round(sizes[0] / n, 4) if n else 0.0,
        "singletons": sum(1 for s in sizes if s == 1),
        "_degs": degs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="label:path entries (label optional; defaults to filename)")
    ap.add_argument("--out", default="docs/figures/sift1m")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--cap", type=int, default=None,
                    help="level-0 max degree maxM0 (default 32 for M=16, else inferred)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for item in args.dumps:
        if ":" in item and not item[1:3] == ":\\":
            label, path = item.split(":", 1)
        else:
            label, path = os.path.splitext(os.path.basename(item))[0], item
        if not os.path.exists(path):
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        adj, cap = load_dump(path)
        results[label] = analyze(adj, args.cap or cap or 32)

    if not results:
        print("no dumps loaded; nothing to do", file=sys.stderr)
        sys.exit(1)

    # summary csv
    csv_path = os.path.join(args.out, "graph_structure.csv")
    cols = ["method", "n", "mean_degree", "median_degree", "max_degree", "cap",
            "frac_at_cap", "n_components", "largest_cc_frac", "singletons"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for label, r in results.items():
            w.writerow([label] + [r[c] for c in cols[1:]])
    print("wrote", csv_path)
    for label, r in results.items():
        print(f"  {label:14s} mean_deg={r['mean_degree']:.2f} max={r['max_degree']} "
              f"frac@cap={r['frac_at_cap']:.3f} components={r['n_components']} "
              f"largest_cc={r['largest_cc_frac']:.3f}")

    # figure: degree histograms (overlaid)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        cap = max(r["cap"] for r in results.values())
        bins = range(0, cap + 2)
        for label, r in results.items():
            ax.hist(r["_degs"], bins=bins, histtype="step", linewidth=1.8,
                    density=True, label=f"{label} (mean {r['mean_degree']:.1f})")
        ax.set_xlabel("out-degree at level 0")
        ax.set_ylabel("fraction of nodes")
        title = "Level-0 out-degree distribution"
        if args.dataset:
            title += f" ({args.dataset.upper()})"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        png = os.path.join(args.out, "degree_distribution.png")
        fig.savefig(png, dpi=150)
        fig.savefig(png.replace(".png", ".pdf"))
        print("wrote", png)
    except Exception as e:  # noqa
        print("plot skipped:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
