"""
Pooled swapped-hint test (CPU only; reads test_predictions.json).

For each (real, swapped) pair of configs that share seeds, split, lr and test set:
  per-test-sample accuracy / macro-F1 averaged over the common seeds of the pair,
  diff = swapped - real, paired bootstrap over test samples (same resampled items for both)
  -> mean diff, 95% CI, two-sided p.
Pooled (per dataset and over all datasets): mean of the per-pair diffs; the bootstrap resamples
test samples jointly across all pairs of a dataset (one shared resample per dataset), so one CI / p.

Pairs reuse configs (e.g. lfrat_all_gv in two pairs), so the pooled mean is a summary of
correlated comparisons, not of independent experiments.

Usage: python pooled_swap_test.py [--boot 5000] [--out results/pooled_swap_test.txt]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from compare_runs import LABELS, acc_boot, macro_f1_boot, point_metrics

PAIRS = [  # (dataset, models dir, real, swapped, lr tag)
    ("restaurant", "models", "aux0.3", "aux0.3_shufhints", "1e-4"),
    ("restaurant", "models", "lf0.3", "lf0.3_shuf", "1e-4"),
    ("restaurant", "models", "lf1.0_gv", "lf1.0_shuf_gv", "1e-4"),
    ("restaurant", "models", "lfrat_all_gv", "lfrat_shuf_all_gv", "1e-4"),
    ("restaurant", "models", "lfrat_all_gv", "lfrat_shufsame_all_gv", "1e-4"),
    ("restaurant", "models", "lora_mtl", "lora_mtl_shuf", "2e-4"),
    ("laptop", "models_laptop", "lfrat_all_gv", "lfrat_shuf_all_gv", "1e-4"),
    ("laptop", "models_laptop", "lfrat_all_gv", "lfrat_shufsame_all_gv", "1e-4"),
]
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def load_runs(models_dir):
    """(name, lr) -> seed -> {id: (gold, pred)}"""
    runs = defaultdict(dict)
    for f in glob.glob(os.path.join(models_dir, "exp_*", "test_predictions.json")):
        m = re.match(r"exp_(.+)_lr([^_]+)_s(\d+)$", os.path.basename(os.path.dirname(f)))
        if not m:
            continue
        rows = json.load(open(f, encoding="utf-8"))
        runs[(m.group(1), m.group(2))][int(m.group(3))] = {
            r["id"]: (LABELS.index(r["gold"]), LABELS.index(r["pred"])) for r in rows}
    return runs


def ci_p(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--out", default="results/pooled_swap_test.txt")
    args = ap.parse_args()

    cache = {}
    usable = []  # dicts per pair
    log(f"POOLED SWAPPED-HINT TEST | diff = swapped - real | {args.boot} bootstrap resamples over test samples")
    log("")
    for ds, mdir, real, swap, lr in PAIRS:
        if mdir not in cache:
            cache[mdir] = load_runs(mdir) if os.path.isdir(mdir) else {}
        runs = cache[mdir]
        a, b = runs.get((real, lr), {}), runs.get((swap, lr), {})
        tag = f"{ds}: {real} vs {swap} (lr {lr})"
        if not a or not b:
            log(f"[skip] {tag}: missing runs ({real}: {len(a)} seeds, {swap}: {len(b)} seeds)")
            continue
        seeds = sorted(set(a) & set(b))
        if not seeds:
            log(f"[skip] {tag}: no common seeds ({sorted(a)} vs {sorted(b)})")
            continue
        ids = set.intersection(*[set(a[s]) for s in seeds], *[set(b[s]) for s in seeds])
        usable.append({"ds": ds, "real": real, "swap": swap, "lr": lr, "seeds": seeds,
                       "runs": (a, b), "ids": ids, "tag": tag})

    if not usable:
        log("no usable pair found")
    else:
        rng = np.random.default_rng(args.seed)
        W_by_ds = {}
        for ds in sorted({p["ds"] for p in usable}):
            pairs = [p for p in usable if p["ds"] == ds]
            common = sorted(set.intersection(*[p["ids"] for p in pairs]))
            for p in pairs:
                if len(p["ids"]) != len(common):
                    log(f"note: {p['tag']} has {len(p['ids'])} test ids; pooled analysis uses the "
                        f"{len(common)} ids shared by all {ds} pairs")
                p["order"] = common
            N = len(common)
            W_by_ds[ds] = rng.multinomial(N, np.full(N, 1.0 / N), size=args.boot).astype(float)

        log(f"{'pair':<62}{'seeds':<16}{'acc real':<10}{'acc swap':<10}{'d acc [95% CI] p':<34}"
            f"{'F1 real':<10}{'F1 swap':<10}{'d F1 [95% CI] p'}")
        for p in usable:
            a, b = p["runs"]
            ids = p["order"]
            gold = np.array([a[p["seeds"][0]][i][0] for i in ids])
            Pr = np.array([[a[s][i][1] for s in p["seeds"]] for i in ids])
            Ps = np.array([[b[s][i][1] for s in p["seeds"]] for i in ids])
            pt = {k: [point_metrics(gold, P[:, j]) for j in range(P.shape[1])] for k, P in (("r", Pr), ("s", Ps))}
            p["acc"] = (np.mean([x[0] for x in pt["r"]]), np.mean([x[0] for x in pt["s"]]))
            p["f1"] = (np.mean([x[1] for x in pt["r"]]), np.mean([x[1] for x in pt["s"]]))
            W = W_by_ds[p["ds"]]
            p["d_acc"] = acc_boot(W, gold, Ps).mean(1) - acc_boot(W, gold, Pr).mean(1)
            p["d_f1"] = macro_f1_boot(W, gold, Ps).mean(1) - macro_f1_boot(W, gold, Pr).mean(1)
            cells = []
            for key, d in (("acc", p["d_acc"]), ("f1", p["d_f1"])):
                lo, hi, pv = ci_p(d)
                r_, s_ = p[key]
                cells.append(f"{r_:<10.4f}{s_:<10.4f}" + f"{s_ - r_:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}".ljust(34))
            log(f"{p['tag']:<62}{','.join(map(str, p['seeds'])):<16}" + "".join(cells))
        log("(point values: metric per seed, averaged over the pair's common seeds; CI/p from the "
            "paired bootstrap over test samples)")
        log("")

        log("POOLED (mean of per-pair diffs; joint bootstrap over the shared test items of each dataset)")
        groups = [(ds, [p for p in usable if p["ds"] == ds]) for ds in sorted(W_by_ds)]
        if len(groups) > 1:
            groups.append(("all", usable))
        for name, pairs in groups:
            parts = []
            for key in ("acc", "f1"):
                point = np.mean([p[key][1] - p[key][0] for p in pairs])
                d = np.mean([p["d_" + key] for p in pairs], axis=0)  # same resample index b per dataset
                lo, hi, pv = ci_p(d)
                ge = sum(1 for p in pairs if p[key][1] - p[key][0] >= 0)
                parts.append(f"d {key} {point:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f} | swapped >= real "
                             f"in {ge}/{len(pairs)} pairs")
            log(f"  {name:<11} ({len(pairs)} pairs): " + "  ||  ".join(parts))
        log("note: pairs share configs (e.g. lfrat_all_gv), so the pooled diffs are correlated "
            "comparisons; 'all' resamples each dataset's items independently with a shared index.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
