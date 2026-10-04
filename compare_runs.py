"""
Compare configs over seeds using test_predictions.json.

Usage: python compare_runs.py [models_dir] [--baseline polonly] [--boot 2000] [--lr 1e-5]

Reads <models_dir>/exp_<name>_lr<lr>_s<seed>/test_predictions.json, aligns by id
(only ids present in every run are used), and prints per config:
  * mean +/- std over seeds of accuracy and macro-F1
  * paired bootstrap of (config - baseline) over test samples: the same resampled
    test indices are used for both; metric is averaged over each config's seeds.
    Reports mean diff, 95% CI and two-sided p (share of resamples on the wrong side of 0).
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np

LABELS = ["positive", "negative", "neutral"]


def macro_f1_boot(W, gold, preds):
    """W: (B,N) resample counts; gold: (N,); preds: (N,S). Returns (B,S) macro-F1."""
    f1s = []
    for c in range(len(LABELS)):
        g = (gold == c)[:, None]
        p = preds == c
        tp = W @ (g & p).astype(float)
        fp = W @ p.astype(float) - tp
        fn = W @ np.repeat(g, preds.shape[1], axis=1).astype(float) - tp
        den = 2 * tp + fp + fn
        f1s.append(np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0))
    return np.mean(f1s, axis=0)


def acc_boot(W, gold, preds):
    return (W @ (preds == gold[:, None]).astype(float)) / W.sum(axis=1, keepdims=True)


def point_metrics(gold, pred):
    acc = float(np.mean(gold == pred))
    f1 = []
    for c in range(len(LABELS)):
        tp = np.sum((gold == c) & (pred == c))
        fp = np.sum((gold != c) & (pred == c))
        fn = np.sum((gold == c) & (pred != c))
        den = 2 * tp + fp + fn
        f1.append(2 * tp / den if den else 0.0)
    return acc, float(np.mean(f1))


def ms(v):
    return f"{st.mean(v):.4f} +/- {st.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models_dir", nargs="?", default="models")
    ap.add_argument("--baseline", default="polonly")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--lr", default=None, help="only runs with this lr tag")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = ap.parse_args()

    runs = defaultdict(dict)  # name -> seed -> {id: (gold, pred)}
    for f in sorted(glob.glob(os.path.join(args.models_dir, "exp_*", "test_predictions.json"))):
        m = re.match(r"exp_(.+)_lr([^_]+)_s(\d+)$", os.path.basename(os.path.dirname(f)))
        if not m or (args.lr and m.group(2) != args.lr):
            continue
        rows = json.load(open(f, encoding="utf-8"))
        runs[m.group(1)][int(m.group(3))] = {
            r["id"]: (LABELS.index(r["gold"]), LABELS.index(r["pred"])) for r in rows}
    if not runs:
        raise SystemExit(f"no test_predictions.json found under {args.models_dir}")

    common = None
    for seeds in runs.values():
        for d in seeds.values():
            common = set(d) if common is None else common & set(d)
    ids = sorted(common)
    if not ids:
        raise SystemExit("no test id is shared by all runs")
    first = next(iter(next(iter(runs.values())).values()))
    gold = np.array([first[i][0] for i in ids])

    preds, pm = {}, {}
    for name, seeds in runs.items():
        sl = sorted(seeds)
        for s in sl:  # gold must agree across runs
            assert all(seeds[s][i][0] == gold[k] for k, i in enumerate(ids)), f"gold mismatch {name} s{s}"
        preds[name] = np.array([[seeds[s][i][1] for s in sl] for i in ids])  # (N,S)
        pm[name] = [point_metrics(gold, preds[name][:, k]) for k in range(len(sl))]
    print(f"{len(ids)} test samples aligned by id; {sum(len(v) for v in runs.values())} runs\n")

    rng = np.random.default_rng(args.seed)
    N = len(ids)
    W = rng.multinomial(N, np.full(N, 1.0 / N), size=args.boot).astype(float)  # (B,N)
    base = args.baseline
    if base in preds:
        b_acc = acc_boot(W, gold, preds[base]).mean(axis=1)
        b_f1 = macro_f1_boot(W, gold, preds[base]).mean(axis=1)

    hdr = f"{'config':<14}{'n':<4}{'acc':<20}{'macro-F1':<20}"
    if base in preds:
        hdr += f"{'dAcc vs ' + base:<30}{'dF1 vs ' + base:<30}"
    print(hdr)
    order = ([base] if base in runs else []) + sorted(n for n in runs if n != base)
    for name in order:
        accs = [a for a, _ in pm[name]]
        f1s = [f for _, f in pm[name]]
        line = f"{name:<14}{len(accs):<4}{ms(accs):<20}{ms(f1s):<20}"
        if base in preds and name != base:
            cells = []
            for fn, bb in ((acc_boot, b_acc), (macro_f1_boot, b_f1)):
                d = fn(W, gold, preds[name]).mean(axis=1) - bb
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                cells.append(f"{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")
            line += "".join(f"{c:<30}" for c in cells)
        print(line)


if __name__ == "__main__":
    main()
