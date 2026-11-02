"""
Rate-controlled analysis of the confirmation runs (CPU only) -> results/rate_report.txt

Per run, on the implicit test rows (data/processed/test_implicit.json):
  a. raw accuracy, raw macro-F1, predicted-neutral rate
  b. AUROC of p_neu for gold-neutral vs gold-polar (rank based, ties averaged)
  c. pos/neg accuracy on gold-polar rows with argmax over (p_pos, p_neg) only
     (+ raw polar accuracy = 3-way argmax correct on gold-polar rows, used by H2)
  d. fixed-rate: s = log p_neu - log max(p_pos, p_neg); the top round(R * n) rows by s are
     neutral, the rest argmax(p_pos, p_neg). R = 0.409 = neutral share of the implicit TRAIN
     rows (never the test labels). Fixed-rate accuracy and macro-F1.
Per config: mean +/- sd over seeds. Paired bootstrap over test rows for the listed pairs
(each resample: every run's metric recomputed on the resampled rows, then averaged over seeds).

Inputs: models/exp_<cfg>_lr1e-4_s<seed>/test_predictions.json (with "probs") for the T5 configs,
        models/exp_<cfg>_lr2e-4_s<seed>/test_probs.json for the LoRA configs.
Usage:  python analyze_rate.py
"""
import argparse
import json
import os
import statistics as st

import numpy as np

LABELS = ["positive", "negative", "neutral"]
T5_CONFIGS = "polonly_all_gv,ce_all_gv,aux0.3_all_gv,aux0.3_all_gv_nullneu,mtisa_in_all_gv,aux0.3_all_gv_shuf,aux0.3_all_gv_shufsame"
LORA_CONFIGS = "lora_pol,lora_mtl,lora_mtl_shuf"
PAIRS = [("aux0.3_all_gv", "polonly_all_gv"), ("mtisa_in_all_gv", "polonly_all_gv"),
         ("aux0.3_all_gv_nullneu", "polonly_all_gv"), ("aux0.3_all_gv_nullneu", "aux0.3_all_gv"),
         ("ce_all_gv", "polonly_all_gv"), ("lora_mtl", "lora_pol"), ("lora_mtl", "lora_mtl_shuf"),
         ("aux0.3_all_gv", "aux0.3_all_gv_shuf"), ("aux0.3_all_gv_shuf", "polonly_all_gv"),
         ("aux0.3_all_gv", "aux0.3_all_gv_shufsame"), ("aux0.3_all_gv_shufsame", "aux0.3_all_gv_shuf"),
         ("aux0.3_all_gv_shufsame", "polonly_all_gv")]
PAIR_METRICS = ["fr_f1", "auroc", "polar_acc", "acc", "neu_rate"]
ALL_METRICS = ["acc", "f1", "neu_rate", "auroc", "polar_acc", "raw_polar_acc", "fr_acc", "fr_f1"]
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def macro_f1(g, p):
    f = []
    for c in range(3):
        tp = np.sum((g == c) & (p == c)); fp = np.sum((g != c) & (p == c)); fn = np.sum((g == c) & (p != c))
        d = 2 * tp + fp + fn
        f.append(2 * tp / d if d else 0.0)
    return float(np.mean(f))


def avg_ranks(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(score, positive):
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = avg_ranks(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fixed_rate_pred(P, rate):
    n = len(P)
    k = int(round(rate * n))
    Pc = np.clip(P, 1e-12, 1.0)
    s = np.log(Pc[:, 2]) - np.log(np.maximum(Pc[:, 0], Pc[:, 1]))
    pred = np.argmax(P[:, :2], axis=1)
    top = np.argsort(-s, kind="mergesort")[:k]
    pred[top] = 2
    assert int(np.sum(pred == 2)) == k, "fixed-rate step must predict exactly k neutrals"
    return pred, k


def metrics(P, g, rate):
    raw = P.argmax(1)
    polar = g != 2
    fr, _ = fixed_rate_pred(P, rate)
    return {"acc": float(np.mean(raw == g)), "f1": macro_f1(g, raw), "neu_rate": float(np.mean(raw == 2)),
            "auroc": auroc(P[:, 2], g == 2),
            "polar_acc": float(np.mean(np.argmax(P[polar, :2], 1) == g[polar])) if polar.any() else float("nan"),
            "raw_polar_acc": float(np.mean(raw[polar] == g[polar])) if polar.any() else float("nan"),
            "fr_acc": float(np.mean(fr == g)), "fr_f1": macro_f1(g, fr)}


def ms(v, pts=True):
    k = 100.0 if pts else 1.0
    v = [x * k for x in v if not np.isnan(x)]
    if not v:
        return "n/a"
    return f"{st.mean(v):.2f}+/-{st.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def welch(a, b):
    """Seed-level Welch t-test (scipy.stats.ttest_ind, equal_var=False) -> (mean diff, t, p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = float(a.mean() - b.mean())
    if len(a) < 2 or len(b) < 2:
        return diff, float("nan"), float("nan")
    try:
        from scipy.stats import ttest_ind
        r = ttest_ind(a, b, equal_var=False)
        return diff, float(r.statistic), float(r.pvalue)
    except ImportError:  # t only, p needs scipy
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return diff, float(diff / se) if se > 0 else float("nan"), float("nan")


def ci_p(d):
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return lo, hi, p


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--test-implicit", default="data/processed/test_implicit.json")
    ap.add_argument("--train-all", default="data/processed/restaurant14_train_all.json",
                    help="only used to print the implicit-train neutral share for reference")
    ap.add_argument("--t5-configs", default=T5_CONFIGS)
    ap.add_argument("--t5-seeds", default="12,13,14,15,16,17")
    ap.add_argument("--t5-lr", default="1e-4")
    ap.add_argument("--lora-configs", default=LORA_CONFIGS)
    ap.add_argument("--lora-seeds", default="6,7,8,9,10,11")
    ap.add_argument("--lora-lr", default="2e-4")
    ap.add_argument("--rate", type=float, default=0.409, help="neutral share of the implicit TRAIN rows")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/rate_report.txt")
    args = ap.parse_args()

    test_path = args.test_implicit if os.path.exists(args.test_implicit) else \
        "data/processed/restaurant14_test_implicit.json"
    test = [r for r in json.load(open(test_path, encoding="utf-8")) if r["gold_polarity"].lower() in LABELS]
    ids = [r["id"] for r in test]
    g = np.array([LABELS.index(r["gold_polarity"].lower()) for r in test])
    N = len(ids)
    k = int(round(args.rate * N))
    log(f"RATE-CONTROLLED ANALYSIS | {N} implicit test rows ({test_path}) | fixed rate R = {args.rate} "
        f"-> {k} neutral predictions per run")
    if os.path.exists(args.train_all):
        tr = [x for x in json.load(open(args.train_all, encoding="utf-8")) if x.get("is_implicit")]
        if tr:
            share = np.mean([x["gold_polarity"].lower() == "neutral" for x in tr])
            log(f"(reference: neutral share of the implicit rows in {args.train_all}: {share:.3f}; R is fixed "
                f"at {args.rate}, test labels are never used for R)")
    log("")

    # ---- load runs: cfg -> seed -> P (N,3) ----
    runs = {}
    specs = [(c, args.t5_seeds, args.t5_lr, "test_predictions.json") for c in args.t5_configs.split(",") if c.strip()]
    specs += [(c, args.lora_seeds, args.lora_lr, "test_probs.json") for c in args.lora_configs.split(",") if c.strip()]
    for cfg, seeds, lr, fname in specs:
        for sd in [int(s) for s in seeds.split(",") if s.strip()]:
            f = os.path.join(args.models_dir, f"exp_{cfg}_lr{lr}_s{sd}", fname)
            if not os.path.exists(f):
                continue
            rows = {r["id"]: r for r in json.load(open(f, encoding="utf-8"))}
            if any("probs" not in rows.get(i, {}) for i in ids):
                log(f"WARNING {f}: rows without 'probs' (run made before probabilities were saved?) - skipped")
                continue
            P = np.array([rows[i]["probs"] for i in ids], float)
            P = P / P.sum(1, keepdims=True)
            runs.setdefault(cfg, {})[sd] = P
    for cfg, _, _, _ in specs:
        if cfg not in runs:
            log(f"note: no usable runs for {cfg}")

    # ---- per config table ----
    point, per_seed = {}, {}
    log(f"{'config':<24}{'n':<4}" + "".join(f"{m:<16}" for m in ALL_METRICS) + "  (all in points; AUROC x100)")
    for cfg, per in runs.items():
        ms_ = {sd: metrics(P, g, args.rate) for sd, P in per.items()}
        point[cfg] = {m: float(np.nanmean([ms_[s][m] for s in ms_])) for m in ALL_METRICS}
        per_seed[cfg] = ms_
        log(f"{cfg:<24}{len(per):<4}" + "".join(f"{ms([ms_[s][m] for s in ms_]):<16}" for m in ALL_METRICS))
    log("")

    # ---- bootstrap: seed-averaged metrics per config per resample ----
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, N, size=(args.boot, N))
    need = sorted({c for p in PAIRS for c in p if c in runs})
    boot = {c: {m: np.empty(args.boot) for m in PAIR_METRICS} for c in need}
    for b in range(args.boot):
        ii = idx[b]
        gb = g[ii]
        for c in need:
            vals = [metrics(P[ii], gb, args.rate) for P in runs[c].values()]
            for m in PAIR_METRICS:
                boot[c][m][b] = np.nanmean([v[m] for v in vals])

    diffs = {}
    log(f"PAIRED BOOTSTRAP ({args.boot} resamples of test rows; A - B; points)")
    log(f"{'A vs B':<48}" + "".join(f"{m:<34}" for m in PAIR_METRICS))
    for a, b in PAIRS:
        if a not in runs or b not in runs:
            log(f"{a + ' vs ' + b:<48}n/a (missing runs)")
            continue
        cells = []
        for m in PAIR_METRICS:
            d = boot[a][m] - boot[b][m]
            lo, hi, p = ci_p(d)
            pt = point[a][m] - point[b][m]
            diffs[(a, b, m)] = (pt, lo, hi, p)
            cells.append(f"{100 * pt:+.2f} [{100 * lo:+.2f},{100 * hi:+.2f}] p={p:.3f}".ljust(34))
        log(f"{a + ' vs ' + b:<48}" + "".join(cells))
        wl = []
        for m in ("auroc", "fr_f1"):
            dm, t, pv = welch([v[m] for v in per_seed[a].values()], [v[m] for v in per_seed[b].values()])
            diffs[(a, b, m, "welch")] = (dm, t, pv)
            wl.append(f"{m} d {100 * dm:+.2f} t={t:.2f} p={pv:.3f}")
        log(f"{'':<48}seed-level Welch ({len(per_seed[a])} vs {len(per_seed[b])} seeds): " + " | ".join(wl))
    log("")

    # ---- verdicts ----
    def d(a, b, m):
        return diffs.get((a, b, m))

    log("VERDICTS")
    parts, ok_all, missing = [], True, False
    for a in ("aux0.3_all_gv", "mtisa_in_all_gv"):
        fr, au = d(a, "polonly_all_gv", "fr_f1"), d(a, "polonly_all_gv", "auroc")
        if fr is None or au is None:
            missing = True
            continue
        c1 = abs(100 * fr[0]) < 1.0
        c2 = au[1] <= 0 <= au[2]
        ok_all &= c1 and c2
        parts.append(f"{a} vs polonly: |d fixed-rate F1| = {abs(100 * fr[0]):.2f} pt ({'<' if c1 else '>='} 1.0), "
                     f"d AUROC {100 * au[0]:+.2f} CI [{100 * au[1]:+.2f},{100 * au[2]:+.2f}] "
                     f"({'includes' if c2 else 'excludes'} 0)")
    if missing:
        log("H1 (hint content is a rate shift): n/a (missing runs)")
    else:
        fails = [p for p in parts if ">=" in p or "excludes" in p]
        log("H1 (hint content is a rate shift): " + ("CONFIRMED" if ok_all else "REJECTED") + " | " + " | ".join(parts)
            + ("" if ok_all else " | fails: " + "; ".join(f.split(':')[0] for f in fails)))

    if "aux0.3_all_gv_nullneu" in point and "polonly_all_gv" in point:
        nn, po = point["aux0.3_all_gv_nullneu"], point["polonly_all_gv"]
        c1 = 100 * nn["neu_rate"] >= 100 * po["neu_rate"] - 1.0
        gain = 100 * (nn["raw_polar_acc"] - po["raw_polar_acc"])
        c2 = gain < 1.0
        log(f"H2 (nullneu restores the neutral rate): {'CONFIRMED' if c1 and c2 else 'REJECTED'} | predicted-neutral "
            f"rate nullneu {100 * nn['neu_rate']:.2f}% vs polonly {100 * po['neu_rate']:.2f}% (need >= "
            f"{100 * po['neu_rate'] - 1.0:.2f}) -> {'ok' if c1 else 'fails'} | raw polar accuracy gain "
            f"{gain:+.2f} pt (need < 1.0) -> {'ok' if c2 else 'fails'}")
    else:
        log("H2 (nullneu restores the neutral rate): n/a (missing runs)")

    fr, au = d("aux0.3_all_gv_nullneu", "polonly_all_gv", "fr_f1"), d("aux0.3_all_gv_nullneu", "polonly_all_gv", "auroc")
    if fr and au:
        lead = 100 * fr[0] >= 1.5 and fr[3] < 0.10 and au[0] > 0
        log(f"LEAD (method signal): {'YES' if lead else 'NO'} | nullneu vs polonly d fixed-rate F1 {100 * fr[0]:+.2f} pt "
            f"(p={fr[3]:.3f}; need >= +1.50 and p < 0.10) | d AUROC {100 * au[0]:+.2f} (need > 0)")
    else:
        log("LEAD (method signal): n/a (missing runs)")

    fr = d("ce_all_gv", "polonly_all_gv", "fr_f1")
    if fr:
        log(f"H3 (classification loss is real): {'CONFIRMED' if fr[0] > 0 and fr[3] < 0.10 else 'REJECTED'} | "
            f"ce vs polonly d fixed-rate F1 {100 * fr[0]:+.2f} pt (p={fr[3]:.3f}; need > 0 and p < 0.10)")
    else:
        log("H3 (classification loss is real): n/a (missing runs)")

    h4 = []
    for b in ("lora_pol", "lora_mtl_shuf"):
        fr, au = d("lora_mtl", b, "fr_f1"), d("lora_mtl", b, "auroc")
        if fr and au:
            h4.append(f"lora_mtl vs {b}: d fixed-rate F1 {100 * fr[0]:+.2f} pt [{100 * fr[1]:+.2f},{100 * fr[2]:+.2f}] "
                      f"p={fr[3]:.3f}, d AUROC {100 * au[0]:+.2f} [{100 * au[1]:+.2f},{100 * au[2]:+.2f}] p={au[3]:.3f}")
        else:
            h4.append(f"lora_mtl vs {b}: n/a (missing runs)")
    log("H4 (Qwen hints at fixed rate): " + " | ".join(h4))

    rs, sp = d("aux0.3_all_gv", "aux0.3_all_gv_shuf", "auroc"), d("aux0.3_all_gv_shuf", "polonly_all_gv", "auroc")
    if rs and sp:
        dd = 100 * rs[0]
        if dd >= 1.5 and rs[3] < 0.05:
            h5 = "CONTENT MATTERS"
        elif abs(dd) < 1.0 and not (sp[1] <= 0 <= sp[2]):
            h5 = "CONTENT IRRELEVANT"
        else:
            h5 = "UNCLEAR"
        log(f"H5 (hint content): {h5} | real - shuf d AUROC {dd:+.2f} [{100 * rs[1]:+.2f},{100 * rs[2]:+.2f}] "
            f"p={rs[3]:.3f} (MATTERS: >= +1.5 and p < 0.05; IRRELEVANT: |d| < 1.0 and shuf vs polonly CI "
            f"excludes 0) | shuf vs polonly d AUROC {100 * sp[0]:+.2f} [{100 * sp[1]:+.2f},{100 * sp[2]:+.2f}] "
            f"p={sp[3]:.3f}")
    else:
        log("H5 (hint content): n/a (missing aux0.3_all_gv / aux0.3_all_gv_shuf / polonly_all_gv runs)")
    w5 = diffs.get(("aux0.3_all_gv", "aux0.3_all_gv_shuf", "auroc", "welch"))
    if w5:
        log(f"H5 seed-level: Welch aux0.3_all_gv vs aux0.3_all_gv_shuf AUROC d {100 * w5[0]:+.2f} "
            f"t={w5[1]:.2f} p={w5[2]:.3f}")

    r_ss = d("aux0.3_all_gv", "aux0.3_all_gv_shufsame", "auroc")
    ss_s = d("aux0.3_all_gv_shufsame", "aux0.3_all_gv_shuf", "auroc")
    r_s = d("aux0.3_all_gv", "aux0.3_all_gv_shuf", "auroc")
    if r_ss and ss_s:
        if 100 * r_ss[0] >= 1.5 and r_ss[3] < 0.05 and abs(100 * ss_s[0]) < 1.0:
            h6 = "SENTENCE CONTENT"
        elif 100 * ss_s[0] >= 1.5 and ss_s[3] < 0.05 and abs(100 * r_ss[0]) < 1.0:
            h6 = "LABEL SIGNAL"
        else:
            h6 = "MIXED"
        fmt = (lambda name, x: f"{name} d AUROC {100 * x[0]:+.2f} [{100 * x[1]:+.2f},{100 * x[2]:+.2f}] p={x[3]:.3f}"
               if x else f"{name}: n/a")
        log(f"H6 (what in the hints matters): {h6} | " + " | ".join([
            fmt("real - shufsame", r_ss), fmt("shufsame - shuf", ss_s), fmt("real - shuf", r_s)])
            + " (SENTENCE CONTENT: real-shufsame >= +1.5, p < 0.05, |shufsame-shuf| < 1.0; LABEL SIGNAL: "
              "shufsame-shuf >= +1.5, p < 0.05, |real-shufsame| < 1.0)")
    else:
        log("H6 (what in the hints matters): n/a (missing aux0.3_all_gv / _shufsame / _shuf runs)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
