"""
D1 / D2 diagnostics (CPU only; reads test_predictions.json).

D1 (data dependence): ce_all_gv (100% implicit train rows) vs ce_all_gv_imp50 / _imp25 / _imp0.
D2 (out-of-domain prior): mams_only_ce (MAMS only, no SemEval training rows) vs ce_all_gv.

Buckets on the implicit test rows (matched to their sentence in restaurant14_test_all.json by
sentence text; other targets = rows of that sentence with a different target):
  NS = gold-neutral implicit rows whose sentence has another target with gold positive/negative
  PS = gold positive/negative implicit rows whose sentence has another target with gold pos/neg

Usage: python analyze_d1_d2.py   -> results/d1_d2_report.txt
"""
import argparse
import json
import os
import statistics as st

import numpy as np

LABELS = ["positive", "negative", "neutral"]
D1_CFGS = [("ce_all_gv", 100), ("ce_all_gv_imp50", 50), ("ce_all_gv_imp25", 25), ("ce_all_gv_imp0", 0)]
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def norm(s):
    return " ".join(str(s).lower().split())


def macro_f1(g, p):
    f = []
    for c in range(3):
        tp = np.sum((g == c) & (p == c)); fp = np.sum((g != c) & (p == c)); fn = np.sum((g == c) & (p != c))
        d = 2 * tp + fp + fn
        f.append(2 * tp / d if d else 0.0)
    return float(np.mean(f))


def ms(v, scale=1.0):
    v = [x * scale for x in v]
    if not v:
        return "n/a"
    return f"{st.mean(v):.4f} +/- {st.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--test-implicit", default="data/processed/test_implicit.json")
    ap.add_argument("--test-all", default="data/processed/restaurant14_test_all.json")
    ap.add_argument("--d1-seeds", default="6,7,8,9,10,11")
    ap.add_argument("--d2-seeds", default="6,7,8")
    ap.add_argument("--out", default="results/d1_d2_report.txt")
    args = ap.parse_args()

    impl_path = args.test_implicit
    if not os.path.exists(impl_path):
        impl_path = "data/processed/restaurant14_test_implicit.json"
    impl = {r["id"]: r for r in json.load(open(impl_path, encoding="utf-8"))
            if r["gold_polarity"].lower() in LABELS}
    by_sent = {}
    for r in json.load(open(args.test_all, encoding="utf-8")):
        by_sent.setdefault(norm(r["sentence"]), []).append(r)

    NS, PS, unmatched = set(), set(), 0
    for i, r in impl.items():
        rows = by_sent.get(norm(r["sentence"]))
        if not rows:
            unmatched += 1
            continue
        others = [o for o in rows if norm(o["target"]) != norm(r["target"])]
        has_pn = any(o["gold_polarity"].lower() in ("positive", "negative") for o in others)
        if not has_pn:
            continue
        if r["gold_polarity"].lower() == "neutral":
            NS.add(i)
        else:
            PS.add(i)
    ids = sorted(impl)
    gold = np.array([LABELS.index(impl[i]["gold_polarity"].lower()) for i in ids])
    ns_mask = np.array([i in NS for i in ids])
    ps_mask = np.array([i in PS for i in ids])
    log(f"D1/D2 DIAGNOSTICS | {len(ids)} implicit test rows ({impl_path}) | NS bucket {len(NS)} rows | "
        f"PS bucket {len(PS)} rows | unmatched sentences {unmatched}")
    log("NS = gold-neutral rows whose sentence has another pos/neg target; PS = gold pos/neg rows whose "
        "sentence has another pos/neg target")
    log("")

    def seeds_of(s):
        return [int(x) for x in s.split(",") if x.strip()]

    def load(cfg, seeds):
        per = {}
        for sd in seeds:
            f = os.path.join(args.models_dir, f"exp_{cfg}_lr{args.lr}_s{sd}", "test_predictions.json")
            if not os.path.exists(f):
                continue
            pred = {r["id"]: r["pred"] for r in json.load(open(f, encoding="utf-8"))}
            if not all(i in pred for i in ids):
                log(f"WARNING {cfg} s{sd}: predictions miss {sum(i not in pred for i in ids)} test ids - skipped")
                continue
            p = np.array([LABELS.index(pred[i]) for i in ids])
            m = {"acc": float(np.mean(p == gold)), "f1": macro_f1(gold, p),
                 "pred_neu": float(np.mean(p == 2)),
                 "ns_err": float(np.mean(p[ns_mask] != gold[ns_mask])) if ns_mask.any() else float("nan"),
                 "ps_rec": float(np.mean(p[ps_mask] == gold[ps_mask])) if ps_mask.any() else float("nan")}
            for c, l in enumerate(LABELS):
                sel = gold == c
                m[f"rec_{l}"] = float(np.mean(p[sel] == c)) if sel.any() else float("nan")
            per[sd] = m
        return per

    results = {}
    for cfg, _ in D1_CFGS:
        results[cfg] = load(cfg, seeds_of(args.d1_seeds))
    results["mams_only_ce"] = load("mams_only_ce", seeds_of(args.d2_seeds))

    keys = ["acc", "f1", "rec_positive", "rec_negative", "rec_neutral", "pred_neu", "ns_err", "ps_rec"]
    log(f"{'config':<18}{'seeds':<14}" + "".join(f"{k:<20}" for k in keys))
    for cfg, per in results.items():
        if not per:
            log(f"{cfg:<18}(no runs)")
            continue
        log(f"{cfg:<18}{','.join(map(str, sorted(per))):<14}"
            + "".join(f"{ms([per[s][k] for s in per]):<20}" for k in keys))
    log("")

    def mean(cfg, k):
        per = results.get(cfg) or {}
        return st.mean([per[s][k] for s in per]) if per else float("nan")

    base = results.get("ce_all_gv") or {}
    if base:
        log(f"SANITY ce_all_gv: macro-F1 {mean('ce_all_gv', 'f1'):.3f} (expected ~0.668) | NS error "
            f"{mean('ce_all_gv', 'ns_err'):.3f} (expected ~0.497) | PS recall {mean('ce_all_gv', 'ps_rec'):.3f} "
            f"(expected ~0.831) | bucket sizes NS {len(NS)} (expected 52), PS {len(PS)} (expected 86)")
    else:
        log("SANITY: no ce_all_gv runs found")
    log("")

    # ---------------- D1 ----------------
    log("=" * 80 + "\nD1: data dependence (implicit train rows 100% -> 50% -> 25% -> 0%)\n" + "=" * 80)
    F = {pct: mean(cfg, "f1") * 100 for cfg, pct in D1_CFGS}
    A = {pct: mean(cfg, "acc") * 100 for cfg, pct in D1_CFGS}
    N = {pct: mean(cfg, "pred_neu") * 100 for cfg, pct in D1_CFGS}
    for cfg, pct in D1_CFGS[1:]:
        log(f"  100% -> {pct:>3}%: macro-F1 drop {F[100] - F[pct]:+.2f} pts | accuracy drop {A[100] - A[pct]:+.2f} pts"
            f" | predicted-neutral rate {N[100]:.1f}% -> {N[pct]:.1f}% ({N[pct] - N[100]:+.1f} pts)")
    drop = F[100] - F[25]
    if np.isnan(drop):
        log("D1: n/a (missing ce_all_gv or ce_all_gv_imp25 runs)")
    else:
        if drop < 1.5:
            verdict = "D1: DATA-FLAT"
        elif drop >= 3.0 and F[100] >= F[50] - 0.5 >= F[25] - 1.0:
            verdict = "D1: DATA-STEEP"
        else:
            verdict = "D1: GREY"
        log(f"drop = F1(100) - F1(25) = {drop:.2f} pts | rule: <1.5 FLAT; >=3.0 and "
            f"F1(100) >= F1(50)-0.5 >= F1(25)-1.0 STEEP ({F[100]:.2f} >= {F[50] - 0.5:.2f} >= {F[25] - 1.0:.2f}); else GREY")
        log(verdict)
        # accuracy = sum_c share_c * recall_c -> neutral share of the accuracy drop
        acc_drop = A[100] - A[25]
        share_neu = float(np.mean(gold == 2))
        neu_part = share_neu * (mean("ce_all_gv", "rec_neutral") - mean("ce_all_gv_imp25", "rec_neutral")) * 100
        if acc_drop > 0:
            frac = neu_part / acc_drop
            log(f"neutral-recall part of the accuracy drop (100 -> 25): {neu_part:+.2f} of {acc_drop:+.2f} pts "
                f"({100 * frac:.0f}%)" + ("  -> drop mainly from neutral prior" if frac > 0.7 else ""))
        else:
            log(f"accuracy did not drop (100 -> 25: {acc_drop:+.2f} pts)")
    log("")

    # ---------------- D2 ----------------
    log("=" * 80 + "\nD2: MAMS-only training vs ce_all_gv (NS error and PS recall)\n" + "=" * 80)
    mams = results.get("mams_only_ce") or {}
    if not mams or not base:
        log("D2: n/a (missing mams_only_ce or ce_all_gv runs)")
    else:
        b_ns, b_ps = mean("ce_all_gv", "ns_err"), mean("ce_all_gv", "ps_rec")
        m_ns, m_ps = mean("mams_only_ce", "ns_err"), mean("mams_only_ce", "ps_rec")
        for name, per in (("ce_all_gv", base), ("mams_only_ce", mams)):
            log(f"  {name:<14} per seed: " + " | ".join(
                f"s{s}: NS err {per[s]['ns_err']:.3f}, PS rec {per[s]['ps_rec']:.3f}, F1 {per[s]['f1']:.3f}"
                for s in sorted(per)))
        ns_ok = m_ns <= b_ns - 0.10
        ps_ok = m_ps >= b_ps - 0.10
        log(f"NS error: mams {m_ns:.3f} vs baseline {b_ns:.3f} (need <= {b_ns - 0.10:.3f}) -> {'ok' if ns_ok else 'not met'}")
        log(f"PS recall: mams {m_ps:.3f} vs baseline {b_ps:.3f} (need >= {b_ps - 0.10:.3f}) -> {'ok' if ps_ok else 'not met'}")
        log("D2: " + ("PASS" if ns_ok and ps_ok else "PRIOR-ONLY" if ns_ok else "FAIL"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
