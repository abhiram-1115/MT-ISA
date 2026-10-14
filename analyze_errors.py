"""
Error analysis of MT-ISA test predictions (CPU only, reads files only).

Usage: python analyze_errors.py [--models-dir models] [--lr 1e-4]
Writes results/error_analysis.txt (and prints it).

1. aggregate confusion matrix + per-class recall (polonly runs; all runs of the given lr)
2. per-test-sample error rate over all runs of the given lr, always-wrong / always-right buckets
3. majority-vote seed ensemble vs single-run mean, per config
4. do validation rows share sentences with train? (replays the split of train_mt_isa.py)
5. does the aux json store the generator's first-iteration answer?
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LABELS = ["positive", "negative", "neutral"]  # == POLARITY_WORDS in mt_isa_model.py

try:  # use the real split code so the replay is exact
    from train_mt_isa import POLARITY_TO_ID, make_validation_split
except Exception as e:  # pragma: no cover - fallback: verbatim copy of the split
    import random
    POLARITY_TO_ID = {w: i for i, w in enumerate(LABELS)}

    def make_validation_split(instances, fraction, seed):
        rng = random.Random(seed)
        by_label = {label: [] for label in POLARITY_TO_ID}
        for x in instances:
            by_label[x["gold_polarity"].lower()].append(x)
        train, val = [], []
        for label in POLARITY_TO_ID:
            rows = by_label[label]
            rng.shuffle(rows)
            n_val = max(1, int(round(len(rows) * fraction))) if rows else 0
            val.extend(rows[:n_val])
            train.extend(rows[n_val:])
        rng.shuffle(train)
        rng.shuffle(val)
        return train, val

OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def keep(x):  # same filter as train_mt_isa.py
    return x.get("is_implicit", True) and x["gold_polarity"].lower() in POLARITY_TO_ID


def norm(s):
    return " ".join(str(s).split()).lower()


def prefix(i):
    return i.rsplit("_aspect_", 1)[0] if "_aspect_" in i else i


def macro_f1(g, p):
    f = []
    for c in range(3):
        tp = np.sum((g == c) & (p == c)); fp = np.sum((g != c) & (p == c)); fn = np.sum((g == c) & (p != c))
        d = 2 * tp + fp + fn
        f.append(2 * tp / d if d else 0.0)
    return float(np.mean(f))


def confusion(golds, preds):
    m = np.zeros((3, 3), dtype=int)
    for g, p in zip(golds, preds):
        m[g, p] += 1
    return m


def print_confusion(title, m, n_runs):
    log(f"{title}  ({n_runs} runs, {m.sum()} predictions)")
    log(f"{'gold/pred':<14}" + "".join(f"{l:<10}" for l in LABELS) + "recall")
    for i, l in enumerate(LABELS):
        r = m[i, i] / m[i].sum() if m[i].sum() else float("nan")
        log(f"{l:<14}" + "".join(f"{m[i, j]:<10}" for j in range(3)) + f"{r:.3f}")
    log(f"overall acc {np.trace(m) / max(1, m.sum()):.4f} | "
        f"macro recall {np.nanmean([m[i, i] / m[i].sum() if m[i].sum() else np.nan for i in range(3)]):.4f}")
    log()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--lr", default="1e-4", help="lr tag of the runs to analyse")
    ap.add_argument("--baseline", default="polonly")
    ap.add_argument("--train-data", default="data/processed/restaurant14_train_implicit.json")
    ap.add_argument("--aux-data", default="data/auxiliary/restaurant14_train_implicit_aux_qwen7b_full.json")
    ap.add_argument("--test-data", default="data/processed/restaurant14_test_implicit.json")
    ap.add_argument("--test-all-data", default="data/processed/restaurant14_test_all.json",
                    help="test file with ALL rows (explicit + implicit), used in section 6")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--out", default="results/error_analysis.txt")
    ap.add_argument("--n-show", type=int, default=20)
    args = ap.parse_args()

    # ---- load runs: name -> seed -> {id: (gold, pred)}
    runs = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(args.models_dir, "exp_*", "test_predictions.json"))):
        m = re.match(r"exp_(.+)_lr([^_]+)_s(\d+)$", os.path.basename(os.path.dirname(f)))
        if not m or m.group(2) != args.lr:
            continue
        rows = json.load(open(f, encoding="utf-8"))
        runs[m.group(1)][int(m.group(3))] = {r["id"]: (LABELS.index(r["gold"]), LABELS.index(r["pred"])) for r in rows}
    if not runs:
        raise SystemExit(f"no test_predictions.json for lr {args.lr} under {args.models_dir}")
    n_runs = sum(len(v) for v in runs.values())
    test = {x["id"]: x for x in json.load(open(args.test_data, encoding="utf-8"))} if os.path.exists(args.test_data) else {}
    log(f"ERROR ANALYSIS | lr={args.lr} | {n_runs} runs, {len(runs)} configs: "
        + ", ".join(f"{k}({len(v)})" for k, v in sorted(runs.items())))
    log(f"test rows with data file: {len(test)}")
    log()

    # ================= 1. confusion =================
    log("=" * 78 + "\n1. CONFUSION MATRICES\n" + "=" * 78)
    def agg(names):
        g, p, n = [], [], 0
        for name in names:
            for d in runs.get(name, {}).values():
                n += 1
                for gg, pp in d.values():
                    g.append(gg); p.append(pp)
        return confusion(g, p), n
    if args.baseline in runs:
        m, n = agg([args.baseline])
        print_confusion(f"{args.baseline} only", m, n)
    m, n = agg(list(runs))
    print_confusion("ALL runs", m, n)

    # ================= 2. per-sample error rate =================
    log("=" * 78 + "\n2. PER-SAMPLE ERROR RATE (all runs)\n" + "=" * 78)
    gold, preds = {}, defaultdict(list)
    for name, seeds in runs.items():
        for d in seeds.values():
            for i, (g, p) in d.items():
                gold[i] = g; preds[i].append(p)
    err = {i: float(np.mean([p != gold[i] for p in ps])) for i, ps in preds.items()}
    wrong = [i for i, e in err.items() if e >= 0.9]
    right = [i for i, e in err.items() if e <= 0.1]
    mid = [i for i in err if i not in set(wrong) | set(right)]
    N = len(err)
    log(f"samples: {N} | wrong in >=90% of runs: {len(wrong)} ({100*len(wrong)/N:.1f}%) | "
        f"right in >=90%: {len(right)} ({100*len(right)/N:.1f}%) | in between: {len(mid)} ({100*len(mid)/N:.1f}%)")
    log("always-wrong by gold label: " + str({LABELS[g]: c for g, c in sorted(Counter(gold[i] for i in wrong).items())}))
    log("always-wrong (gold -> most common pred): " + str(dict(Counter(
        f"{LABELS[gold[i]]}->{LABELS[Counter(preds[i]).most_common(1)[0][0]]}" for i in wrong).most_common())))
    log(f"mean error rate per sample: {np.mean(list(err.values())):.4f}")
    log()
    log(f"{min(args.n_show, len(wrong))} always-wrong samples (highest error rate first):")
    for i in sorted(wrong, key=lambda i: (-err[i], i))[:args.n_show]:
        t = test.get(i, {})
        top, cnt = Counter(preds[i]).most_common(1)[0]
        log(f"- {i} | err {err[i]:.2f} | gold {LABELS[gold[i]]} | most common pred {LABELS[top]} ({cnt}/{len(preds[i])})")
        log(f"    target: {t.get('target', '?')}")
        log(f"    sentence: {t.get('sentence', '?')}")
    log()

    # ================= 3. majority-vote ensemble =================
    log("=" * 78 + "\n3. MAJORITY-VOTE SEED ENSEMBLE vs SINGLE-RUN MEAN\n" + "=" * 78)
    log(f"{'config':<20}{'n':<4}{'single acc':<20}{'ens acc':<10}{'single F1':<20}{'ens F1':<10}{'ties':<6}")
    for name in ([args.baseline] if args.baseline in runs else []) + sorted(n for n in runs if n != args.baseline):
        seeds = sorted(runs[name])
        ids = sorted(set.intersection(*[set(runs[name][s]) for s in seeds]))
        g = np.array([runs[name][seeds[0]][i][0] for i in ids])
        P = np.array([[runs[name][s][i][1] for s in seeds] for i in ids])  # (N, S)
        accs = [float(np.mean(P[:, k] == g)) for k in range(len(seeds))]
        f1s = [macro_f1(g, P[:, k]) for k in range(len(seeds))]
        ens, ties = [], 0
        for row in P:
            c = Counter(row.tolist()); top = max(c.values())
            tied = {l for l, v in c.items() if v == top}
            ties += len(tied) > 1
            ens.append(next(p for p in row if p in tied))  # tie -> lowest seed's vote among tied
        ens = np.array(ens)
        sa = f"{st.mean(accs):.4f}+/-{st.stdev(accs):.4f}" if len(accs) > 1 else f"{accs[0]:.4f}"
        sf = f"{st.mean(f1s):.4f}+/-{st.stdev(f1s):.4f}" if len(f1s) > 1 else f"{f1s[0]:.4f}"
        log(f"{name:<20}{len(seeds):<4}{sa:<20}{np.mean(ens == g):<10.4f}{sf:<20}{macro_f1(g, ens):<10.4f}{ties:<6}")
    log("(ties: samples where the vote was tied; broken by the lowest seed's prediction among tied labels)")
    log()

    # ================= 4. val/train sentence overlap =================
    log("=" * 78 + "\n4. DOES VALIDATION SHARE SENTENCES WITH TRAIN?\n" + "=" * 78)
    train_all = json.load(open(args.train_data, encoding="utf-8"))
    aux_rows = json.load(open(args.aux_data, encoding="utf-8"))
    aux_ids = {a["instance_id"] for a in aux_rows}
    with_aux = [x for x in train_all if keep(x) and x["id"] in aux_ids]
    log("Split code (train_mt_isa.py): make_validation_split(with_aux, 0.15, seed) shuffles INSTANCES "
        "stratified by polarity;\nit never groups by sentence. Run uses no --val-data, so val comes from this split.")
    groups = defaultdict(list)
    for x in with_aux:
        groups[prefix(x["id"])].append(x)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    same_sent = sum(1 for v in multi.values() if len({norm(x["sentence"]) for x in v}) == 1)
    log(f"id format: '<sentence id>_aspect_<k>' (e.g. '296_aspect_3'): prefix = source sentence, k = k-th target in it.")
    log(f"check on train pool ({len(with_aux)} rows): {len(groups)} distinct prefixes, {len(multi)} with >1 row; "
        f"{same_sent}/{len(multi)} multi-row prefixes have one identical sentence.")
    sent_counts = Counter(norm(x["sentence"]) for x in with_aux)
    log(f"rows whose sentence occurs in >1 row of the pool: "
        f"{sum(c for c in sent_counts.values() if c > 1)}/{len(with_aux)}")
    seeds_used = sorted({s for v in runs.values() for s in v})
    fr_sent, fr_pref = [], []
    for sd in seeds_used:
        tr, va = make_validation_split(with_aux, args.val_fraction, sd)
        trs = {norm(x["sentence"]) for x in tr}; trp = {prefix(x["id"]) for x in tr}
        a = sum(norm(x["sentence"]) in trs for x in va) / max(1, len(va))
        b = sum(prefix(x["id"]) in trp for x in va) / max(1, len(va))
        fr_sent.append(a); fr_pref.append(b)
        log(f"seed {sd:<6} train {len(tr)} val {len(va)} | val rows with sentence also in train: "
            f"{100*a:.1f}% | with sentence-id prefix also in train: {100*b:.1f}%")
    if fr_sent:
        log(f"mean over {len(fr_sent)} seeds: sentence {100*np.mean(fr_sent):.1f}% "
            f"(min {100*min(fr_sent):.1f}, max {100*max(fr_sent):.1f})")
    log("(train side measured before the <unk>-label filter, which drops very few rows)")
    if test:
        trs = {norm(x["sentence"]) for x in with_aux}
        ov = sum(norm(t["sentence"]) in trs for t in test.values()) / len(test)
        log(f"for reference: test rows whose sentence appears in the train pool: {100*ov:.1f}%")
    log()

    # ================= 5. generator first-answer accuracy =================
    log("=" * 78 + "\n5. GENERATOR FIRST-ANSWER ACCURACY (no-hint)\n" + "=" * 78)
    keys = Counter(k for a in aux_rows for k in a)
    log("aux record fields: " + ", ".join(f"{k}({c})" for k, c in sorted(keys.items())))
    cand = [k for k in keys if re.search(r"first|initial|pred|iter0|polarity", k, re.I)]
    log(f"fields that could hold a first-iteration polarity: {cand}")
    gold_by_id = {x["id"]: x["gold_polarity"].lower() for x in train_all}
    cmp_ = [(str(a.get("gold_polarity", "")).lower(), gold_by_id[a["instance_id"]])
            for a in aux_rows if a["instance_id"] in gold_by_id and "gold_polarity" in a]
    if cmp_:
        agree = sum(x == y for x, y in cmp_)
        log(f"aux gold_polarity == train gold_polarity: {agree}/{len(cmp_)} ({100*agree/len(cmp_):.1f}%)"
            " -> it is the gold label, not a generator prediction" if agree == len(cmp_)
            else f"aux gold_polarity == train gold_polarity: {agree}/{len(cmp_)} - DIFFERS, inspect")
    it = Counter(a.get("refinement_iterations") for a in aux_rows)
    conv = sum(bool(a.get("converged")) for a in aux_rows)
    log(f"refinement_iterations: {dict(sorted(it.items(), key=lambda kv: str(kv[0])))} | "
        f"converged {conv}/{len(aux_rows)} ({100*conv/len(aux_rows):.1f}%)")
    if not cand or cand == ["gold_polarity"]:
        log("RESULT: the generator's first-iteration polarity is NOT stored in the aux json, so its no-hint "
            "accuracy cannot be computed.\nOnly refinement_iterations / converged are stored "
            "(converged at iteration 1 may mean the first answer was accepted, but that is not stored as such).")
    log()

    # ================= 6. multi-target sentences: always-wrong vs always-right =================
    log("=" * 78 + "\n6. OTHER TARGETS IN THE SAME SENTENCE (always-wrong vs always-right)\n" + "=" * 78)
    if os.path.exists(args.test_all_data):
        all_rows = json.load(open(args.test_all_data, encoding="utf-8"))
        log(f"test file with all rows: {args.test_all_data} ({len(all_rows)} rows, "
            f"{sum(bool(r.get('is_implicit')) for r in all_rows)} implicit)")
    else:
        all_rows = list(test.values())
        log(f"NOTE: {args.test_all_data} not found -> using the IMPLICIT test file ({len(all_rows)} rows); "
            "'other targets' are then only the other implicit targets.")
    by_key, by_sent = defaultdict(list), defaultdict(list)
    for r in all_rows:
        k = r.get("sentence_id") or prefix(r.get("id", ""))
        by_key[k].append(r)
        by_sent[norm(r["sentence"])].append(r)

    def others_of(i):
        t = test.get(i)
        if t is None:
            return None, None
        rows, how = by_key.get(prefix(i)), "id"
        if not rows:
            rows, how = by_sent.get(norm(t["sentence"])), "sentence text"
        if not rows:
            return None, None
        return [r for r in rows if norm(r["target"]) != norm(t["target"])], how

    info, via, miss = {}, Counter(), 0
    for i in err:
        o, how = others_of(i)
        if o is None:
            miss += 1
        else:
            info[i] = o
            via[how] += 1
    log(f"test samples matched to their sentence in that file: {len(info)}/{len(err)} "
        f"(via {dict(via)}; unmatched {miss}). 'Other targets' = rows of the same sentence whose target "
        "text differs.")
    log()
    log(f"{'group':<14}{'gold':<10}{'n':<6}{'has other target':<20}{'others gold: pos/neg/neu':<28}{'mean #others'}")
    for gname, ids_ in (("always-wrong", wrong), ("always-right", right)):
        for gl in range(3):
            sel = [i for i in ids_ if gold[i] == gl and i in info]
            if not sel:
                log(f"{gname:<14}{LABELS[gl]:<10}{0:<6}")
                continue
            multi = [i for i in sel if info[i]]
            oc = Counter(LABELS.index(r["gold_polarity"].lower()) for i in multi for r in info[i])
            log(f"{gname:<14}{LABELS[gl]:<10}{len(sel):<6}"
                f"{len(multi)}/{len(sel)} ({100*len(multi)/len(sel):.0f}%)".ljust(20)
                + f"{oc[0]}/{oc[1]}/{oc[2]}".ljust(28)
                + f"{np.mean([len(info[i]) for i in sel]):.2f}")
    log()
    ex = [i for i in sorted(wrong, key=lambda i: (-err[i], i)) if gold[i] == 2 and info.get(i)]
    log(f"always-wrong gold-neutral samples with other targets: {len(ex)} (showing {min(10, len(ex))})")
    for i in ex[:10]:
        t = test[i]
        top = Counter(preds[i]).most_common(1)[0][0]
        log(f"- {i} | err {err[i]:.2f} | most common pred {LABELS[top]}")
        log(f"    sentence: {t['sentence']}")
        log(f"    this target: {t['target']} [{t['gold_polarity']}]")
        log("    other targets: " + "; ".join(f"{r['target']} [{r['gold_polarity']}]" for r in info[i]))
    log()
    log("CLASS DISTRIBUTION")
    def dist(name, gs):
        c = Counter(gs); n = max(1, sum(c.values()))
        log(f"{name:<44}{n:<7}" + " | ".join(f"{l} {c[l]} ({100*c[l]/n:.1f}%)" for l in LABELS))
    dist("train pool (implicit, with aux; used for training)", [x["gold_polarity"].lower() for x in with_aux])
    if test:
        dist("test implicit (what the runs are scored on)", [t["gold_polarity"].lower() for t in test.values()])
    dist(f"test file of section 6 ({os.path.basename(args.test_all_data)})",
         [r["gold_polarity"].lower() for r in all_rows])
    tr_all = args.train_data.replace("_implicit", "_all")
    if os.path.exists(tr_all) and tr_all != args.train_data:
        dist(f"train all rows ({os.path.basename(tr_all)})",
             [r["gold_polarity"].lower() for r in json.load(open(tr_all, encoding="utf-8"))])
    log()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
