"""
Class-prior correction of the LoRA runs (train_lora_cls.py).

For every run folder  models/exp_<cfg>_lr2e-4_s<seed>  (laptop: models_laptop/) that still has
best.pt (run_experiments.ps1 keeps best.pt of lora* configs; for older runs re-run with -KeepCkpt):
  1. rebuild the exact model (build_model + best adapter) and the exact grouped val split of that seed
  2. 3-class log-probs (log-softmax over the first-token logits of the 3 labels) for
       - the implicit test file (main metric)
       - the IMPLICIT rows of the run's validation split
     cached in <run>/prior_cache.npz (so re-running only needs the CPU)
  3. methods
       raw     : argmax of the log-probs (equals test_predictions.json)
       prior   : p'(y|x) ~ p(y|x) * p_target(y) / p_train(y)
                 p_train  = class distribution of the run's training rows (all rows, after the split)
                 p_target = class distribution of the IMPLICIT rows of the training split
                 (test labels are never used)
       valbias : add one scalar b to the neutral log-prob; b in [-3, 3] step 0.1 maximises macro-F1
                 on the implicit val rows (ties -> smallest |b|)
  4. writes per run test_metrics_prior.json, test_metrics_valbias.json,
     test_predictions_prior.json, test_predictions_valbias.json
  5. results/prior_correction.txt: per config mean +/- std (raw / prior / valbias) and paired
     bootstrap of prior-raw and valbias-raw over test samples (same method as compare_runs.py)

Usage: python prior_correct.py [--dataset restaurant|laptop] [--configs lora_pol,lora_mtl,lora_mtl_shuf]
                               [--seeds 6,7,8] [--summary-only] [--recompute]
"""
import argparse
import glob
import json
import logging
import os
import re
import statistics as st
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_runs import acc_boot, macro_f1_boot  # noqa: E402  (numpy only)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prior_correct")

LABELS = ["positive", "negative", "neutral"]
NEUTRAL = 2
GRID = [round(-3.0 + 0.1 * k, 1) for k in range(61)]
METHODS = ["raw", "prior", "valbias"]


# ----------------------------------------------------------------------------
# small numpy metrics
# ----------------------------------------------------------------------------
def macro_f1(g, p):
    f = []
    for c in range(3):
        tp = np.sum((g == c) & (p == c)); fp = np.sum((g != c) & (p == c)); fn = np.sum((g == c) & (p != c))
        d = 2 * tp + fp + fn
        f.append(2 * tp / d if d else 0.0)
    return float(np.mean(f))


def acc_of(g, p):
    return float(np.mean(g == p))


def confusion(g, p):
    m = np.zeros((3, 3), dtype=int)
    for a, b in zip(g, p):
        m[a, b] += 1
    return m


# ----------------------------------------------------------------------------
# GPU part: log-probs of one run
# ----------------------------------------------------------------------------
def compute_cache(run_dir: Path, a: Namespace, cache_path: Path):
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig  # noqa: F401  (fail early if peft is missing)
    import train_lora_cls as T
    from train_mt_isa import POLARITY_TO_ID, make_grouped_validation_split, load_json, set_seed
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required to compute log-probs (use --summary-only for cached runs).")
    device = torch.device("cuda")
    set_seed(a.seed)

    # ---- same rows / split as train_lora_cls.py ----
    train_all = T.ensure_id(load_json(a.train_all_data))
    pool = [x for x in train_all if T.valid_gold(x)]
    if a.max_instances:
        pool = pool[: a.max_instances]
    train_rows, val_rows = make_grouped_validation_split(pool, a.validation_fraction, a.seed)
    train_impl = [x for x in train_rows if x.get("is_implicit")]
    val_impl = [x for x in val_rows if x.get("is_implicit")]
    cnt = lambda rows: np.array([sum(x["gold_polarity"].lower() == l for x in rows) for l in LABELS], float)
    c_train, c_target = cnt(train_rows), cnt(train_impl)
    logger.info("train rows %d (%s) | implicit train rows %d (%s) | implicit val rows %d",
                len(train_rows), c_train.astype(int).tolist(), len(train_impl),
                c_target.astype(int).tolist(), len(val_impl))
    test_rows = [x for x in T.ensure_id(load_json(a.test_data))
                 if x.get("is_implicit", True) and T.valid_gold(x)]

    tok = AutoTokenizer.from_pretrained(a.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    first = [tok(" " + w, add_special_tokens=False).input_ids[0] for w in LABELS]
    first_ids = torch.tensor(first, device=device)

    model, _ = T.build_model(a, device)
    sd = torch.load(run_dir / "best.pt", map_location=device)
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    model.eval()
    collate = T.make_collate(tok.pad_token_id, ["polarity"])

    def logprobs(rows, desc):
        items, c = [], __import__("collections").Counter()
        for x in rows:
            p = tok(T.polarity_prompt(x["sentence"], x["target"]) + T.ANSWER, add_special_tokens=False).input_ids
            t = tok(" " + x["gold_polarity"].lower(), add_special_tokens=False).input_ids + [tok.eos_token_id]
            p, t = T.fit(p, t, a.max_len, c)
            items.append({"id": x["id"], "gold_id": POLARITY_TO_ID[x["gold_polarity"].lower()],
                          "polarity": (p, t)})
        loader = DataLoader(T.SeqDataset(items), batch_size=a.eval_batch_size, shuffle=False,
                            collate_fn=collate)
        ids, gold, lp = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = T.move(batch, device)
                b = batch["polarity"]
                logits = T.forward_task(model, b, True)
                z = T.label_logits(logits, b["prompt_len"], first_ids)
                lp.append(torch.log_softmax(z, dim=-1).double().cpu().numpy())
                gold.extend(batch["gold"].cpu().tolist())
                ids.extend(batch["ids"])
                del logits, batch
        logger.info("%s: %d rows", desc, len(ids))
        return np.array(ids), np.array(gold), np.concatenate(lp)

    t_ids, t_gold, t_lp = logprobs(test_rows, "test (implicit)")
    v_ids, v_gold, v_lp = logprobs(val_impl, "val (implicit rows)")
    np.savez(cache_path, test_ids=t_ids, test_gold=t_gold, test_logp=t_lp,
             val_ids=v_ids, val_gold=v_gold, val_logp=v_lp, c_train=c_train, c_target=c_target)
    del model
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# CPU part: methods
# ----------------------------------------------------------------------------
def apply_methods(c):
    c_train, c_target = c["c_train"], c["c_target"]
    p_train = c_train / c_train.sum()
    p_target = c_target / c_target.sum()
    logw = np.log(np.maximum(p_target / p_train, 1e-6))
    t_lp, v_lp, v_gold = c["test_logp"], c["val_logp"], c["val_gold"]

    def renorm(lp):
        return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)

    out = {"p_train": p_train.tolist(), "p_target": p_target.tolist(), "log_weights": logw.tolist()}
    preds = {"raw": t_lp.argmax(1), "prior": renorm(t_lp + logw).argmax(1)}

    def with_b(lp, b):
        z = lp.copy(); z[:, NEUTRAL] += b
        return z

    scores = [(macro_f1(v_gold, with_b(v_lp, b).argmax(1)), -abs(b), b) for b in GRID]
    best_f1, _, b = max(scores)
    preds["valbias"] = with_b(t_lp, b).argmax(1)
    out["b"] = b
    out["val_f1_raw"] = macro_f1(v_gold, v_lp.argmax(1))
    out["val_f1_at_b"] = best_f1
    out["n_val_implicit"] = int(len(v_gold))
    return preds, out


def write_run(run_dir: Path, c, preds, info):
    gold = c["test_gold"]
    ids = [str(i) for i in c["test_ids"]]
    for m in ("prior", "valbias"):
        cm = confusion(gold, preds[m])
        raw = {"raw_acc": acc_of(gold, preds["raw"]), "raw_macro_f1": macro_f1(gold, preds["raw"])}
        d = {"method": m, "n": int(len(gold)), "acc": acc_of(gold, preds[m]),
             "macro_f1": macro_f1(gold, preds[m]), "confusion_matrix": cm.tolist(),
             "label_order": LABELS, "p_train": info["p_train"], "p_target": info["p_target"], **raw}
        if m == "prior":
            d["log_weights"] = info["log_weights"]
        else:
            d.update({k: info[k] for k in ("b", "val_f1_raw", "val_f1_at_b", "n_val_implicit")})
        json.dump(d, open(run_dir / f"test_metrics_{m}.json", "w", encoding="utf-8"), indent=2)
        json.dump([{"id": i, "gold": LABELS[g], "pred": LABELS[p]}
                   for i, g, p in zip(ids, gold, preds[m])],
                  open(run_dir / f"test_predictions_{m}.json", "w", encoding="utf-8"), indent=2)
        logger.info("%s | %s: acc %.4f (raw %.4f) macro-F1 %.4f (raw %.4f)%s\nconfusion (gold rows, pred cols; %s):\n%s",
                    run_dir.name, m, d["acc"], raw["raw_acc"], d["macro_f1"], raw["raw_macro_f1"],
                    f" | b={info['b']:+.1f}" if m == "valbias" else "", LABELS, cm)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["restaurant", "laptop"], default="restaurant")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--configs", default="lora_pol,lora_mtl,lora_mtl_shuf")
    ap.add_argument("--lr-tag", default="2e-4")
    ap.add_argument("--seeds", default="", help="comma list; default: every seed found")
    ap.add_argument("--summary-only", action="store_true", help="CPU only: use cached log-probs")
    ap.add_argument("--recompute", action="store_true", help="ignore existing prior_cache.npz")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mdir = Path(args.models_dir or ("models_laptop" if args.dataset == "laptop" else "models"))
    cfgs = [c for c in re.split(r"[,\s]+", args.configs) if c]
    want = {int(s) for s in re.split(r"[,\s]+", args.seeds) if s}
    out_path = Path(args.out or ("results/prior_correction_laptop.txt" if args.dataset == "laptop"
                                 else "results/prior_correction.txt"))

    runs = {}  # cfg -> seed -> dict
    for cfg in cfgs:
        for rd in sorted(mdir.glob(f"exp_{cfg}_lr{args.lr_tag}_s*")):
            m = re.match(rf"exp_{re.escape(cfg)}_lr{re.escape(args.lr_tag)}_s(\d+)$", rd.name)
            if not m or not (rd / "test_metrics.json").exists():
                continue
            seed = int(m.group(1))
            if want and seed not in want:
                continue
            cache = rd / "prior_cache.npz"
            if args.recompute or not cache.exists():
                if args.summary_only:
                    logger.warning("%s: no cache, skipped (--summary-only)", rd.name)
                    continue
                if not (rd / "best.pt").exists():
                    logger.warning("%s: best.pt missing (it was deleted) - re-run this config with "
                                   "-KeepCkpt after deleting the folder; skipped", rd.name)
                    continue
                stored = json.load(open(rd / "test_metrics.json", encoding="utf-8")).get("args", {})
                a = Namespace(**{"load_4bit": False, "max_instances": None, "validation_fraction": 0.15,
                                 "max_len": 192, "eval_batch_size": 16, "lora_r": 16, "lora_alpha": 32,
                                 "lora_dropout": 0.05, **stored})
                a.seed = seed
                logger.info("=== %s: computing log-probs ===", rd.name)
                compute_cache(rd, a, cache)
            c = dict(np.load(cache, allow_pickle=False))
            preds, info = apply_methods(c)
            agree = None
            pj = rd / "test_predictions.json"
            if pj.exists():
                ref = {r["id"]: r["pred"] for r in json.load(open(pj, encoding="utf-8"))}
                agree = np.mean([LABELS[p] == ref.get(str(i)) for i, p in zip(c["test_ids"], preds["raw"])])
                logger.info("%s: raw argmax agrees with test_predictions.json on %.1f%% of rows",
                            rd.name, 100 * agree)
            write_run(rd, c, preds, info)
            runs.setdefault(cfg, {})[seed] = {"c": c, "preds": preds, "info": info, "agree": agree}

    # ---------------- summary ----------------
    L = [f"PRIOR CORRECTION | dataset={args.dataset} | models={mdir} | lr tag {args.lr_tag}",
         "raw = argmax; prior = p*p_target/p_train (p_target = implicit train rows); valbias = scalar b on the "
         "neutral log-prob, grid [-3,3] step 0.1, chosen on the implicit val rows. No test labels used for fitting.",
         ""]
    if not runs:
        L.append("no runs with cached log-probs / best.pt found.")
    rng = np.random.default_rng(0)
    for cfg, by_seed in runs.items():
        seeds = sorted(by_seed)
        first = by_seed[seeds[0]]["c"]
        ids0, gold = list(first["test_ids"]), first["test_gold"]
        for s in seeds:  # same test set / order in every run
            assert list(by_seed[s]["c"]["test_ids"]) == ids0 and (by_seed[s]["c"]["test_gold"] == gold).all()
        N = len(ids0)
        P = {m: np.stack([by_seed[s]["preds"][m] for s in seeds], axis=1) for m in METHODS}  # (N,S)
        L.append("=" * 78)
        L.append(f"{cfg} | seeds {seeds} | n_test {N}")
        L.append("=" * 78)

        def ms(v):
            return f"{st.mean(v):.4f} +/- {st.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"
        L.append(f"{'method':<10}{'acc':<20}{'macro-F1':<20}recall pos/neg/neu (all seeds pooled)")
        for m in METHODS:
            accs = [acc_of(gold, P[m][:, k]) for k in range(len(seeds))]
            f1s = [macro_f1(gold, P[m][:, k]) for k in range(len(seeds))]
            cm = sum(confusion(gold, P[m][:, k]) for k in range(len(seeds)))
            rec = "/".join(f"{cm[i, i] / max(1, cm[i].sum()):.3f}" for i in range(3))
            L.append(f"{m:<10}{ms(accs):<20}{ms(f1s):<20}{rec}")
        bs = [by_seed[s]["info"]["b"] for s in seeds]
        L.append(f"valbias b per seed: {bs} (mean {np.mean(bs):+.2f}); val F1 raw -> at b: "
                 + ", ".join(f"{by_seed[s]['info']['val_f1_raw']:.3f}->{by_seed[s]['info']['val_f1_at_b']:.3f}"
                             for s in seeds)
                 + f" | n_val_implicit {[by_seed[s]['info']['n_val_implicit'] for s in seeds]}")
        i0 = by_seed[seeds[0]]["info"]
        L.append(f"p_train {np.round(i0['p_train'], 3).tolist()} | p_target {np.round(i0['p_target'], 3).tolist()} "
                 f"(seed {seeds[0]}; order {LABELS})")
        ag = [by_seed[s]["agree"] for s in seeds if by_seed[s]["agree"] is not None]
        if ag:
            L.append(f"raw argmax vs test_predictions.json agreement: min {100 * min(ag):.1f}%")
        W = rng.multinomial(N, np.full(N, 1.0 / N), size=args.boot).astype(float)
        base = {"acc": acc_boot(W, gold, P["raw"]).mean(axis=1),
                "f1": macro_f1_boot(W, gold, P["raw"]).mean(axis=1)}
        L.append(f"paired bootstrap over test samples ({args.boot} resamples, metric averaged over seeds; "
                 "difference vs raw, 95% CI, two-sided p):")
        for m in ("prior", "valbias"):
            cells = []
            for key, fn in (("acc", acc_boot), ("f1", macro_f1_boot)):
                d = fn(W, gold, P[m]).mean(axis=1) - base[key]
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                cells.append(f"d{key} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")
            L.append(f"  {m} - raw: " + " | ".join(cells))
        L.append("")
    text = "\n".join(L)
    print(text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
