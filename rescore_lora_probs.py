"""
Write 3-class probabilities for the existing LoRA runs (run in venv_lora).

Reuses the scoring code of prior_correct.py (compute_cache: softmax over the 3 label first-token
logits at the answer position, rebuilt exactly as in train_lora_cls.py). An existing
<run>/prior_cache.npz from prior_correct.py is reused as is; otherwise it is computed (needs best.pt).

For every run folder models/exp_<cfg>_lr2e-4_s<seed> (cfg in lora_pol, lora_mtl, lora_mtl_shuf;
seeds 6-11) writes <run>/test_probs.json: [{id, gold, pred, probs}] for the implicit test rows,
probs = [p_positive, p_negative, p_neutral]. Checks that argmax(probs) equals the stored pred of
test_predictions.json for every row (prints the agreement; fails at the end if any run differs).

Usage: .\\venv_lora\\Scripts\\python.exe rescore_lora_probs.py [--configs ...] [--seeds 6,7,8,9,10,11]
"""
import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np

from prior_correct import LABELS, compute_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rescore_lora_probs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--configs", default="lora_pol,lora_mtl,lora_mtl_shuf")
    ap.add_argument("--seeds", default="6,7,8,9,10,11")
    ap.add_argument("--lr-tag", default="2e-4")
    ap.add_argument("--recompute", action="store_true", help="ignore an existing prior_cache.npz")
    args = ap.parse_args()

    bad, done = [], 0
    for cfg in [c for c in args.configs.split(",") if c.strip()]:
        for sd in [int(s) for s in args.seeds.split(",") if s.strip()]:
            rd = Path(args.models_dir) / f"exp_{cfg}_lr{args.lr_tag}_s{sd}"
            if not (rd / "test_predictions.json").exists():
                logger.warning("%s: missing (no test_predictions.json) - skipped", rd)
                continue
            cache = rd / "prior_cache.npz"
            if args.recompute or not cache.exists():
                if not (rd / "best.pt").exists():
                    logger.warning("%s: no best.pt and no prior_cache.npz - cannot score, skipped", rd)
                    continue
                stored = json.load(open(rd / "test_metrics.json", encoding="utf-8")).get("args", {})
                a = Namespace(**{"load_4bit": False, "max_instances": None, "validation_fraction": 0.15,
                                 "max_len": 192, "eval_batch_size": 16, "lora_r": 16, "lora_alpha": 32,
                                 "lora_dropout": 0.05, **stored})
                a.seed = sd
                logger.info("=== %s: scoring ===", rd.name)
                compute_cache(rd, a, cache)
            c = np.load(cache, allow_pickle=False)
            ids = [str(i) for i in c["test_ids"]]
            gold = c["test_gold"]
            probs = np.exp(c["test_logp"])
            probs = probs / probs.sum(axis=1, keepdims=True)
            pred = probs.argmax(axis=1)
            stored_pred = {r["id"]: r["pred"] for r in json.load(open(rd / "test_predictions.json", encoding="utf-8"))}
            agree = [LABELS[p] == stored_pred.get(i) for i, p in zip(ids, pred)]
            rows = [{"id": i, "gold": LABELS[int(g)], "pred": LABELS[int(p)],
                     "probs": [round(float(v), 5) for v in pr]}
                    for i, g, p, pr in zip(ids, gold, pred, probs)]
            with open(rd / "test_probs.json", "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            frac = float(np.mean(agree))
            logger.info("%s: %d rows | argmax(probs) == stored pred: %.1f%% (%d/%d) -> test_probs.json",
                        rd.name, len(rows), 100 * frac, sum(agree), len(agree))
            done += 1
            if frac < 1.0:
                bad.append((rd.name, len(agree) - sum(agree)))
    logger.info("written test_probs.json for %d runs", done)
    assert not bad, f"argmax(probs) differs from the stored pred in: {bad}"


if __name__ == "__main__":
    main()
