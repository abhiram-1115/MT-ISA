"""
Cross-fitted LoRA teacher for knowledge distillation (run in venv_lora).

5 folds over ALL train rows (restaurant14_train_all.json, 3602 rows), grouped by sentence_id
(no sentence in two folds) and stratified by polarity (greedy), fold seed 0.
For each fold k: train a fresh Qwen2.5-3B-Instruct LoRA with the lora_pol settings of
train_lora_cls.py (tasks pol, same prompt / LoRA / lr / batch / max_len, 3 epochs, no early
stopping, no validation) on the other 4 folds, then predict 3-class probabilities
(softmax over the 3 label first-token logits at the answer position) on fold k.

Output: data/teacher/<dataset>14_train_all_teacher_probs.json
        {instance_id: [p_positive, p_negative, p_neutral]} for every row (out-of-fold)
        + <same name>.meta.json (fold of every row, per-fold / overall OOF metrics, args)
Finished folds are cached in data/teacher/<name>.fold<k>.json, so a crashed run resumes.

Usage: .\\venv_lora\\Scripts\\python.exe teacher_folds.py [--dataset laptop]
"""
import argparse
import gc
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_lora_cls as T
from train_mt_isa import POLARITY_TO_ID, POLARITY_WORDS, load_json, sentence_group, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("teacher_folds")


def make_folds(rows, n_folds: int, seed: int):
    """Sentence-grouped, polarity-stratified folds. Groups are visited largest first (random
    order within a size, seeded) and each goes to the fold where it least increases the squared
    deviation of the per-class counts from their targets (total_c / n_folds)."""
    classes = list(POLARITY_TO_ID)
    groups = {}
    for x in rows:
        groups.setdefault(sentence_group(x), []).append(x)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))       # stable: random order within a size
    total = Counter(x["gold_polarity"].lower() for x in rows)
    target = {c: total[c] / n_folds for c in classes}
    cnt = [Counter() for _ in range(n_folds)]
    fold_of = {}
    for k in keys:
        g = Counter(x["gold_polarity"].lower() for x in groups[k])

        def cost(f):
            return sum((cnt[f][c] + g[c] - target[c]) ** 2 - (cnt[f][c] - target[c]) ** 2
                       for c in classes)
        f = min(range(n_folds), key=lambda f: (cost(f), sum(cnt[f].values()), f))
        cnt[f] += g
        for x in groups[k]:
            fold_of[x["id"]] = f
    return fold_of, cnt


def metrics(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    if len(gold) == 0:
        return {"n": 0, "acc": 0.0, "macro_f1": 0.0}
    return {"n": int(len(gold)), "acc": float(np.mean(gold == pred)),
            "macro_f1": float(f1_score(gold, pred, labels=[0, 1, 2], average="macro", zero_division=0))}


def train_and_predict(args, tok, train_rows, pred_rows, device, first_ids, fold):
    cnt = Counter()

    def build(x):
        gold = x["gold_polarity"].lower()
        p = tok(T.polarity_prompt(x["sentence"], x["target"]) + T.ANSWER, add_special_tokens=False).input_ids
        t = tok(" " + gold, add_special_tokens=False).input_ids + [tok.eos_token_id]
        return {"id": x["id"], "gold_id": POLARITY_TO_ID[gold], "polarity": T.fit(p, t, args.max_len, cnt)}

    collate = T.make_collate(tok.pad_token_id, ["polarity"])
    train_loader = DataLoader(T.SeqDataset([build(x) for x in train_rows]), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    pred_loader = DataLoader(T.SeqDataset([build(x) for x in pred_rows]), batch_size=args.eval_batch_size,
                             shuffle=False, collate_fn=collate)
    if cnt["truncated"]:
        logger.info("fold %d: %d sequences truncated to --max-len %d", fold, cnt["truncated"], args.max_len)

    model, trainable = T.build_model(args, device)
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total = max(1, steps_per_epoch * args.num_epochs)
    warm = int(args.warmup_ratio * total)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(1, args.num_epochs + 1):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tot, n_ok, n_bad, consec_bad = 0.0, 0, 0, 0
        pbar = tqdm(train_loader, desc=f"fold {fold} ep {epoch}", leave=False, disable=not sys.stderr.isatty())
        for i, batch in enumerate(pbar):
            batch = T.move(batch, device)
            b = batch["polarity"]
            logits = T.forward_task(model, b, True)
            loss = T.per_sample_nll(logits, b["labels"]).mean()
            del logits
            if not torch.isfinite(loss):
                n_bad += 1
                consec_bad += 1
                logger.warning("fold %d: non-finite loss at step %d - batch skipped", fold, i)
                optimizer.zero_grad(set_to_none=True)
                if consec_bad >= 20:
                    raise RuntimeError("20 consecutive non-finite losses - training diverged.")
                del loss, batch
                continue
            consec_bad = 0
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if torch.isfinite(gn):
                    optimizer.step()
                    scheduler.step()
                else:
                    n_bad += 1
                    logger.warning("fold %d: non-finite grad norm at step %d - update skipped", fold, i)
                optimizer.zero_grad(set_to_none=True)
            tot += float(loss.detach())
            n_ok += 1
            pbar.set_postfix(loss=f"{tot / n_ok:.4f}")
            del loss, batch
        logger.info("fold %d epoch %d/%d: train loss %.4f | skipped %d | %.0f s | GPU peak %.2f GiB",
                    fold, epoch, args.num_epochs, tot / max(1, n_ok), n_bad, time.time() - t0,
                    torch.cuda.max_memory_allocated() / 2**30)

    model.eval()
    probs, ids, gold = [], [], []
    with torch.no_grad():
        for batch in pred_loader:
            batch = T.move(batch, device)
            b = batch["polarity"]
            logits = T.forward_task(model, b, True)
            z = T.label_logits(logits, b["prompt_len"], first_ids)
            probs.extend(torch.softmax(z, dim=-1).double().cpu().tolist())
            ids.extend(batch["ids"])
            gold.extend(batch["gold"].cpu().tolist())
            del logits, batch

    # free everything before the next fold
    del model, trainable, optimizer, scheduler, train_loader, pred_loader
    gc.collect()
    torch.cuda.empty_cache()
    return ids, gold, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(T.PATHS), default="restaurant")
    ap.add_argument("--train-all-data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="training seed (fold k uses seed + k)")
    ap.add_argument("--folds", default="", help="comma list of folds to run (default: all)")
    ap.add_argument("--recompute", action="store_true", help="ignore cached fold files")
    # lora_pol settings of train_lora_cls.py
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--num-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-instances", type=int, default=None, help="debug: first N valid rows")
    args = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer
    import re
    if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (4, 37):
        raise SystemExit(f"transformers {transformers.__version__}: Qwen2.5 needs >= 4.37 - use venv_lora.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    device = torch.device("cuda")

    args.train_all_data = args.train_all_data or T.PATHS[args.dataset]["train"]
    name = f"{'laptop14' if args.dataset == 'laptop' else 'restaurant14'}_train_all_teacher_probs"
    if args.max_instances:
        name += f"_first{args.max_instances}"   # debug runs never overwrite / reuse the real file
    out = Path(args.out or f"data/teacher/{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("transformers %s | torch %s | args %s", transformers.__version__, torch.__version__, vars(args))

    rows = [x for x in T.ensure_id(load_json(args.train_all_data)) if T.valid_gold(x)]
    if args.max_instances:
        rows = rows[: args.max_instances]
    fold_of, fold_cnt = make_folds(rows, args.n_folds, args.fold_seed)
    for f in range(args.n_folds):
        logger.info("fold %d: %d rows | %s", f, sum(fold_cnt[f].values()), dict(fold_cnt[f]))
    sent_folds = {}
    for x in rows:
        sent_folds.setdefault(sentence_group(x), set()).add(fold_of[x["id"]])
    shared = sum(1 for v in sent_folds.values() if len(v) > 1)
    logger.info("%d rows, %d sentences; sentences in more than one fold: %d", len(rows), len(sent_folds), shared)
    if shared:
        raise RuntimeError("a sentence is split across folds")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    first = [tok(" " + w, add_special_tokens=False).input_ids[0] for w in POLARITY_WORDS]
    if len(set(first)) != 3:
        raise AssertionError(f"label first tokens not distinct: {first}")
    logger.info("label first tokens %s -> %s", POLARITY_WORDS, [tok.convert_ids_to_tokens(i) for i in first])
    first_ids = torch.tensor(first, device=device)

    run = [int(f) for f in args.folds.split(",") if f.strip()] or list(range(args.n_folds))
    by_id = {x["id"]: x for x in rows}
    for f in run:
        cache = out.with_name(f"{out.stem}.fold{f}.json")
        if cache.exists() and not args.recompute:
            logger.info("fold %d: cached in %s - skipped", f, cache)
            continue
        tr = [x for x in rows if fold_of[x["id"]] != f]
        te = [x for x in rows if fold_of[x["id"]] == f]
        logger.info("=" * 70)
        logger.info("fold %d/%d: train %d rows -> predict %d rows", f, args.n_folds, len(tr), len(te))
        set_seed(args.seed + f)
        t0 = time.time()
        ids, gold, probs = train_and_predict(args, tok, tr, te, device, first_ids, f)
        m = metrics(gold, np.argmax(probs, axis=1))
        logger.info("fold %d OOF: acc %.4f | macro-F1 %.4f | %.0f min", f, m["acc"], m["macro_f1"],
                    (time.time() - t0) / 60)
        json.dump({"fold": f, "metrics": m, "probs": dict(zip(ids, probs))},
                  open(cache, "w", encoding="utf-8"))

    # ---- merge all folds ----
    probs, per_fold = {}, {}
    for f in range(args.n_folds):
        cache = out.with_name(f"{out.stem}.fold{f}.json")
        if not cache.exists():
            logger.warning("fold %d missing - final file not written (run the remaining folds)", f)
            return
        d = json.load(open(cache, encoding="utf-8"))
        probs.update(d["probs"])
        per_fold[f] = d["metrics"]
    missing = [i for i in by_id if i not in probs]
    if missing:
        raise RuntimeError(f"{len(missing)} rows have no out-of-fold probability (e.g. {missing[:3]})")

    ids = list(by_id)
    gold = [POLARITY_TO_ID[by_id[i]["gold_polarity"].lower()] for i in ids]
    pred = [int(np.argmax(probs[i])) for i in ids]
    impl = [k for k, i in enumerate(ids) if by_id[i].get("is_implicit")]
    overall = metrics(gold, pred)
    implicit = metrics([gold[k] for k in impl], [pred[k] for k in impl])
    for f in range(args.n_folds):
        logger.info("fold %d OOF: acc %.4f | macro-F1 %.4f (n=%d)", f, per_fold[f]["acc"],
                    per_fold[f]["macro_f1"], per_fold[f]["n"])
    logger.info("OVERALL OOF (%d rows): acc %.4f | macro-F1 %.4f", overall["n"], overall["acc"], overall["macro_f1"])
    logger.info("IMPLICIT rows OOF (%d rows): acc %.4f | macro-F1 %.4f", implicit["n"], implicit["acc"],
                implicit["macro_f1"])

    json.dump({i: probs[i] for i in ids}, open(out, "w", encoding="utf-8"))
    json.dump({"label_order": POLARITY_WORDS, "fold_of": {i: fold_of[i] for i in ids},
               "per_fold": per_fold, "overall": overall, "implicit": implicit, "args": vars(args)},
              open(out.with_name(out.stem + ".meta.json"), "w", encoding="utf-8"), indent=2)
    logger.info("written %s (%d rows, order %s)", out, len(ids), POLARITY_WORDS)


if __name__ == "__main__":
    main()
