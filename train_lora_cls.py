"""
LoRA fine-tuning of a decoder-only LLM (default Qwen/Qwen2.5-3B-Instruct) for aspect sentiment.

Reuses the data, the sentence-grouped validation split (make_grouped_validation_split from
train_mt_isa.py), the prompts and the metrics of the --train-all mode of train_mt_isa.py.

  prompt  : <existing prompt text>\nAnswer:       target: " positive" | " negative" | " neutral" + eos
  loss    : token NLL on the target tokens only (prompt tokens are masked)
  predict : logits at the last prompt position, restricted to the first token of the 3 label words
  --tasks pol : polarity only
  --tasks mtl : aspect + opinion + polarity, equal weight 1.0; rows whose aux record is
                missing / empty / <unk> get aspect+opinion loss 0 (denominator = batch size)
  --shuffle-hints : aux records permuted across train rows (cyclic shift of a seeded permutation)

Outputs (same files/format as train_mt_isa.py, so compare_runs.py works):
  <output-dir>/test_metrics.json, test_predictions.json, history.json, test_metrics_all.json, best.pt
  default output-dir: models/exp_<name>_lr2e-4_s<seed>   (laptop: models_laptop/)
"""
import argparse
import contextlib
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_mt_isa import (POLARITY_TO_ID, POLARITY_WORDS, ID_TO_POLARITY, aspect_prompt,
                          has_unk_label, label_stats, load_json,
                          make_grouped_validation_split, opinion_prompt, polarity_prompt,
                          sentence_group, set_seed)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PATHS = {
    "restaurant": dict(train="data/processed/restaurant14_train_all.json",
                       aux="data/auxiliary/restaurant14_train_all_aux_qwen7b_full.json",
                       test="data/processed/restaurant14_test_implicit.json",
                       test_all="data/processed/restaurant14_test_all.json"),
    "laptop": dict(train="data/processed/laptop14_train_all.json",
                   aux="data/auxiliary/laptop14_train_all_aux_qwen7b_full.json",
                   test="data/processed/laptop14_test_implicit.json",
                   test_all="data/processed/laptop14_test_all.json"),
}
ANSWER = "\nAnswer:"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def ensure_id(rows):  # *_all.json rows only have "instance_id"
    for x in rows:
        x.setdefault("id", x.get("instance_id"))
    return rows


def valid_gold(x):
    return x["gold_polarity"].lower() in POLARITY_TO_ID


class SeqDataset(Dataset):
    """One item per row: token-id lists (prompt, target) for each task."""

    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(pad_id: int, tasks: List[str]):
    def collate(batch):
        out = {"ids": [b["id"] for b in batch],
               "gold": torch.tensor([b["gold_id"] for b in batch])}
        if "aux_mask" in batch[0]:
            out["aux_mask"] = torch.tensor([b["aux_mask"] for b in batch], dtype=torch.float)
        for t in tasks:
            seqs = [b[t][0] + b[t][1] for b in batch]
            L = max(len(s) for s in seqs)
            ids = torch.tensor([s + [pad_id] * (L - len(s)) for s in seqs])
            att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
            lab = torch.tensor([[-100] * len(b[t][0]) + b[t][1] + [-100] * (L - len(s))
                                for b, s in zip(batch, seqs)])
            out[t] = {"input_ids": ids, "attention_mask": att, "labels": lab,
                      "prompt_len": torch.tensor([len(b[t][0]) for b in batch])}
        return out
    return collate


def fit(prompt_ids: List[int], target_ids: List[int], max_len: int, counter: Counter):
    """Keep the tail of the prompt (it ends with 'Answer:') if prompt+target is too long."""
    room = max_len - len(target_ids)
    if len(prompt_ids) > room:
        counter["truncated"] += 1
        prompt_ids = prompt_ids[-room:]
    return prompt_ids, target_ids


# ----------------------------------------------------------------------------
# loss / prediction on the causal LM
# ----------------------------------------------------------------------------
def per_sample_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean NLL over the target tokens of each row (fp32). Only target positions are upcast."""
    tgt = labels[:, 1:]
    mask = tgt != -100
    sel = logits[:, :-1][mask].float()
    nll = F.cross_entropy(sel, tgt[mask], reduction="none")
    rows = mask.nonzero()[:, 0]
    tot = torch.zeros(labels.size(0), device=logits.device).index_add_(0, rows, nll)
    return tot / mask.sum(dim=1).clamp(min=1).float()


def label_logits(logits: torch.Tensor, prompt_len: torch.Tensor, first_ids: torch.Tensor):
    """[B,3] logits of the label first-tokens at the answer position (last prompt token)."""
    pos = (prompt_len - 1).to(logits.device)
    at = logits[torch.arange(logits.size(0), device=logits.device), pos]
    return at.index_select(-1, first_ids).float()


def forward_task(model, b, use_amp):
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    with ctx:
        return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits


def move(b, device):
    return {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict)
                else (v.to(device) if torch.is_tensor(v) else v)) for k, v in b.items()}


def evaluate(model, loader, device, first_ids, use_amp, desc):
    model.eval()
    tot, n, ids, gold, pred = 0.0, 0, [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty()):
            batch = move(batch, device)
            p = batch["polarity"]
            logits = forward_task(model, p, use_amp)
            nll = per_sample_nll(logits, p["labels"])
            if torch.isfinite(nll).all():
                tot += float(nll.sum()); n += nll.numel()
            pred.extend(label_logits(logits, p["prompt_len"], first_ids).argmax(-1).cpu().tolist())
            gold.extend(batch["gold"].cpu().tolist())
            ids.extend(batch["ids"])
            del logits, batch
    f1 = float(f1_score(gold, pred, labels=[0, 1, 2], average="macro", zero_division=0)) if gold else 0.0
    acc = float(np.mean(np.array(gold) == np.array(pred))) if gold else 0.0
    return {"loss": tot / max(1, n), "f1": f1, "acc": acc, "ids": ids, "gold": gold, "pred": pred}


def eval_rows(rows, tok, args, device, first_ids, use_amp, model, collate, desc):
    cnt = Counter()
    items = []
    for x in rows:
        p = tok(polarity_prompt(x["sentence"], x["target"]) + ANSWER, add_special_tokens=False).input_ids
        t = tok(" " + x["gold_polarity"].lower(), add_special_tokens=False).input_ids + [tok.eos_token_id]
        p, t = fit(p, t, args.max_len, cnt)
        items.append({"id": x["id"], "gold_id": POLARITY_TO_ID[x["gold_polarity"].lower()],
                      "polarity": (p, t)})
    loader = DataLoader(SeqDataset(items), batch_size=args.eval_batch_size, shuffle=False,
                        collate_fn=collate)
    return evaluate(model, loader, device, first_ids, use_amp, desc)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", choices=["pol", "mtl"], default="pol")
    ap.add_argument("--shuffle-hints", action="store_true")
    ap.add_argument("--dataset", choices=list(PATHS), default="restaurant")
    ap.add_argument("--train-all-data", default=None)
    ap.add_argument("--train-all-aux", default=None)
    ap.add_argument("--test-data", default=None, help="implicit test file (main metric)")
    ap.add_argument("--test-all-data", default=None, help="all test rows (extra metric)")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--unk-tokenizer", default="google/flan-t5-base",
                    help="tokenizer used for the <unk> aux check, so the masked rows are the same "
                         "as in train_mt_isa.py (Qwen's byte-level BPE has no <unk>)")
    ap.add_argument("--name", default=None, help="run name; default lora_<tasks>[_shuf]")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--num-epochs", "--epochs", dest="num_epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--max-instances", type=int, default=None, help="debug: cap train rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", help="accepted for run_experiments.ps1; bf16 is always on")
    args = ap.parse_args()

    d = PATHS[args.dataset]
    args.train_all_data = args.train_all_data or d["train"]
    args.train_all_aux = args.train_all_aux or d["aux"]
    args.test_data = args.test_data or d["test"]
    args.test_all_data = args.test_all_data or d["test_all"]
    name = args.name or ("lora_" + args.tasks + ("_shuf" if args.shuffle_hints else ""))
    if args.shuffle_hints and args.tasks != "mtl":
        raise ValueError("--shuffle-hints needs --tasks mtl.")
    out_dir = Path(args.output_dir or
                   f"{'models_laptop' if args.dataset == 'laptop' else 'models'}/exp_{name}_lr2e-4_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- environment checks (this script never changes the installed packages) ----
    import transformers
    try:
        import peft
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise SystemExit("peft is not installed (pip install 'peft>=0.7'): " + str(e))
    ver = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
    if ver < (4, 37):
        raise SystemExit(
            f"transformers {transformers.__version__} is installed but Qwen2.5 needs >= 4.37 "
            "(requirements.txt pins 4.35.0). This script does not upgrade anything: use an "
            "environment with transformers >= 4.37 for the LoRA runs.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required (bf16 LoRA on a 3B model).")
    device = torch.device("cuda")
    use_amp = True
    set_seed(args.seed)
    logger.info("transformers %s | peft %s | torch %s | args: %s",
                transformers.__version__, peft.__version__, torch.__version__, vars(args))

    # ---- data: same rows and split as train_mt_isa.py --train-all ----
    train_all = ensure_id(load_json(args.train_all_data))
    pool = [x for x in train_all if valid_gold(x)]
    logger.info("train rows: %d | valid polarity: %d", len(train_all), len(pool))
    if args.max_instances:
        pool = pool[: args.max_instances]
    train_rows, val_rows = make_grouped_validation_split(pool, args.validation_fraction, args.seed)
    shared = {sentence_group(x) for x in val_rows} & {sentence_group(x) for x in train_rows}
    logger.info("grouped val split: train %d | val %d | sentences shared: %d", len(train_rows),
                len(val_rows), len(shared))
    if shared:
        raise RuntimeError("sentence overlap between train and val")
    label_stats(train_rows, "train")
    label_stats(val_rows, "val")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id, eos_id = tok.pad_token_id, tok.eos_token_id

    first = []
    for w in POLARITY_WORDS:
        ids = tok(" " + w, add_special_tokens=False).input_ids
        first.append(ids[0])
        logger.info("label %-9s ' %s' -> token ids %s (first: %r)", w, w, ids,
                    tok.convert_ids_to_tokens(ids[0]))
    if len(set(first)) != len(POLARITY_WORDS):
        raise AssertionError(f"label first tokens are not distinct: {first}")
    first_ids = torch.tensor(first, device=device)

    # ---- aux records for --tasks mtl ----
    cnt = Counter()
    aux_rec: Dict[str, Optional[Dict]] = {}
    aux_ok: Dict[str, bool] = {}
    if args.tasks == "mtl":
        aux_by_id = {a["instance_id"]: a for a in load_json(args.train_all_aux)}
        try:
            unk_tok = AutoTokenizer.from_pretrained(args.unk_tokenizer)
        except Exception as e:  # pragma: no cover
            logger.warning("cannot load %s for the <unk> check (%s); using the LLM tokenizer.",
                           args.unk_tokenizer, e)
            unk_tok = tok
        why = Counter()
        for x in train_rows:
            a = aux_by_id.get(x["id"])
            if a is None:
                st = "missing"
            elif not (a.get("aspect") or "").strip() or not (a.get("opinion") or "").strip():
                st = "empty"
            elif has_unk_label(a, unk_tok):
                st = "unk"
            else:
                st = "ok"
            aux_rec[x["id"]], aux_ok[x["id"]] = a, st == "ok"
            why[st] += 1
        n_mask = len(train_rows) - why["ok"]
        logger.info("mtl aux: %d rows | aux loss MASKED for %d (%.1f%%): missing %d, empty %d, <unk> %d | "
                    "usable %d | aux losses averaged over ALL rows in the batch", len(train_rows),
                    n_mask, 100 * n_mask / max(1, len(train_rows)), why["missing"], why["empty"],
                    why["unk"], why["ok"])
        if args.shuffle_hints:
            ids_ = [x["id"] for x in train_rows]
            n = len(ids_)
            order = np.random.RandomState(args.seed).permutation(n)
            donor = {ids_[order[k]]: ids_[order[(k + 1) % n]] for k in range(n)}
            old_rec, old_ok = dict(aux_rec), dict(aux_ok)
            for i, dn in donor.items():
                aux_rec[i], aux_ok[i] = old_rec[dn], old_ok[dn]
            logger.info("--shuffle-hints: every train row uses another row's aux record "
                        "(record and its validity move together)")
            for i in ids_[:5]:
                logger.info("[shuffle-hints] %s -> hints of %s", i, donor[i])

    def build(x, train: bool):
        s, t = x["sentence"], x["target"]
        gold = x["gold_polarity"].lower()
        item = {"id": x["id"], "gold_id": POLARITY_TO_ID[gold]}
        pol_p = tok(polarity_prompt(s, t) + ANSWER, add_special_tokens=False).input_ids
        pol_t = tok(" " + gold, add_special_tokens=False).input_ids + [eos_id]
        item["polarity"] = fit(pol_p, pol_t, args.max_len, cnt)
        if train and args.tasks == "mtl":
            a = aux_rec.get(x["id"]) or {}
            asp = (a.get("aspect") or "").strip() or t
            opi = (a.get("opinion") or "").strip() or "none"
            for key, prompt, ans in (("aspect", aspect_prompt(s, t), asp),
                                     ("opinion", opinion_prompt(s, t, asp), opi)):
                p = tok(prompt + ANSWER, add_special_tokens=False).input_ids
                tt = tok(" " + ans, add_special_tokens=False).input_ids + [eos_id]
                item[key] = fit(p, tt, args.max_len, cnt)
            item["aux_mask"] = 1.0 if aux_ok.get(x["id"], False) else 0.0
        return item

    tasks = ["polarity"] + (["aspect", "opinion"] if args.tasks == "mtl" else [])
    train_items = [build(x, True) for x in train_rows]
    val_items = [build(x, False) for x in val_rows]
    logger.info("sequences truncated to --max-len %d (prompt tail kept): %d", args.max_len, cnt["truncated"])
    collate_train = make_collate(pad_id, tasks)
    collate_eval = make_collate(pad_id, ["polarity"])
    train_loader = DataLoader(SeqDataset(train_items), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_train, drop_last=False)
    val_loader = DataLoader(SeqDataset(val_items), batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=collate_eval)
    ex = train_items[0]
    for k in tasks:
        logger.info("[%s] input : %s", k, tok.decode(ex[k][0]))
        logger.info("[%s] target: %r", k, tok.decode(ex[k][1]))

    # ---- model: bf16 base + LoRA on all linear layers ----
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for checkpointing with a frozen embedding
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:            # fp32 adapters: bf16 weights would swallow small updates
        p.data = p.data.float()
    n_tr = sum(p.numel() for p in trainable)
    logger.info("trainable LoRA params: %.2fM of %.2fM", n_tr / 1e6,
                sum(p.numel() for p in model.parameters()) / 1e6)

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total = max(1, steps_per_epoch * args.num_epochs)
    warm = int(args.warmup_ratio * total)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- training ----
    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.num_epochs + 1):
        logger.info("=" * 70)
        logger.info("Epoch %d/%d", epoch, args.num_epochs)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        parts = {k: [0.0, 0] for k in tasks}
        tot_loss, n_ok, n_bad, consec_bad = 0.0, 0, 0, 0
        gold, pred = [], []
        pbar = tqdm(train_loader, desc=f"train {epoch}", leave=False, disable=not sys.stderr.isatty())
        for i, batch in enumerate(pbar):
            batch = move(batch, device)
            step_losses, ok = {}, True
            for k in tasks:        # one backward per task: same gradient as the summed loss, less memory
                b = batch[k]
                logits = forward_task(model, b, use_amp)
                nll = per_sample_nll(logits, b["labels"])
                if k != "polarity":
                    nll = nll * batch["aux_mask"]       # masked rows -> 0, mean over the full batch
                loss = nll.mean()
                if not torch.isfinite(loss):
                    ok = False
                    del logits, nll, loss
                    break
                (loss / args.grad_accum).backward()
                step_losses[k] = float(loss.detach())
                if k == "polarity":
                    pred.extend(label_logits(logits.detach(), b["prompt_len"], first_ids)
                                .argmax(-1).cpu().tolist())
                    gold.extend(batch["gold"].cpu().tolist())
                del logits, nll, loss
            if not ok:
                n_bad += 1
                consec_bad += 1
                logger.warning("Non-finite loss at step %d - batch skipped", i)
                optimizer.zero_grad(set_to_none=True)
                if consec_bad >= 20:
                    raise RuntimeError("20 consecutive non-finite losses - training diverged.")
                del batch
                continue
            consec_bad = 0
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if torch.isfinite(gn):
                    optimizer.step()
                    scheduler.step()
                else:
                    n_bad += 1
                    logger.warning("Non-finite grad norm at step %d - update skipped", i)
                optimizer.zero_grad(set_to_none=True)
            for k, v in step_losses.items():
                parts[k][0] += v; parts[k][1] += 1
            tot_loss += sum(step_losses.values()); n_ok += 1
            pbar.set_postfix(loss=f"{tot_loss / n_ok:.4f}")
            del batch
        tr_f1 = float(f1_score(gold, pred, labels=[0, 1, 2], average="macro", zero_division=0)) if gold else 0.0
        tr_acc = float(np.mean(np.array(gold) == np.array(pred))) if gold else 0.0
        va = evaluate(model, val_loader, device, first_ids, use_amp, f"val {epoch}")
        secs = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        parts_m = {k: v[0] / max(1, v[1]) for k, v in parts.items()}
        logger.info("Train loss %.4f | F1 %.4f | Acc %.4f | parts %s | skipped %d",
                    tot_loss / max(1, n_ok), tr_f1, tr_acc,
                    {k: round(v, 4) for k, v in parts_m.items()}, n_bad)
        logger.info("Val   loss %.4f | F1 %.4f | Acc %.4f", va["loss"], va["f1"], va["acc"])
        logger.info("Epoch %d: %.0f s | GPU memory peak %.2f GiB", epoch, secs, peak)
        history.append({"epoch": epoch, "train_loss": tot_loss / max(1, n_ok), "train_f1": tr_f1,
                        "train_acc": tr_acc, "train_parts": parts_m, "train_skipped": n_bad,
                        "val_loss": va["loss"], "val_f1": va["f1"], "val_acc": va["acc"],
                        "seconds": secs, "gpu_mem_peak_gib": peak})
        if va["f1"] > best_f1:
            best_f1, best_epoch = va["f1"], epoch
            torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                       out_dir / "best.pt")
            logger.info("New best (val macro-F1 %.4f) saved.", best_f1)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Done. Best val macro-F1 %.4f at epoch %d", best_f1, best_epoch)

    # ---- test: best adapter ----
    sd = torch.load(out_dir / "best.pt", map_location=device)
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    test_rows = [x for x in ensure_id(load_json(args.test_data))
                 if x.get("is_implicit", True) and valid_gold(x)]
    label_stats(test_rows, "test")
    te = eval_rows(test_rows, tok, args, device, first_ids, use_amp, model, collate_eval, "test")
    logger.info("TEST (best ckpt, epoch %d): Acc %.4f | macro-F1 %.4f", best_epoch, te["acc"], te["f1"])
    logger.info("Confusion matrix (rows=gold, cols=pred; order %s):\n%s", POLARITY_WORDS,
                confusion_matrix(te["gold"], te["pred"], labels=[0, 1, 2]))
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"acc": te["acc"], "macro_f1": te["f1"], "best_epoch": best_epoch,
                   "val_f1": best_f1, "args": vars(args)}, f, indent=2)
    with open(out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump([{"id": i, "gold": ID_TO_POLARITY[g], "pred": ID_TO_POLARITY[p]}
                   for i, g, p in zip(te["ids"], te["gold"], te["pred"])], f, indent=2)

    all_rows = [x for x in ensure_id(load_json(args.test_all_data)) if valid_gold(x)]
    label_stats(all_rows, "test_all")
    ta = eval_rows(all_rows, tok, args, device, first_ids, use_amp, model, collate_eval, "test_all")
    cm = confusion_matrix(ta["gold"], ta["pred"], labels=[0, 1, 2])
    logger.info("TEST ALL (%d rows, best ckpt, epoch %d): Acc %.4f | macro-F1 %.4f",
                len(all_rows), best_epoch, ta["acc"], ta["f1"])
    logger.info("Confusion matrix test_all (rows=gold, cols=pred; order %s):\n%s", POLARITY_WORDS, cm)
    implicit_of = {x["id"]: bool(x.get("is_implicit")) for x in all_rows}
    subsets = {}
    for nm, flag in (("implicit", True), ("explicit", False)):
        idx = [k for k, i in enumerate(ta["ids"]) if implicit_of[i] == flag]
        if idx:
            g = [ta["gold"][k] for k in idx]
            p = [ta["pred"][k] for k in idx]
            subsets[nm] = {"n": len(idx), "acc": float(np.mean(np.array(g) == np.array(p))),
                           "macro_f1": float(f1_score(g, p, labels=[0, 1, 2], average="macro",
                                                      zero_division=0))}
    logger.info("test_all subsets: %s", subsets)
    with open(out_dir / "test_metrics_all.json", "w", encoding="utf-8") as f:
        json.dump({"n": len(all_rows), "acc": ta["acc"], "macro_f1": ta["f1"],
                   "confusion_matrix": cm.tolist(), "label_order": POLARITY_WORDS,
                   "subsets": subsets, "best_epoch": best_epoch, "val_f1": best_f1,
                   "test_file": args.test_all_data}, f, indent=2)


if __name__ == "__main__":
    main()
