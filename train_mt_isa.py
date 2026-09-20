"""
MT-ISA training script (v2)

Fixes vs. the previous version
------------------------------
* Label padding is -100 (before, pad tokens were trained as targets).
* Polarity is trained with LM loss on the label word (see mt_isa_model.py).
* D-AWL (input / output / input_output / none) is really applied.
* T-AWL uncertainty parameters get their own optimizer group
  (higher LR, no weight decay) so the task weights can actually move.
* Dynamic padding (no 256-token padding), gradient accumulation, optional bf16.
* NaN/Inf guards: non-finite losses / gradients are skipped and counted, and
  every printed average is computed over finite steps only.
* Only the best model weights are saved (no 20 x 3 GB checkpoints).
* Optional held-out test evaluation of the best checkpoint (--test-data).
* Data diagnostics + zero-shot (epoch 0) validation score printed at start.
"""

import argparse
import contextlib
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from mt_isa_model import CONF_MIN, POLARITY_WORDS, MTISAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

POLARITY_TO_ID = {w: i for i, w in enumerate(POLARITY_WORDS)}
ID_TO_POLARITY = {i: w for w, i in POLARITY_TO_ID.items()}


# ----------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_conf(x, default: float = CONF_MIN) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return min(1.0, max(CONF_MIN, x))


def convergence_conf(aux: Dict) -> float:
    """Alternative confidence from the refinement trace (usable with existing aux data)."""
    if not aux.get("converged", False):
        return CONF_MIN
    iters = int(aux.get("refinement_iterations", 1) or 1)
    return max(CONF_MIN, 1.0 - 0.05 * (iters - 1))


def make_validation_split(instances: List[Dict], fraction: float, seed: int):
    """Stratified split by polarity; never touches the test set."""
    rng = random.Random(seed)
    by_label = {label: [] for label in POLARITY_TO_ID}
    for x in instances:
        by_label[x["gold_polarity"].lower()].append(x)
    train, val = [], []
    for label in POLARITY_TO_ID:  # fixed order -> reproducible
        rows = by_label[label]
        rng.shuffle(rows)
        n_val = max(1, int(round(len(rows) * fraction))) if rows else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def describe_aux(aux_rows: List[Dict]):
    if not aux_rows:
        return
    n = len(aux_rows)
    conv = sum(1 for a in aux_rows if a.get("converged"))
    iters = [a.get("refinement_iterations", 0) for a in aux_rows]
    same = sum(
        1 for a in aux_rows
        if str(a.get("aspect", "")).strip().lower() == str(a.get("target", "")).strip().lower()
    )
    a_conf = Counter(round(clean_conf(a.get("aspect_confidence")), 2) for a in aux_rows)
    o_conf = Counter(round(clean_conf(a.get("opinion_confidence")), 2) for a in aux_rows)
    ops = Counter(str(a.get("opinion", "")).strip().lower() for a in aux_rows)
    logger.info("---- auxiliary data diagnostics (%d rows) ----", n)
    logger.info("converged: %d (%.1f%%) | mean iterations: %.2f",
                conv, 100 * conv / n, sum(iters) / n)
    logger.info("aspect == target: %d (%.1f%%)", same, 100 * same / n)
    logger.info("aspect-conf values (top): %s", a_conf.most_common(6))
    logger.info("opinion-conf values (top): %s", o_conf.most_common(6))
    logger.info("most common opinions: %s", ops.most_common(8))
    logger.info("------------------------------------------------")


# ----------------------------------------------------------------------------
# prompts (Flan-T5 is instruction-tuned: natural questions work better than [SEP])
# ----------------------------------------------------------------------------
def aspect_prompt(sentence: str, target: str) -> str:
    return f'Given the sentence "{sentence}", what is the mentioned aspect towards {target}?'


def opinion_prompt(sentence: str, target: str, aspect: str) -> str:
    return (
        f'Given the sentence "{sentence}", the mentioned aspect towards {target} '
        f'is "{aspect}". What is the underlying opinion towards {target}?'
    )


def polarity_prompt(sentence: str, target: str) -> str:
    return (
        f'Given the sentence "{sentence}", what is the sentiment polarity towards '
        f"{target}? Options: positive, negative, neutral."
    )


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
class MTISADataset(Dataset):
    """Returns python lists; padding happens in the collate function."""

    def __init__(
        self,
        instances: List[Dict],
        tokenizer,
        aux_by_id: Optional[Dict[str, Dict]] = None,
        conf_source: str = "stored",
        max_length: int = 128,
        label_max_length: int = 64,
    ):
        self.instances = instances
        self.tok = tokenizer
        self.aux_by_id = aux_by_id
        self.conf_source = conf_source
        self.max_length = max_length
        self.label_max_length = label_max_length

    def __len__(self):
        return len(self.instances)

    def _enc(self, text: str, max_len: int) -> List[int]:
        return self.tok(text, max_length=max_len, truncation=True).input_ids  # ends with </s>

    def _conf(self, aux: Dict, key: str) -> float:
        if self.conf_source == "none":
            return 1.0
        if self.conf_source == "convergence":
            return convergence_conf(aux)
        return clean_conf(aux.get(key))

    def __getitem__(self, idx):
        inst = self.instances[idx]
        sentence, target = inst["sentence"], inst["target"]
        gold = inst["gold_polarity"].lower()

        item = {
            "instance_id": inst["id"],
            "polarity_input_ids": self._enc(polarity_prompt(sentence, target), self.max_length),
            "polarity_labels": self._enc(gold, 8),
            "polarity_label_id": POLARITY_TO_ID[gold],
        }

        if self.aux_by_id is not None:
            aux = self.aux_by_id[inst["id"]]
            aspect_text = (aux.get("aspect") or "").strip() or target
            opinion_text = (aux.get("opinion") or "").strip() or "none"
            item.update({
                "aspect_input_ids": self._enc(aspect_prompt(sentence, target), self.max_length),
                "aspect_labels": self._enc(aspect_text, self.label_max_length),
                "aspect_confidence": self._conf(aux, "aspect_confidence"),
                "opinion_input_ids": self._enc(
                    opinion_prompt(sentence, target, aspect_text), self.max_length),
                "opinion_labels": self._enc(opinion_text, self.label_max_length),
                "opinion_confidence": self._conf(aux, "opinion_confidence"),
            })
        return item


def _pad(seqs: List[List[int]], value: int) -> torch.Tensor:
    length = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (length - len(s)) for s in seqs], dtype=torch.long)


def make_collate(pad_id: int):
    def collate(batch: List[Dict]):
        out = {
            "instance_id": [b["instance_id"] for b in batch],
            "polarity_label_id": torch.tensor([b["polarity_label_id"] for b in batch]),
        }
        prefixes = ["polarity"]
        if "aspect_input_ids" in batch[0]:
            prefixes += ["aspect", "opinion"]
        for p in prefixes:
            ids = _pad([b[f"{p}_input_ids"] for b in batch], pad_id)
            out[f"{p}_input_ids"] = ids
            out[f"{p}_attention_mask"] = (ids != pad_id).long()
            out[f"{p}_labels"] = _pad([b[f"{p}_labels"] for b in batch], -100)
            if p != "polarity":
                out[f"{p}_confidence"] = torch.tensor(
                    [b[f"{p}_confidence"] for b in batch], dtype=torch.float)
        return out
    return collate


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ----------------------------------------------------------------------------
# one epoch (train or eval) with NaN / Inf guards
# ----------------------------------------------------------------------------
def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scheduler=None,
    grad_accum: int = 1,
    bf16: bool = False,
    max_consecutive_bad: int = 20,
    desc: str = "",
):
    train = optimizer is not None
    model.train(train)
    use_amp = bf16 and device.type == "cuda"

    total_loss, n_ok, n_bad, consec_bad = 0.0, 0, 0, 0
    part_sum = {"aspect": 0.0, "opinion": 0.0, "polarity": 0.0}
    part_cnt = {"aspect": 0, "opinion": 0, "polarity": 0}
    ids, gold, pred = [], [], []

    if train:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=desc, leave=False)
    for i, batch in enumerate(pbar):
        batch = move_batch(batch, device)
        inputs = {k: v for k, v in batch.items()
                  if k not in ("instance_id", "polarity_label_id")}

        with torch.set_grad_enabled(train):
            amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                       if use_amp else contextlib.nullcontext())
            with amp_ctx:
                out = model(**inputs)
            loss = out["combined_loss"].float()

            if not torch.isfinite(loss):
                n_bad += 1
                consec_bad += 1
                logger.warning("Non-finite loss at step %d (%s) - batch skipped", i, desc)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                if consec_bad >= max_consecutive_bad:
                    raise RuntimeError(
                        f"{consec_bad} consecutive non-finite losses - training diverged. "
                        "Try a lower --lr / --awl-lr, or drop --bf16."
                    )
                continue
            consec_bad = 0

            if train:
                (loss / grad_accum).backward()
                if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                        scheduler.step()
                    else:
                        n_bad += 1
                        logger.warning("Non-finite grad norm at step %d - update skipped", i)
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach())
        n_ok += 1
        for k in part_sum:
            v = out.get(f"{k}_loss")
            if v is not None and torch.isfinite(v):
                part_sum[k] += float(v.detach())
                part_cnt[k] += 1

        ids.extend(batch["instance_id"])
        pred.extend(out["polarity_logits"].argmax(dim=-1).detach().cpu().tolist())
        gold.extend(batch["polarity_label_id"].detach().cpu().tolist())
        pbar.set_postfix(loss=f"{total_loss / n_ok:.4f}")

    if gold:
        f1 = float(f1_score(gold, pred, labels=[0, 1, 2], average="macro", zero_division=0))
        acc = float(np.mean(np.array(gold) == np.array(pred)))
    else:
        f1, acc = 0.0, 0.0

    return {
        "loss": total_loss / max(1, n_ok),
        "f1": f1,
        "acc": acc,
        "parts": {k: part_sum[k] / max(1, part_cnt[k]) for k in part_sum},
        "skipped": n_bad,
        "ids": ids,
        "gold": gold,
        "pred": pred,
    }


def label_stats(instances: List[Dict], name: str):
    c = Counter(x["gold_polarity"].lower() for x in instances)
    n = max(1, len(instances))
    logger.info("%s: %d instances | %s | majority-class acc %.3f",
                name, len(instances), dict(c), max(c.values(), default=0) / n)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/processed/restaurants_train_implicit.json")
    ap.add_argument("--aux-data", default="data/auxiliary/restaurants_train_implicit_aux.json")
    ap.add_argument("--val-data", default=None,
                    help="Optional separate validation JSON; otherwise split from train.")
    ap.add_argument("--test-data", default=None,
                    help="Optional test JSON, evaluated once with the best checkpoint.")
    ap.add_argument("--model-name", default="google/flan-t5-base")
    ap.add_argument("--output-dir", default="models/mt_isa_v2")

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = bs * accum")
    ap.add_argument("--eval-batch-size", type=int, default=32,
                    help="batch size for validation/test (no grads, so it can be larger)")
    ap.add_argument("--lr", type=float, default=1e-5, help="backbone LR")
    ap.add_argument("--awl-lr", type=float, default=1e-2, help="LR for T-AWL sigma params")
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--num-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--max-length", type=int, default=128)

    ap.add_argument("--d-awl-strategy", default="input",
                    choices=["input", "output", "input_output", "none"])
    ap.add_argument("--t-awl-version", default="alf2", choices=["alf1", "alf2"])
    ap.add_argument("--freeze-t-awl", action="store_true",
                    help="Keep sigma^2=1 for all tasks (plain equal-weight MTL ablation).")
    ap.add_argument("--conf-source", default="stored",
                    choices=["stored", "convergence", "none"],
                    help="stored = confidences in the aux JSON; convergence = derived from "
                         "converged/refinement_iterations; none = all 1.0")
    ap.add_argument("--drop-nonconverged", action="store_true",
                    help="Remove non-converged aux rows from the TRAIN split only.")

    ap.add_argument("--polarity-only", action="store_true",
                    help="Baseline: train on the polarity task only (no auxiliary tasks).")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-instances", type=int, default=None, help="debug: cap train size")
    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon
    else:
        device = torch.device("cpu")
    if args.bf16 and device.type != "cuda":
        logger.warning("--bf16 is only used on CUDA; running in fp32 on %s.", device)
    logger.info("Device: %s | args: %s", device, vars(args))

    # ---------------- data ----------------
    train_all = load_json(args.train_data)
    aux_rows = load_json(args.aux_data)
    aux_by_id = {a["instance_id"]: a for a in aux_rows}
    describe_aux(aux_rows)

    def keep(x):
        return x.get("is_implicit", True) and x["gold_polarity"].lower() in POLARITY_TO_ID

    pool = [x for x in train_all if keep(x)]
    with_aux = [x for x in pool if x["id"] in aux_by_id]
    logger.info("Train instances: %d | implicit+valid: %d | with aux: %d",
                len(train_all), len(pool), len(with_aux))
    if not with_aux:
        raise ValueError("No training instance has an auxiliary record - check the id fields.")
    if args.max_instances:
        with_aux = with_aux[: args.max_instances]

    if args.val_data:
        val_instances = [x for x in load_json(args.val_data) if keep(x)]
        train_instances = with_aux
    else:
        train_instances, val_instances = make_validation_split(
            with_aux, args.validation_fraction, args.seed)

    if args.drop_nonconverged:
        before = len(train_instances)
        train_instances = [x for x in train_instances
                           if aux_by_id[x["id"]].get("converged", False)]
        logger.info("Dropped non-converged from train: %d -> %d", before, len(train_instances))

    label_stats(train_instances, "train")
    label_stats(val_instances, "val")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate = make_collate(tokenizer.pad_token_id)

    train_ds = MTISADataset(train_instances, tokenizer,
                            None if args.polarity_only else aux_by_id,
                            args.conf_source, args.max_length)
    val_ds = MTISADataset(val_instances, tokenizer, None, args.conf_source, args.max_length)
    loader_kw = dict(collate_fn=collate, num_workers=args.num_workers,
                     pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_kw)

    # sanity print: one decoded example per task + label check
    ex = collate([train_ds[0]])
    for p in (["polarity"] if args.polarity_only else ["aspect", "opinion", "polarity"]):
        lab = ex[f"{p}_labels"][0].clone()
        lab[lab == -100] = tokenizer.pad_token_id
        logger.info("[%s] input : %s", p, tokenizer.decode(ex[f"{p}_input_ids"][0], skip_special_tokens=True))
        logger.info("[%s] target: %s", p, tokenizer.decode(lab, skip_special_tokens=True))
    if not args.polarity_only:
        logger.info("[example confidences] aspect=%.2f opinion=%.2f",
                    float(ex["aspect_confidence"][0]), float(ex["opinion_confidence"][0]))

    # ---------------- model / optimizer ----------------
    model = MTISAModel(
        model_name=args.model_name,
        d_awl_strategy=args.d_awl_strategy,
        t_awl_version=args.t_awl_version,
        tokenizer=tokenizer,
    ).to(device)

    if args.freeze_t_awl or args.polarity_only:
        model.t_awl.log_sigma_sq.requires_grad_(False)

    tawl_ids = {id(p) for p in model.t_awl.parameters()}
    backbone_params = [p for p in model.parameters()
                       if id(p) not in tawl_ids and p.requires_grad]
    tawl_params = [p for p in model.t_awl.parameters() if p.requires_grad]
    groups = [{"params": backbone_params, "lr": args.lr, "weight_decay": 0.01}]
    if tawl_params:
        groups.append({"params": tawl_params, "lr": args.awl_lr, "weight_decay": 0.0})
    optimizer = AdamW(groups)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    warmup = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- epoch 0: zero-shot check of the eval pipeline ----------------
    with torch.no_grad():
        r0 = run_epoch(model, val_loader, device, bf16=args.bf16, desc="val (zero-shot)")
    logger.info("Zero-shot Flan-T5 on val: loss %.4f | F1 %.4f | Acc %.4f",
                r0["loss"], r0["f1"], r0["acc"])

    # ---------------- training loop ----------------
    best_f1, best_epoch, patience_counter = -1.0, 0, 0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        logger.info("=" * 70)
        logger.info("Epoch %d/%d", epoch, args.num_epochs)

        tr = run_epoch(model, train_loader, device, optimizer, scheduler,
                       args.grad_accum, args.bf16, desc=f"train {epoch}")
        with torch.no_grad():
            va = run_epoch(model, val_loader, device, bf16=args.bf16, desc=f"val {epoch}")

        w_norm = model.t_awl.get_task_weights(normalized=True)
        sig = model.t_awl.get_sigma_sq()

        logger.info("Train loss %.4f | F1 %.4f | Acc %.4f | parts %s | skipped %d",
                    tr["loss"], tr["f1"], tr["acc"],
                    {k: round(v, 4) for k, v in tr["parts"].items()}, tr["skipped"])
        logger.info("Val   loss %.4f | F1 %.4f | Acc %.4f", va["loss"], va["f1"], va["acc"])
        logger.info("Task weights (normalised): %s | sigma^2: %s",
                    {k: round(v, 3) for k, v in w_norm.items()},
                    {k: round(v, 3) for k, v in sig.items()})

        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"], "train_f1": tr["f1"], "train_acc": tr["acc"],
            "train_parts": tr["parts"], "train_skipped": tr["skipped"],
            "val_loss": va["loss"], "val_f1": va["f1"], "val_acc": va["acc"],
            "task_weights_normalised": w_norm, "sigma_sq": sig,
        })

        if va["f1"] > best_f1:
            best_f1, best_epoch, patience_counter = va["f1"], epoch, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            logger.info("New best (val macro-F1 %.4f) saved.", best_f1)
        else:
            patience_counter += 1
            logger.info("No improvement: %d/%d", patience_counter, args.patience)
            if patience_counter >= args.patience:
                logger.info("Early stopping.")
                break

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Done. Best val macro-F1 %.4f at epoch %d", best_f1, best_epoch)

    # ---------------- optional test evaluation ----------------
    if args.test_data:
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
        test_instances = [x for x in load_json(args.test_data) if keep(x)]
        label_stats(test_instances, "test")
        test_ds = MTISADataset(test_instances, tokenizer, None, args.conf_source, args.max_length)
        test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_kw)
        with torch.no_grad():
            te = run_epoch(model, test_loader, device, bf16=args.bf16, desc="test")
        logger.info("TEST (best ckpt, epoch %d): Acc %.4f | macro-F1 %.4f",
                    best_epoch, te["acc"], te["f1"])
        logger.info("Confusion matrix (rows=gold, cols=pred; order %s):\n%s",
                    POLARITY_WORDS,
                    confusion_matrix(te["gold"], te["pred"], labels=[0, 1, 2]))
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"acc": te["acc"], "macro_f1": te["f1"], "best_epoch": best_epoch,
                       "val_f1": best_f1, "args": vars(args)}, f, indent=2)
        with open(out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
            json.dump([{"id": i, "gold": ID_TO_POLARITY[g], "pred": ID_TO_POLARITY[p]}
                       for i, g, p in zip(te["ids"], te["gold"], te["pred"])], f, indent=2)


if __name__ == "__main__":
    main()
