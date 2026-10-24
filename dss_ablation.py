"""Distilling Step-by-Step explanation ablation for SemEval-2014 ABSA.

The script intentionally owns its data, generation, training, and evaluation
artifacts so the existing MT-ISA implementation is unaffected.
"""

import argparse
import ast
import contextlib
import gc
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("dss_ablation")
LABELS = ("positive", "negative", "neutral")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class _RemoteSplit:
    """Small Dataset-like adapter used only when pyarrow is unavailable."""

    def __init__(self, rows: List[Dict], features: List[Dict]):
        self._rows = rows
        self.column_names = [feature["name"] for feature in features]
        self.features = {feature["name"]: feature for feature in features}

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, index):
        return self._rows[index]


def _load_remote_split(dataset_name: str, split_name: str, row_limit: Optional[int] = None) -> _RemoteSplit:
    import requests

    params = {"dataset": dataset_name, "config": "default", "split": split_name}
    first = requests.get(
        "https://datasets-server.huggingface.co/first-rows", params=params, timeout=60
    )
    first.raise_for_status()
    first_data = first.json()
    size = requests.get(
        "https://datasets-server.huggingface.co/size",
        params={"dataset": dataset_name, "config": "default"},
        timeout=60,
    )
    size.raise_for_status()
    row_count = next(item["num_rows"] for item in size.json()["size"]["splits"] if item["split"] == split_name)
    wanted = min(row_count, row_limit) if row_limit is not None else row_count
    rows = []
    if wanted <= 100 and first_data.get("rows"):
        rows = [item["row"] for item in first_data["rows"][:wanted]]
    else:
        for offset in range(0, wanted, 100):
            request_params = {**params, "offset": offset, "length": min(100, wanted - offset)}
            response = None
            for attempt in range(3):
                response = requests.get(
                    "https://datasets-server.huggingface.co/rows",
                    params=request_params,
                    timeout=60,
                )
                if response.status_code != 429:
                    break
                time.sleep(2 ** attempt)
            response.raise_for_status()
            rows.extend(item["row"] for item in response.json()["rows"])
    return _RemoteSplit(rows, first_data["features"])


def load_dataset_splits(dataset_name: str, train_limit: Optional[int] = None, test_limit: Optional[int] = None):
    """Load the named HF dataset and return its train/test splits."""
    try:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name)
        if "train" not in dataset or "test" not in dataset:
            raise ValueError(f"Expected train and test splits; found {list(dataset)}")
        return dataset["train"], dataset["test"]
    except Exception as exc:
        if "pyarrow" not in str(exc).lower() and "dll" not in str(exc).lower():
            raise
        LOGGER.warning("HF datasets loader unavailable (%s); using dataset-server fallback", exc)
        return (
            _load_remote_split(dataset_name, "train", train_limit),
            _load_remote_split(dataset_name, "test", test_limit),
        )


def inspect_dataset(dataset_name: str) -> None:
    train, test = load_dataset_splits(dataset_name)
    print("train columns:", train.column_names)
    print("train features:", train.features)
    print("train first row:", train[0])
    print("test columns:", test.column_names)
    print("test features:", test.features)


def label_names(split, label_col: str) -> Optional[List[str]]:
    feature = split.features.get(label_col)
    names = getattr(feature, "names", None)
    return list(names) if names else None


def normalize_label(value, names: Optional[Sequence[str]]) -> Optional[str]:
    if names is not None and isinstance(value, (int, np.integer)):
        if 0 <= int(value) < len(names):
            value = names[int(value)]
        else:
            return None
    label = str(value).strip().lower()
    return label if label in LABELS else None


def parse_absa_output(value) -> List[Tuple[str, str]]:
    """Parse Nahin24's serialized list of [aspect, polarity] pairs."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(value, (list, tuple)):
        return []
    pairs = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
    return pairs


def normalize_examples(split, text_col: str, aspect_col: str, label_col: str) -> Tuple[List[Dict], int]:
    # Nahin24/semeval-2014 stores one review in input and its aspect/polarity
    # pairs as a serialized list in output, rather than separate columns.
    is_nahin24 = (
        text_col not in split.column_names
        and aspect_col not in split.column_names
        and "input" in split.column_names
        and "output" in split.column_names
    )
    if is_nahin24:
        examples = []
        dropped = 0
        for row_index, row in enumerate(split):
            review_values = row.get("input") or [""]
            review = str(review_values[0] if isinstance(review_values, list) else review_values)
            for aspect_index, (aspect, polarity) in enumerate(parse_absa_output(row.get("output"))):
                label = normalize_label(polarity, None)
                if label is None:
                    dropped += 1
                    continue
                examples.append({
                    "id": f"{row_index}_{aspect_index}",
                    "text": review,
                    "aspect": aspect,
                    "label": label,
                })
        return examples, dropped
    names = label_names(split, label_col)
    examples = []
    dropped = 0
    for index, row in enumerate(split):
        label = normalize_label(row.get(label_col), names)
        if label is None:
            dropped += 1
            continue
        if text_col not in row or aspect_col not in row:
            raise KeyError(
                f"Missing required column(s): {text_col!r}, {aspect_col!r}. "
                f"Available columns: {split.column_names}"
            )
        example_id = row.get("id", row.get("idx", index))
        examples.append({
            "id": str(example_id),
            "text": str(row[text_col]),
            "aspect": str(row[aspect_col]),
            "label": label,
        })
    return examples, dropped


def read_explanations(path: str) -> Dict[str, str]:
    explanations = {}
    source = Path(path)
    if not source.exists():
        return explanations
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if item.get("id") is not None and item.get("explanation"):
                    explanations[str(item["id"])] = str(item["explanation"]).strip()
            except json.JSONDecodeError:
                LOGGER.warning("Skipping invalid cache line %d in %s", line_number, path)
    return explanations


def base_input(example: Mapping[str, str]) -> str:
    return f"Review: {example['text']} Aspect: {example['aspect']} Sentiment toward the aspect?"


def explanation_prompt(example: Mapping[str, str]) -> str:
    return (
        f"Review: {example['text']}\n"
        f"Aspect: {example['aspect']}\n"
        f"Gold label: {example['label']}\n\n"
        f"In one or two sentences, explain why the sentiment toward this aspect is "
        f"{example['label']}, pointing to the specific words or details in the review. "
        "Answer in English only."
    )


def generate_explanations(
    examples: Sequence[Mapping[str, str]],
    cache_path: str,
    model: str,
    fake: bool = False,
) -> Dict[str, str]:
    explanations = read_explanations(cache_path)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    client = None
    if not fake:
        from ollama_client import OllamaClient
        client = OllamaClient(model=model, temperature=0.3, num_predict=80, max_retries=3)
    with Path(cache_path).open("a", encoding="utf-8", buffering=1) as handle:
        for example in examples:
            example_id = str(example["id"])
            if example_id in explanations:
                continue
            explanation = ""
            for attempt in range(1, 4):
                if fake:
                    explanation = f"Fake explanation for the {example['label']} sentiment."
                else:
                    try:
                        explanation = client.generate(
                            explanation_prompt(example),
                            system="Respond only in English.",
                            temperature=0.3,
                            num_predict=80,
                        ).strip()
                    except RuntimeError as exc:
                        LOGGER.warning("Generation failed for %s (attempt %d/3): %s", example_id, attempt, exc)
                        explanation = ""
                if explanation and not CJK_RE.search(explanation):
                    break
                LOGGER.warning("Invalid explanation for %s (attempt %d/3)", example_id, attempt)
                explanation = ""
            if not explanation:
                LOGGER.warning("Skipping %s after three invalid explanation attempts", example_id)
                continue
            item = {"id": example_id, "explanation": explanation}
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
            handle.flush()
            explanations[example_id] = explanation
    return explanations


def make_donor_map(examples: Sequence[Mapping[str, str]], mode: str, seed: int) -> Dict[str, str]:
    """Assign a deranged donor id, grouped by label for same_label mode."""
    if mode not in {"random", "same_label"}:
        raise ValueError(f"Unknown shuffle mode: {mode}")
    rng = random.Random(seed)
    groups = [list(examples)]
    if mode == "same_label":
        by_label = defaultdict(list)
        for example in examples:
            by_label[example["label"]].append(example)
        groups = list(by_label.values())
    donors = {}
    for group in groups:
        ids = [str(example["id"]) for example in group]
        if len(ids) <= 1:
            if ids:
                LOGGER.warning("Cannot derange a one-item donor group; using itself: %s", ids[0])
                donors[ids[0]] = ids[0]
            continue
        order = ids[:]
        rng.shuffle(order)
        for index, receiver_id in enumerate(order):
            donors[receiver_id] = order[(index + 1) % len(order)]
    return donors


def build_arm_records(
    examples: Sequence[Mapping[str, str]],
    explanations: Mapping[str, str],
    arm: str,
    shuffle_mode: str = "same_label",
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Build one arm; arms 2 and 3 share the explanation-eligible subset."""
    if arm not in {"answer_only", "correct", "shuffled"}:
        raise ValueError(f"Unknown arm: {arm}")
    eligible = [example for example in examples if str(example["id"]) in explanations]
    records = []
    for example in eligible:
        records.append({
            "id": str(example["id"]),
            "task": "label",
            "input": f"[label] {base_input(example)}",
            "target": example["label"],
        })
    if arm == "answer_only":
        return records
    donor_map = make_donor_map(eligible, shuffle_mode, seed)
    for example in eligible:
        receiver_id = str(example["id"])
        donor_id = receiver_id if arm == "correct" else donor_map[receiver_id]
        records.append({
            "id": receiver_id,
            "task": "rationale",
            "input": f"[rationale] {base_input(example)}",
            "target": explanations[donor_id],
        })
    return records


def _tokenize_records(records: Sequence[Mapping[str, str]], tokenizer, max_in: int, max_out: int):
    tokenized = []
    for record in records:
        encoded = tokenizer(record["input"], max_length=max_in, truncation=True, padding=False)
        target = tokenizer(text_target=record["target"], max_length=max_out, truncation=True, padding=False)
        tokenized.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": target["input_ids"],
        })
    return tokenized


def collate_batch(batch: Sequence[Mapping[str, Sequence[int]]], pad_token_id: int) -> Dict[str, "torch.Tensor"]:
    import torch

    def pad(key: str, value: int):
        width = max(len(item[key]) for item in batch)
        result = torch.full((len(batch), width), value, dtype=torch.long)
        for row_index, item in enumerate(batch):
            values = torch.tensor(item[key], dtype=torch.long)
            result[row_index, : len(values)] = values
        return result

    return {
        "input_ids": pad("input_ids", pad_token_id),
        "attention_mask": pad("attention_mask", 0),
        "labels": pad("labels", -100),
    }


def macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    """Unweighted F1 over the three sentiment labels."""
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    scores = []
    for label in LABELS:
        true_positive = sum(actual == label and guess == label for actual, guess in zip(gold, predicted))
        support = sum(actual == label for actual in gold)
        predicted_count = sum(guess == label for guess in predicted)
        if support == 0 or predicted_count == 0:
            scores.append(0.0)
            continue
        precision = true_positive / predicted_count
        recall = true_positive / support
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _evaluate(model, tokenizer, examples: Sequence[Mapping[str, str]], max_test: Optional[int]) -> Tuple[float, float, int, List[Dict]]:
    import torch

    if max_test is not None:
        examples = list(examples)[:max_test]
    predictions = []
    model.eval()
    for start in range(0, len(examples), 16):
        batch = list(examples[start:start + 16])
        inputs = tokenizer(
            [f"[label] {base_input(example)}" for example in batch],
            max_length=160,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                num_beams=1,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for example, raw in zip(batch, decoded):
            match = re.search(r"\b(positive|negative|neutral)\b", raw.lower())
            predicted = match.group(1) if match else None
            predictions.append({
                "id": str(example["id"]),
                "gold": example["label"],
                "prediction": predicted,
                "raw": raw,
            })
    gold = [item["gold"] for item in predictions]
    parsed = [item["prediction"] or "__unparsed__" for item in predictions]
    unparsed = sum(item["prediction"] is None for item in predictions)
    acc = sum(actual == guess for actual, guess in zip(gold, parsed)) / len(gold) if gold else 0.0
    return float(acc), float(macro_f1(gold, parsed)), int(unparsed), predictions


def train_one(
    records: Sequence[Mapping[str, str]],
    test_examples: Sequence[Mapping[str, str]],
    args,
    arm: str,
    seed: int,
) -> Dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenized = _tokenize_records(records, tokenizer, args.max_in, args.max_out)
    use_bf16 = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    batches_per_epoch = (len(tokenized) + args.bs - 1) // args.bs
    updates_per_epoch = (batches_per_epoch + args.accum - 1) // args.accum
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / total_updates)
    )
    start = time.perf_counter()
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        order = list(range(len(tokenized)))
        random.shuffle(order)
        model.train()
        for batch_start in range(0, len(order), args.bs):
            batch = [tokenized[index] for index in order[batch_start:batch_start + args.bs]]
            tensors = collate_batch(batch, tokenizer.pad_token_id)
            tensors = {key: value.to(device) for key, value in tensors.items()}
            autocast_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_bf16 else contextlib.nullcontext()
            )
            with autocast_context:
                loss = model(**tensors).loss
            raw_loss = float(loss.detach().float())
            (loss / args.accum).backward()
            is_last_batch = batch_start + args.bs >= len(order)
            if ((batch_start // args.bs + 1) % args.accum == 0) or is_last_batch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps % 50 == 0:
                    LOGGER.info("%s seed=%d epoch=%d optimizer_step=%d loss=%.4f", arm, seed, epoch + 1, optimizer_steps, raw_loss)
    elapsed = time.perf_counter() - start
    acc, macro_f1, unparsed, predictions = _evaluate(model, tokenizer, test_examples, args.max_test)
    peak_mb = None
    if torch.cuda.is_available():
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    result = {
        "arm": arm,
        "shuffle_mode": args.shuffle_mode,
        "seed": seed,
        "acc": acc,
        "macro_f1": macro_f1,
        "unparsed": unparsed,
        "predictions": predictions,
        "wall_seconds": round(elapsed, 3),
        "peak_gpu_mb": peak_mb,
        "n_train": len(records),
        "fake_explanations": bool(args.fake_explanations),
    }
    del model, tokenizer, tokenized, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize(path: str) -> None:
    rows = []
    if Path(path).exists():
        with Path(path).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    if any(row.get("fake_explanations", False) for row in rows):
        print("WARNING: results include fake explanations; do not interpret them as real DSS results.")
    print("arm          shuffle      n seeds   accuracy (%)        macro-F1 (%)")
    print("-" * 72)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["shuffle_mode"])].append(row)
    for (arm, mode), group in sorted(groups.items()):
        acc = [100 * row["acc"] for row in group]
        f1 = [100 * row["macro_f1"] for row in group]
        acc_sd = stdev(acc) if len(acc) > 1 else 0.0
        f1_sd = stdev(f1) if len(f1) > 1 else 0.0
        print(f"{arm:<12} {mode:<12} {len(group):>7}   {mean(acc):>6.2f} +/- {acc_sd:<6.2f}   {mean(f1):>6.2f} +/- {f1_sd:<6.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "train", "summarize"))
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--dataset", default="Nahin24/semeval-2014")
    parser.add_argument("--text_col", default="sentence")
    parser.add_argument("--aspect_col", default="aspect")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--cache", default="explanations.jsonl")
    parser.add_argument("--results", default="dss_results.jsonl")
    parser.add_argument("--ollama_model", "--model", dest="ollama_model", default="qwen2.5:7b")
    parser.add_argument("--fake_explanations", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backbone", default="google/flan-t5-base")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", choices=("answer_only", "correct", "shuffled"), default=["answer_only", "correct", "shuffled"])
    parser.add_argument("--shuffle_mode", choices=("random", "same_label"), default="same_label")
    parser.add_argument("--max_train", type=int)
    parser.add_argument("--max_test", type=int)
    parser.add_argument("--max_in", type=int, default=160)
    parser.add_argument("--max_out", type=int, default=80)
    parser.add_argument("--output_dir", default="dss_checkpoints")
    return parser


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.inspect:
        inspect_dataset(args.dataset)
        return
    if not args.stage:
        raise SystemExit("Choose --stage generate|train|summarize, or use --inspect.")
    if args.stage == "summarize":
        summarize(args.results)
        return
    limits = (
        (args.limit, 1)
        if args.stage == "generate"
        else (args.max_train, args.max_test)
    )
    train_split, test_split = load_dataset_splits(args.dataset, *limits)
    train_examples, dropped_train = normalize_examples(train_split, args.text_col, args.aspect_col, args.label_col)
    test_examples, dropped_test = normalize_examples(test_split, args.text_col, args.aspect_col, args.label_col)
    print(f"Dropped labels: train={dropped_train}, test={dropped_test}")
    if args.stage == "generate":
        examples = train_examples[:args.limit] if args.limit is not None else train_examples
        generate_explanations(examples, args.cache, args.ollama_model, args.fake_explanations)
        return
    explanations = read_explanations(args.cache)
    eligible = [example for example in train_examples if example["id"] in explanations]
    if args.max_train is not None:
        eligible = eligible[:args.max_train]
    if not eligible:
        raise SystemExit(f"No cached explanations found in {args.cache}; run --stage generate first.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for arm in args.arms:
            records = build_arm_records(eligible, explanations, arm, args.shuffle_mode, seed)
            result = train_one(records, test_examples, args, arm, seed)
            with Path(args.results).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                handle.flush()
            print(json.dumps({key: result[key] for key in ("arm", "seed", "acc", "macro_f1", "unparsed", "wall_seconds", "peak_gpu_mb")}))


if __name__ == "__main__":
    main()
