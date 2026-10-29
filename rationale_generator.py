"""
Rationale generation for rationale distillation (FACT-SR / RVISA style), via Ollama.

For every row (id, sentence, target, gold_polarity):
  step 1  rationalize (gold visible): one or two sentences (max 40 words) explaining why the
          sentiment toward the target is <gold>, naming the target and using only evidence from
          the sentence, not starting with a label word
  step 2  verify (gold hidden): sentence + target + rationale -> polarity
  keep    only if verify == gold AND the rationale mentions the target (case-insensitive
          substring, or any target token of >= 4 characters)
  up to 3 attempts per row; attempts 2-3 re-run step 1 with temperature 0.7

Output: data/auxiliary/restaurant14_train_all_rationale_qwen7b.json
        list of {instance_id, rationale, passed, attempts, gold_polarity, verify_polarity, fail_reasons}
        (+ a .jsonl checkpoint written as rows finish; --resume skips finished ids)

Usage: python rationale_generator.py [--model qwen2.5:7b] [--workers 1] [--resume] [--max-instances N]
Needs only requests + tqdm (requirements-gen.txt) and a running `ollama serve`.
"""
import argparse
import json
import logging
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from isa_prompts import VALID_POLARITIES, _extract_polarity_word, looks_non_english
from ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rationale_generator")
_write_lock = threading.Lock()

SYSTEM = "You explain sentiment labels in aspect-based sentiment analysis. Respond only in English."
MAX_WORDS = 40


def rationale_prompt(sentence: str, target: str, gold: str) -> str:
    return (
        f'Sentence: "{sentence}"\n'
        f'Target: "{target}"\n'
        f'The sentiment toward "{target}" is {gold}.\n'
        f'In one or two sentences (at most {MAX_WORDS} words), explain why the sentiment toward '
        f'"{target}" is {gold}. Refer to "{target}" by name and use only evidence from the sentence. '
        f"Do not start with the word positive, negative or neutral. Write only the explanation."
    )


def verify_prompt(sentence: str, target: str, rationale: str) -> str:
    return (
        f'Sentence: "{sentence}"\n'
        f'Target: "{target}"\n'
        f'Explanation: "{rationale}"\n'
        f'Based on the sentence and the explanation, what is the sentiment polarity toward "{target}"? '
        f"Answer with exactly one word: positive, negative, or neutral."
    )


def clean_rationale(text: str) -> str:
    t = " ".join((text or "").split())
    t = re.sub(r"^(rationale|explanation|answer)\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip(" \"'")
    sents = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(sents[:2]).strip()


def mentions_target(rationale: str, target: str) -> bool:
    r = rationale.lower()
    if target.lower() in r:
        return True
    return any(tok in r for tok in re.findall(r"\w+", target.lower()) if len(tok) >= 4)


def starts_with_label(rationale: str) -> bool:
    m = re.match(r"\W*(\w+)", rationale.lower())
    return bool(m) and m.group(1) in VALID_POLARITIES


class RationaleGenerator:
    def __init__(self, client: OllamaClient, attempts: int = 3, first_temperature: float = 0.3,
                 retry_temperature: float = 0.7):
        self.client = client
        self.attempts = attempts
        self.t_first = first_temperature
        self.t_retry = retry_temperature

    def _gen(self, prompt: str, temperature: float, num_predict: int) -> str:
        text = self.client.generate(prompt, system=SYSTEM, temperature=temperature, num_predict=num_predict)
        if looks_non_english(text):
            text = self.client.generate(prompt, system=SYSTEM + " Your previous answer was not in English. "
                                        "Answer again, in English only.", temperature=temperature,
                                        num_predict=num_predict)
        return "" if looks_non_english(text) else text

    def run(self, inst: Dict) -> Dict:
        s, t, gold = inst["sentence"], inst["target"], inst["gold_polarity"].strip().lower()
        reasons: List[str] = []
        rationale, verdict = "", ""
        for k in range(1, self.attempts + 1):
            temp = self.t_first if k == 1 else self.t_retry
            rationale = clean_rationale(self._gen(rationale_prompt(s, t, gold), temp, 120))
            if not rationale:
                reasons.append("empty")
                continue
            if starts_with_label(rationale):
                reasons.append("label_start")
                continue
            if len(rationale.split()) > MAX_WORDS + 10:   # small tolerance over the 40-word limit
                reasons.append("too_long")
                continue
            if not mentions_target(rationale, t):
                reasons.append("no_target")
                continue
            verdict = _extract_polarity_word(self._gen(verify_prompt(s, t, rationale), 0.0, 10))
            if verdict != gold:
                reasons.append(f"verify_{verdict or 'unparsable'}")
                continue
            return {"instance_id": inst["id"], "rationale": rationale, "passed": True, "attempts": k,
                    "gold_polarity": gold, "verify_polarity": verdict, "fail_reasons": reasons}
        return {"instance_id": inst["id"], "rationale": rationale, "passed": False,
                "attempts": self.attempts, "gold_polarity": gold, "verify_polarity": verdict,
                "fail_reasons": reasons}


def load_rows(path: Path) -> List[Dict]:
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("skipping malformed checkpoint line")
    return rows


def print_stats(rows: List[Dict]):
    if not rows:
        print("no rows")
        return
    n = len(rows)
    ok = sum(1 for r in rows if r.get("passed"))
    by, ok_by = Counter(r["gold_polarity"] for r in rows), Counter(r["gold_polarity"] for r in rows if r.get("passed"))
    att = Counter(r["attempts"] for r in rows if r.get("passed"))
    why = Counter(x for r in rows for x in r.get("fail_reasons", []))
    print("\n" + "=" * 60 + "\nRATIONALE GENERATION STATISTICS\n" + "=" * 60)
    print(f"rows      : {n}")
    print(f"passed    : {ok} ({100 * ok / n:.1f}%)")
    for l in VALID_POLARITIES:
        print(f"  {l:9s} {ok_by[l]:5d}/{by[l]:5d} = {100 * ok_by[l] / max(1, by[l]):.1f}%")
    print(f"passed at attempt: {dict(sorted(att.items()))}")
    print(f"failed-attempt reasons: {dict(why.most_common())}")
    lens = [len(r["rationale"].split()) for r in rows if r.get("passed")]
    if lens:
        print(f"rationale words (passed): mean {sum(lens) / len(lens):.1f}, max {max(lens)}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/restaurant14_train_all_gen.json")
    ap.add_argument("--output", default="data/auxiliary/restaurant14_train_all_rationale_qwen7b.json")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.3, help="step 1, first attempt")
    ap.add_argument("--retry-temperature", type=float, default=0.7, help="step 1, attempts 2-3")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    ckpt = out.with_suffix(".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.stats_only:
        print_stats(load_rows(ckpt))
        return

    rows = json.load(open(args.input, encoding="utf-8"))
    for r in rows:
        r.setdefault("id", r.get("instance_id"))
    rows = [r for r in rows if r["gold_polarity"].strip().lower() in VALID_POLARITIES]
    if args.max_instances:
        rows = rows[: args.max_instances]
    done = {r["instance_id"] for r in load_rows(ckpt)} if args.resume else set()
    if ckpt.exists() and not args.resume:
        logger.warning("%s exists and --resume was not passed: new rows are appended (duplicates "
                       "possible). Delete it for a clean run or pass --resume.", ckpt)
    todo = [r for r in rows if r["id"] not in done]
    logger.info("rows: %d | already done: %d | to do: %d | model %s | workers %d",
                len(rows), len(done), len(todo), args.model, args.workers)

    if todo:
        client = OllamaClient(model=args.model, base_url=args.base_url, temperature=args.temperature)
        gen = RationaleGenerator(client, args.attempts, args.temperature, args.retry_temperature)
        failed = [0]

        def work(inst):
            try:
                res = gen.run(inst)
            except RuntimeError as e:   # request failed after retries: not checkpointed, --resume retries
                logger.error("[%s] %s", inst["id"], e)
                failed[0] += 1
                return
            with _write_lock:
                with open(ckpt, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")

        if args.workers <= 1:
            for inst in tqdm(todo, desc="rationales"):
                work(inst)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for _ in tqdm(as_completed([ex.submit(work, i) for i in todo]), total=len(todo), desc="rationales"):
                    pass
        if failed[0]:
            logger.warning("%d rows failed (request errors) - re-run with --resume", failed[0])

    # final JSON: last record per id, in input order
    last = {r["instance_id"]: r for r in load_rows(ckpt)}
    final = [last[r["id"]] for r in rows if r["id"] in last]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    logger.info("wrote %d rows to %s (%d of %d input rows done)", len(final), out, len(final), len(rows))
    print_stats(final)


if __name__ == "__main__":
    main()
