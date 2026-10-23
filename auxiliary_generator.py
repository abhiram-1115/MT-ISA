"""
Phase 1: Auxiliary Task Construction with Self-Refine (v2)
Implements the LLM-based generation with polarity intervention (Algorithm 1
from the MT-ISA paper), using ollama_client.py.

Fixes vs. the previous version
-------------------------------
* Saves each result to a JSONL checkpoint file AS IT COMPLETES. A crash or
  Ctrl+C loses at most the one in-flight instance, not the whole run.
* --resume skips instance ids already present in that checkpoint, so you
  can stop and continue a long generation run.
* Optional --workers > 1 runs requests concurrently via a thread pool.
  Ollama serializes requests per model by default - set the server-side
  environment variable OLLAMA_NUM_PARALLEL (e.g. 4) before `ollama serve`
  if you want real concurrency; otherwise extra workers just queue.
* At the end, converts the JSONL checkpoint into the JSON array format
  train_mt_isa.py expects. --stats-only reprints statistics for an existing
  checkpoint without generating anything.
* Confidence now comes from the model's own self-rating (see
  ollama_client.py) instead of a length-based formula.
* --backend openai uses an OpenAI-compatible gateway (openai_client.py, e.g. GPT-4o-mini,
  temperature 0.7 as in the paper) with the same prompts; it needs only requests + tqdm
  (requirements-gen.txt). Token usage is tracked, the running cost is logged every 100
  rows, and the run stops cleanly (checkpoint kept, no final JSON) above --max-usd.
"""

import argparse
import json
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from tqdm import tqdm

from isa_prompts import AspectOpinionExtractor  # backend clients are imported lazily in main()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


@dataclass
class AuxiliaryData:
    """Auxiliary task data with confidence scores."""
    instance_id: str
    sentence: str
    target: str
    gold_polarity: str
    aspect: str
    aspect_confidence: float
    opinion: str
    opinion_confidence: float
    refinement_iterations: int
    converged: bool

    def to_dict(self) -> Dict:
        return asdict(self)


class AuxiliaryGenerator:
    """Generate auxiliary task data using self-refine with polarity intervention."""

    def __init__(self, ollama_client, max_refinement_epochs: int = 10):  # OllamaClient or OpenAIClient
        self.extractor = AspectOpinionExtractor(ollama_client)
        self.max_refinement_epochs = max_refinement_epochs

    def generate_for_instance(
        self, instance_id: str, sentence: str, target: str, gold_polarity: str,
        verbose: bool = False,
    ) -> AuxiliaryData:
        feedback: Optional[str] = None
        best_aspect, best_aspect_conf = "", 0.0
        best_opinion, best_opinion_conf = "", 0.0
        converged = False
        epoch_used = 0

        for epoch in range(self.max_refinement_epochs):
            epoch_used = epoch + 1

            aspect, aspect_conf = self.extractor.extract_aspect(sentence, target, feedback)
            if not aspect:
                if verbose:
                    logger.warning("[%s] no aspect at iter %d, skipping", instance_id, epoch_used)
                continue

            opinion, opinion_conf = self.extractor.extract_opinion(sentence, target, aspect, feedback)
            if not opinion:
                if verbose:
                    logger.warning("[%s] no opinion at iter %d, skipping", instance_id, epoch_used)
                continue

            predicted_polarity, _ = self.extractor.infer_polarity(sentence, target, aspect, opinion)
            if not predicted_polarity:
                if verbose:
                    logger.warning("[%s] no polarity at iter %d, skipping", instance_id, epoch_used)
                continue

            best_aspect, best_aspect_conf = aspect, aspect_conf
            best_opinion, best_opinion_conf = opinion, opinion_conf

            if predicted_polarity == gold_polarity.strip().lower():
                converged = True
                if verbose:
                    logger.info("[%s] converged at iter %d", instance_id, epoch_used)
                break

            feedback = self.extractor.generate_feedback(
                sentence, target, aspect, opinion, predicted_polarity, gold_polarity
            )

        return AuxiliaryData(
            instance_id=instance_id, sentence=sentence, target=target,
            gold_polarity=gold_polarity,
            aspect=best_aspect, aspect_confidence=best_aspect_conf,
            opinion=best_opinion, opinion_confidence=best_opinion_conf,
            refinement_iterations=epoch_used, converged=converged,
        )


# --------------------------------------------------------------------------
# checkpointing helpers
# --------------------------------------------------------------------------
def load_done_ids(jsonl_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["instance_id"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping malformed checkpoint line")
    return done


def append_result(jsonl_path: Path, result: AuxiliaryData):
    with _write_lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")


def load_all_rows(jsonl_path: Path) -> List[Dict]:
    rows = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def jsonl_to_json(jsonl_path: Path, json_path: Path):
    rows = load_all_rows(jsonl_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logger.info("Wrote %d rows to %s", len(rows), json_path)


def print_statistics(jsonl_path: Path):
    rows = load_all_rows(jsonl_path)
    if not rows:
        logger.info("No rows in %s yet.", jsonl_path)
        return
    total = len(rows)
    converged = sum(1 for r in rows if r.get("converged"))
    by_label = Counter(r["gold_polarity"].lower() for r in rows)
    conv_by_label = Counter(r["gold_polarity"].lower() for r in rows if r.get("converged"))
    empty_aspect = sum(1 for r in rows if not (r.get("aspect") or "").strip())
    empty_opinion = sum(1 for r in rows if not (r.get("opinion") or "").strip())
    iters = [r.get("refinement_iterations", 0) for r in rows]

    print("\n" + "=" * 60)
    print("AUXILIARY DATA GENERATION STATISTICS")
    print("=" * 60)
    print(f"Total instances : {total}")
    print(f"Converged       : {converged} ({100 * converged / total:.1f}%)")
    print("Converged by gold polarity:")
    for label in ("positive", "negative", "neutral"):
        n = by_label.get(label, 0)
        c = conv_by_label.get(label, 0)
        print(f"  {label:9s} {c:4d}/{n:4d} = {100 * c / max(1, n):.1f}%")
    print(f"Empty aspect    : {empty_aspect} ({100 * empty_aspect / total:.1f}%)")
    print(f"Empty opinion   : {empty_opinion} ({100 * empty_opinion / total:.1f}%)")
    print(f"Mean iterations : {sum(iters) / total:.2f}")
    print("=" * 60)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate auxiliary data using an LLM (v2)")
    parser.add_argument("--input", default="data/processed/restaurant14_train_implicit.json")
    parser.add_argument("--output", default="data/auxiliary/restaurant14_train_implicit_aux.json",
                         help="final JSON array path (what train_mt_isa.py reads)")
    parser.add_argument("--output-jsonl", default=None,
                         help="checkpoint file; default is --output with a .jsonl extension")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                         help="ollama (local, default) or openai (OpenAI-compatible gateway; URL/key "
                              "from AI_GATEWAY_URL / OPENAI_API_KEY_4O_MINI)")
    parser.add_argument("--model", default=None,
                         help="model; default 'mistral' (ollama) or 'azure/gpt-4o-mini' (openai)")
    parser.add_argument("--temperature", type=float, default=None,
                         help="default: ollama client default (0.3); openai 0.7 (paper)")
    parser.add_argument("--max-usd", type=float, default=10.0,
                         help="openai: stop cleanly once the estimated cost exceeds this")
    parser.add_argument("--price-in", type=float, default=0.15, help="USD per 1M input tokens")
    parser.add_argument("--price-out", type=float, default=0.60, help="USD per 1M output tokens")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--max-epochs", type=int, default=10, help="max self-refine iterations per instance")
    parser.add_argument("--max-instances", type=int, default=None, help="cap instances (for a quick test)")
    parser.add_argument("--workers", type=int, default=1,
                         help="concurrent requests; see the OLLAMA_NUM_PARALLEL note above "
                              "(openai backend: 4 recommended)")
    parser.add_argument("--resume", action="store_true",
                         help="skip instance ids already present in the checkpoint file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stats-only", action="store_true",
                         help="just print statistics for an existing checkpoint and exit")
    args = parser.parse_args()

    out_path = Path(args.output)
    jsonl_path = Path(args.output_jsonl) if args.output_jsonl else out_path.with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stats_only:
        print_statistics(jsonl_path)
        return

    with open(args.input, "r", encoding="utf-8") as f:
        all_instances = json.load(f)
    if args.max_instances:
        all_instances = all_instances[: args.max_instances]

    done_ids = load_done_ids(jsonl_path) if args.resume else set()
    if done_ids:
        logger.info("Resuming: %d instances already in %s, skipping them", len(done_ids), jsonl_path)
    elif jsonl_path.exists():
        logger.warning(
            "%s already exists and --resume was not passed - it will be REUSED "
            "for any matching ids, but new rows are appended, which can create "
            "duplicates. Delete it first for a clean run, or pass --resume.",
            jsonl_path,
        )

    if all_instances and "id" not in all_instances[0]:
        raise SystemExit(f"{args.input} rows have no 'id' field (an *_all.json file?). "
                         "Build the input first: python make_gen_input.py")
    todo = [x for x in all_instances if x["id"] not in done_ids]
    logger.info("Instances to generate: %d (of %d total)", len(todo), len(all_instances))
    if not todo:
        logger.info("Nothing to do.")
        jsonl_to_json(jsonl_path, out_path)
        print_statistics(jsonl_path)
        return

    if args.backend == "openai":
        from openai_client import OpenAIClient
        args.model = args.model or "azure/gpt-4o-mini"
        temperature = 0.7 if args.temperature is None else args.temperature
        logger.info("OpenAI-compatible backend: model %s, temperature %s, workers %d, budget $%.2f",
                    args.model, temperature, args.workers, args.max_usd)
        client = OpenAIClient(model=args.model, temperature=temperature)
    else:
        from ollama_client import OllamaClient
        args.model = args.model or "mistral"
        logger.info("Connecting to Ollama model: %s", args.model)
        kw = {} if args.temperature is None else {"temperature": args.temperature}
        client = OllamaClient(model=args.model, base_url=args.base_url, **kw)
    generator = AuxiliaryGenerator(client, max_refinement_epochs=args.max_epochs)
    is_openai = args.backend == "openai"
    stop = threading.Event()
    state = {"done": 0, "failed": 0}
    state_lock = threading.Lock()

    def cost() -> float:
        return client.cost_usd(args.price_in, args.price_out) if is_openai else 0.0

    def _work(inst: Dict) -> Optional[AuxiliaryData]:
        if stop.is_set():
            return None
        try:
            result = generator.generate_for_instance(
                instance_id=inst["id"], sentence=inst["sentence"],
                target=inst["target"], gold_polarity=inst["gold_polarity"],
                verbose=args.verbose,
            )
        except RuntimeError as e:
            if not is_openai:
                raise              # ollama: unchanged behaviour
            # openai: not written to the checkpoint, so --resume retries this row later
            logger.error("[%s] failed after retries: %s", inst["id"], e)
            with state_lock:
                state["failed"] += 1
            return None
        append_result(jsonl_path, result)
        return result

    def _after_row():
        if not is_openai:
            return
        with state_lock:
            state["done"] += 1
            n = state["done"]
        c = cost()
        if n % 100 == 0:
            u = client.usage()
            logger.info("%d rows | tokens in %d out %d (%d requests, %d with estimated usage) | "
                        "cost $%.3f of $%.2f", n, u["prompt_tokens"], u["completion_tokens"],
                        u["requests"], u["estimated_requests"], c, args.max_usd)
        if c > args.max_usd and not stop.is_set():
            stop.set()
            logger.warning("Estimated cost $%.3f exceeds --max-usd %.2f: stopping (in-flight rows "
                           "finish; checkpoint kept). Re-run with --resume and a higher --max-usd.",
                           c, args.max_usd)

    if args.workers <= 1:
        for inst in tqdm(todo, desc=f"Generating ({args.model})"):
            if stop.is_set():
                break
            _work(inst)
            _after_row()
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_work, inst) for inst in todo]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Generating ({args.model})"):
                if stop.is_set():
                    continue          # pending rows return immediately once stop is set
                _after_row()

    if is_openai:
        u = client.usage()
        print(f"\nTOKENS: input {u['prompt_tokens']:,} | output {u['completion_tokens']:,} | "
              f"requests {u['requests']:,} ({u['estimated_requests']:,} with usage estimated from "
              f"characters) | estimated cost ${cost():.3f} "
              f"(${args.price_in}/1M in, ${args.price_out}/1M out)")
        if state["failed"]:
            print(f"{state['failed']} rows failed after retries and are NOT in the checkpoint - "
                  "re-run with --resume to retry them.")
        if stop.is_set() or state["failed"]:
            print(f"Run incomplete: final JSON NOT written. Checkpoint: {jsonl_path}")
            print_statistics(jsonl_path)
            return

    jsonl_to_json(jsonl_path, out_path)
    print_statistics(jsonl_path)


if __name__ == "__main__":
    main()
