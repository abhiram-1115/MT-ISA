"""
Compare auxiliary-data files from different generator models side by side.

Usage:
  python compare_aux.py \
    --file mistral7b=data/auxiliary/restaurant14_train_implicit_aux_mistral7b.json \
    --file llama1b=data/auxiliary/restaurant14_train_implicit_aux_llama1b.json \
    --file qwen3b=data/auxiliary/restaurant14_train_implicit_aux_qwen3b.json

Each --file is "name=path" and can be repeated any number of times.
Prints one row per file: overall convergence, convergence broken down by
gold polarity (this is the number that matters most - see the neutral
column), empty-label rate, and mean self-refine iterations.
"""
import argparse
import json
from collections import Counter


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stats(rows):
    total = len(rows)
    by_label = Counter(r["gold_polarity"].lower() for r in rows)
    conv_by_label = Counter(r["gold_polarity"].lower() for r in rows if r.get("converged"))
    converged = sum(conv_by_label.values())
    empty_aspect = sum(1 for r in rows if not (r.get("aspect") or "").strip())
    empty_opinion = sum(1 for r in rows if not (r.get("opinion") or "").strip())
    mean_iters = sum(r.get("refinement_iterations", 0) for r in rows) / max(1, total)
    return {
        "total": total,
        "converged_pct": 100 * converged / max(1, total),
        "pos_pct": 100 * conv_by_label.get("positive", 0) / max(1, by_label.get("positive", 1)),
        "neg_pct": 100 * conv_by_label.get("negative", 0) / max(1, by_label.get("negative", 1)),
        "neu_pct": 100 * conv_by_label.get("neutral", 0) / max(1, by_label.get("neutral", 1)),
        "empty_aspect": empty_aspect,
        "empty_opinion": empty_opinion,
        "mean_iters": mean_iters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", required=True,
                     help="name=path to an auxiliary JSON file; repeatable")
    args = ap.parse_args()

    results = []
    for spec in args.file:
        if "=" not in spec:
            raise SystemExit(f"--file must be name=path, got: {spec}")
        name, path = spec.split("=", 1)
        results.append((name, stats(load(path))))

    hdr = (f"{'model':<16}{'n':<6}{'conv%':<8}{'pos%':<8}{'neg%':<8}{'neu%':<8}"
           f"{'empty_a':<9}{'empty_o':<9}{'iters':<7}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in results:
        print(f"{name:<16}{s['total']:<6}{s['converged_pct']:<8.1f}{s['pos_pct']:<8.1f}"
              f"{s['neg_pct']:<8.1f}{s['neu_pct']:<8.1f}{s['empty_aspect']:<9}"
              f"{s['empty_opinion']:<9}{s['mean_iters']:<7.2f}")
    print()
    print("neu% is convergence on NEUTRAL gold instances - your biggest test class")
    print("and the one that was weakest (17.6%) with the original Mistral/Qwen data.")
    print("Pick the generator(s) with the best conv% / neu% before spending time training on them.")


if __name__ == "__main__":
    main()
