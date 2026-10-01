"""
Inspect what the generated aspect/opinion labels actually contain.

Usage:
  python check_labels.py                       # compares the old file and the qwen7b file
  python check_labels.py name=path name2=path2 # any files you like

Key question: do the opinions carry information beyond the gold polarity word?
"""
import json
import random
import re
import sys
from collections import Counter

DEFAULTS = [
    ("old_mistral_qwen", "data/auxiliary/restaurants_train_implicit_aux.json"),
    ("qwen7b", "data/auxiliary/restaurant14_train_implicit_aux_qwen7b_full.json"),
]
POL = ("positive", "negative", "neutral")


def norm(s):
    return re.sub(r"[^a-z ]", "", (s or "").lower()).strip()


def report(name, path):
    rows = json.load(open(path, encoding="utf-8"))
    n = len(rows)
    op = [norm(r.get("opinion")) for r in rows]
    just_pol = sum(1 for o in op if o in POL)
    has_pol = sum(1 for o in op if any(re.search(rf"\b{p}\b", o) for p in POL))
    same_aspect = sum(1 for r in rows if norm(r.get("aspect")) == norm(r["target"]))
    avg_words = sum(len((r.get("opinion") or "").split()) for r in rows) / n
    distinct = len(set(op))

    print("=" * 70)
    print(f"{name}   ({n} rows)   {path}")
    print("=" * 70)
    print(f"opinion is ONLY a polarity word : {just_pol:4d} ({100*just_pol/n:.1f}%)")
    print(f"opinion mentions a polarity word: {has_pol:4d} ({100*has_pol/n:.1f}%)")
    print(f"aspect == target                : {same_aspect:4d} ({100*same_aspect/n:.1f}%)")
    print(f"average opinion length          : {avg_words:.1f} words")
    print(f"distinct opinions               : {distinct}")
    print("most common opinions:", Counter(op).most_common(8))

    print("\nBy gold polarity (converged rows only): how often the opinion is just a polarity word")
    for g in POL:
        sub = [r for r in rows if r["gold_polarity"].lower() == g and r.get("converged")]
        k = sum(1 for r in sub if norm(r.get("opinion")) in POL)
        print(f"  {g:9s} converged={len(sub):4d}   opinion-is-polarity-word={k:4d} ({100*k/max(1,len(sub)):.1f}%)")

    random.seed(0)
    for g in ("neutral", "negative"):
        sub = [r for r in rows if r["gold_polarity"].lower() == g and r.get("converged")]
        print(f"\n6 random CONVERGED {g} examples (sentence | target | aspect | opinion):")
        for r in random.sample(sub, min(6, len(sub))):
            print(f"  - {r['sentence'][:90]}")
            print(f"      target={r['target']!r}  aspect={r['aspect']!r}  opinion={r['opinion']!r}")
    print()


def main():
    specs = []
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            specs.append((k, v))
    for name, path in specs or DEFAULTS:
        try:
            report(name, path)
        except FileNotFoundError:
            print(f"[skip] {name}: file not found: {path}\n")


if __name__ == "__main__":
    main()
