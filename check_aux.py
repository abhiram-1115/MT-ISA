"""
Quick checks on the auxiliary data file.
Usage: python check_aux.py [path_to_aux_json]
  1) converged rate per gold polarity (is convergence tied to the label?)
  2) aux labels the T5 tokenizer can't represent (they'd become empty/<unk> targets)
"""
import json
import sys
from collections import Counter

from transformers import AutoTokenizer

aux_path = sys.argv[1] if len(sys.argv) > 1 else "data/auxiliary/restaurants_train_implicit_aux.json"
aux = json.load(open(aux_path, encoding="utf-8"))
print("rows:", len(aux))

# 1) does convergence depend on the gold label?
tot = Counter(a["gold_polarity"].lower() for a in aux)
conv = Counter(a["gold_polarity"].lower() for a in aux if a.get("converged"))
print("\nconverged rate by gold polarity")
for k in ("positive", "negative", "neutral"):
    n = tot.get(k, 0)
    print(f"  {k:9s} {conv.get(k, 0):4d}/{n:4d} = {100 * conv.get(k, 0) / max(1, n):.1f}%")

# 2) labels the T5 tokenizer cannot represent
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
bad = []
for a in aux:
    for field in ("aspect", "opinion"):
        text = (a.get(field) or "").strip()
        if not text:
            continue  # empty -> replaced by target / 'none' during training
        ids = tok(text).input_ids
        if tok.unk_token_id in ids or tok.decode(ids, skip_special_tokens=True).strip() == "":
            bad.append((a["instance_id"], field, text))
print(f"\nlabels containing <unk> or decoding to empty: {len(bad)}")
for b in bad[:15]:
    print("  ", b)
