"""
Aggregate MT-ISA experiment results.

Usage: python summarize.py [models_dir] [results_dir]
Reads  <models_dir>/exp_<name>_lr<lr>_s<seed>/test_metrics.json
Writes <results_dir>/summary.txt and <results_dir>/summary.csv
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

models_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(results_dir, exist_ok=True)

rows = []
for f in sorted(glob.glob(os.path.join(models_dir, "exp_*", "test_metrics.json"))):
    d = os.path.basename(os.path.dirname(f))
    m = re.match(r"exp_(.+)_lr([^_]+)_s(\d+)$", d)
    if not m:
        continue
    j = json.load(open(f))
    rows.append({
        "name": m.group(1), "lr": m.group(2), "seed": int(m.group(3)),
        "test_acc": j["acc"], "test_f1": j["macro_f1"],
        "val_f1": j["val_f1"], "best_epoch": j["best_epoch"],
    })

lines = []
lines.append("PER RUN")
lines.append(f"{'name':<16}{'lr':<8}{'seed':<6}{'test_acc':<10}{'test_F1':<10}{'val_F1':<10}{'best_ep':<8}")
for r in sorted(rows, key=lambda r: (r["name"], r["lr"], r["seed"])):
    lines.append(f"{r['name']:<16}{r['lr']:<8}{r['seed']:<6}{r['test_acc']:<10.4f}"
                 f"{r['test_f1']:<10.4f}{r['val_f1']:<10.4f}{r['best_epoch']:<8}")

groups = defaultdict(list)
for r in rows:
    groups[(r["name"], r["lr"])].append(r)


def ms(vals):
    if len(vals) < 2:
        return f"{vals[0]:.4f}"
    return f"{st.mean(vals):.4f} +/- {st.stdev(vals):.4f}"


lines.append("")
lines.append("GROUPED (mean +/- std over seeds; n = number of seeds)")
lines.append(f"{'name':<16}{'lr':<8}{'n':<4}{'test_acc':<20}{'test_F1':<20}")
for (name, lr), rs in sorted(groups.items()):
    lines.append(f"{name:<16}{lr:<8}{len(rs):<4}{ms([r['test_acc'] for r in rs]):<20}"
                 f"{ms([r['test_f1'] for r in rs]):<20}")

text = "\n".join(lines)
print(text)
with open(os.path.join(results_dir, "summary.txt"), "w") as f:
    f.write(text + "\n")
with open(os.path.join(results_dir, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
    w.writeheader()
    w.writerows(rows)
