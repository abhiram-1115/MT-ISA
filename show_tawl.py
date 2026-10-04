"""
Print T-AWL task weights per epoch from history.json.

Usage: python show_tawl.py [run_dir_or_glob ...]
Default: models/exp_*  (each dir must contain history.json)
Example: python show_tawl.py models/exp_mtl_tawl_lr1e-5_s1 "models/exp_in_stored_*"
"""
import glob
import json
import os
import sys

pats = sys.argv[1:] or ["models/exp_*"]
dirs = []
for p in pats:
    dirs += [p] if os.path.isdir(p) else sorted(glob.glob(p))

for d in dirs:
    f = os.path.join(d, "history.json")
    if not os.path.isfile(f):
        continue
    hist = json.load(open(f, encoding="utf-8"))
    print("\n" + os.path.basename(os.path.normpath(d)))
    print(f"{'ep':<4}{'w_aspect':<10}{'w_opinion':<11}{'w_polarity':<12}"
          f"{'sig2_asp':<10}{'sig2_opn':<10}{'sig2_pol':<10}{'val_F1':<8}")
    for h in hist:
        w, s = h["task_weights_normalised"], h["sigma_sq"]
        print(f"{h['epoch']:<4}{w['aspect']:<10.3f}{w['opinion']:<11.3f}{w['polarity']:<12.3f}"
              f"{s['aspect']:<10.3f}{s['opinion']:<10.3f}{s['polarity']:<10.3f}{h['val_f1']:<8.4f}")
