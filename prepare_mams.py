"""
Prepare MAMS-ATSA (Jiang et al., 2019) as an extra polarity training set.

1. Get the raw XML (train.xml, val.xml) of github.com/siat-nlp/MAMS-for-ABSA
   (data/MAMS-ATSA/raw/) into data/external/MAMS-for-ABSA/ - downloaded from
   raw.githubusercontent.com, or `git clone` as a fallback.
2. One row per aspectTerm:
     {id, instance_id: mams_<split>_<6-digit n>, sentence_id: mams_<split>_s<sentence index>,
      sentence, target, gold_polarity, from, to, dataset: "mams", split}
   No "is_implicit" key (train_mt_isa.py keep() treats a missing key as True, which --val-data needs).
3. Leakage filter: drop every MAMS sentence (with all its aspects) whose normalised text
   (lowercase, characters outside [a-z0-9 ] -> space, spaces collapsed) equals, or shares any
   8-word n-gram with, a sentence of data/processed/restaurant14_test_all.json or
   restaurant14_train_all.json.
4. Write data/external/mams_train.json and data/external/mams_val.json; print class counts.

Usage: python prepare_mams.py
Needs only the standard library (+ git for the fallback).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

REPO = "https://github.com/siat-nlp/MAMS-for-ABSA"
RAW = "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/{branch}/data/MAMS-ATSA/raw/{name}"
LOCAL = "data/external/MAMS-for-ABSA"
RAW_DIR = os.path.join(LOCAL, "data", "MAMS-ATSA", "raw")
LABELS = ("positive", "negative", "neutral")
NGRAM = 8


def fetch_raw():
    names = ("train.xml", "val.xml")
    if all(os.path.exists(os.path.join(RAW_DIR, n)) for n in names):
        print(f"MAMS raw XML found in {RAW_DIR}")
        return
    os.makedirs(RAW_DIR, exist_ok=True)
    try:
        for n in names:
            dst = os.path.join(RAW_DIR, n)
            if os.path.exists(dst):
                continue
            last = None
            for branch in ("master", "main"):
                try:
                    url = RAW.format(branch=branch, name=n)
                    with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
                        f.write(r.read())
                    print(f"downloaded {url}")
                    break
                except Exception as e:  # try the next branch name
                    last = e
                    if os.path.exists(dst):
                        os.remove(dst)
            else:
                raise RuntimeError(f"download of {n} failed: {last}")
        return
    except Exception as e:
        print(f"download failed ({e}); trying git clone {REPO}")
    tmp = LOCAL + "_clone"
    subprocess.run(["git", "clone", "--depth", "1", REPO, tmp], check=True)
    src = os.path.join(tmp, "data", "MAMS-ATSA", "raw")
    for n in names:
        with open(os.path.join(src, n), "rb") as fi, open(os.path.join(RAW_DIR, n), "wb") as fo:
            fo.write(fi.read())
    print(f"copied raw XML from the clone in {tmp}")


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def ngrams(text: str, n: int = NGRAM):
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def reference(path):
    rows = json.load(open(path, encoding="utf-8"))
    sents = {norm(r["sentence"]) for r in rows}
    grams = set()
    for s in sents:
        grams |= ngrams(s)
    return sents, grams


def parse(split: str):
    root = ET.parse(os.path.join(RAW_DIR, f"{split}.xml")).getroot()
    sentences = []
    for si, sent in enumerate(root.iter("sentence")):
        text = (sent.findtext("text") or "").strip()
        aspects = []
        for a in sent.iter("aspectTerm"):
            pol = (a.get("polarity") or "").strip().lower()
            if pol in LABELS:
                aspects.append({"term": a.get("term", ""), "polarity": pol,
                                "from": a.get("from"), "to": a.get("to")})
        sentences.append((si, text, aspects))
    return sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/external")
    ap.add_argument("--test-all", default="data/processed/restaurant14_test_all.json")
    ap.add_argument("--train-all", default="data/processed/restaurant14_train_all.json")
    args = ap.parse_args()

    fetch_raw()
    refs = {"test": reference(args.test_all), "train": reference(args.train_all)}
    os.makedirs(args.out_dir, exist_ok=True)

    for split in ("train", "val"):
        sentences = parse(split)
        rows, removed = [], Counter()
        n_aspects = sum(len(a) for _, _, a in sentences)
        for si, text, aspects in sentences:
            ns = norm(text)
            grams = ngrams(ns)
            hit = None
            for name in ("test", "train"):          # test leakage is reported first
                ref_s, ref_g = refs[name]
                if ns in ref_s or (grams & ref_g):
                    hit = name
                    break
            if hit:
                removed[hit + "_sent"] += 1
                removed[hit + "_rows"] += len(aspects)
                continue
            for a in aspects:
                n = len(rows) + 1
                iid = f"mams_{split}_{n:06d}"
                rows.append({"id": iid, "instance_id": iid, "sentence_id": f"mams_{split}_s{si}",
                             "sentence": text, "target": a["term"], "gold_polarity": a["polarity"],
                             "from": a["from"], "to": a["to"], "dataset": "mams", "split": split})
        out = os.path.join(args.out_dir, f"mams_{split}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"\n[{split}] {len(sentences)} sentences / {n_aspects} aspect rows in XML")
        print(f"  removed vs restaurant14_test_all : {removed['test_sent']} sentences, {removed['test_rows']} rows")
        print(f"  removed vs restaurant14_train_all: {removed['train_sent']} sentences, {removed['train_rows']} rows")
        c = Counter(r["gold_polarity"] for r in rows)
        print(f"  written {out}: {len(rows)} rows | " + ", ".join(f"{l} {c[l]}" for l in LABELS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
