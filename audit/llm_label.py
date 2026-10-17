"""
Label every row of audit/annotation_sheet.xlsx with an Ollama model (default qwen2.5:7b,
temperature 0) using the same guideline as the human annotators.

Output: audit/labels_qwen7b.csv  (item,label); label is empty if the output stayed unparsable.
Resumes: items already in the output file are skipped.
Usage: python audit/llm_label.py [--model qwen2.5:7b] [--retries 3]
"""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from guideline import GUIDELINE, LABELS, SHEET  # noqa: E402
from ollama_client import OllamaClient  # noqa: E402

SYSTEM = "You are a careful annotator for aspect-based sentiment analysis. Respond only in English."
RETRY_NOTE = ("\n\nYour previous answer could not be parsed. Reply with exactly one word: "
              "positive, negative or neutral.")


def parse(text: str) -> str:
    low = text.strip().lower()
    found = [w for w in LABELS if re.search(rf"\b{w}\b", low)]
    if len(found) == 1:
        return found[0]
    for w in LABELS:            # several words mentioned: accept only a leading answer
        if low.startswith(w):
            return w
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="audit/annotation_sheet.xlsx")
    ap.add_argument("--out", default="audit/labels_qwen7b.csv")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    from openpyxl import load_workbook
    ws = load_workbook(args.sheet, read_only=True, data_only=True)[SHEET]
    it = ws.iter_rows(values_only=True)
    header = [str(h).strip().lower() if h else "" for h in next(it)]
    ci = {h: header.index(h) for h in ("item", "sentence", "target")}
    rows = [(str(r[ci["item"]]).strip(), r[ci["sentence"]], r[ci["target"]]) for r in it if r and r[ci["item"]]]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {r["item"] for r in csv.DictReader(f) if r.get("label")}
    client = OllamaClient(model=args.model, base_url=args.base_url, temperature=0.0, num_predict=20)

    new_file = not os.path.exists(args.out)
    n_fail = 0
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["item", "label"])
        for k, (item, sentence, target) in enumerate(rows, 1):
            if item in done:
                continue
            prompt = (f"{GUIDELINE}\n\nSentence: {sentence}\nTarget: {target}\n\n"
                      "Answer with exactly one word: positive, negative or neutral.")
            label, text = "", ""
            for attempt in range(1 + args.retries):
                text = client.generate(prompt + (RETRY_NOTE if attempt else ""), system=SYSTEM, temperature=0.0)
                label = parse(text)
                if label:
                    break
            if not label:
                n_fail += 1
                print(f"[unparsable] {item}: {text!r}")
            w.writerow([item, label])
            f.flush()
            if k % 25 == 0:
                print(f"{k}/{len(rows)} done")
    print(f"written {args.out} | unparsable after retries: {n_fail}")


if __name__ == "__main__":
    main()
