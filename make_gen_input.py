"""
Build the generator input from an all-rows file: copies instance_id into id (what
auxiliary_generator.py expects). Needs only the standard library.

Usage: python make_gen_input.py [--dataset restaurant|laptop] [--input ...] [--output ...]
Default: data/processed/restaurant14_train_all.json -> data/processed/restaurant14_train_all_gen.json
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["restaurant", "laptop"], default="restaurant")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    name = "laptop14" if args.dataset == "laptop" else "restaurant14"
    src = args.input or f"data/processed/{name}_train_all.json"
    dst = args.output or f"data/processed/{name}_train_all_gen.json"

    rows = json.load(open(src, encoding="utf-8"))
    n_set = 0
    for r in rows:
        if "id" not in r:
            r["id"] = r["instance_id"]
            n_set += 1
        elif r.get("instance_id") and r["id"] != r["instance_id"]:
            raise SystemExit(f"row has id {r['id']!r} != instance_id {r['instance_id']!r}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids in input")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"{src} -> {dst}: {len(rows)} rows ({n_set} ids copied from instance_id)")


if __name__ == "__main__":
    main()
