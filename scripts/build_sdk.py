"""SD-K (SceneDiff kitchen subset) test and dev items."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_protocol2 as P2  # noqa: E402
import build_sds as B  # noqa: E402
from still_common import ROOT, dump  # noqa: E402


def test_items(seq):
    return [dict(it, set="sdk_test") for it in B.one((seq, "test")) + P2.extra(seq)]


def dev_items(seq):
    return [dict(it, set="sdk_dev") for it in B.one((seq, "test"))]


def main():
    sp = {s: json.load(open(B.SRC / f"splits/{s}_split.json"))["kitchen"] for s in ("val", "test")}
    with Pool(32) as pool:
        test = [it for r in pool.map(test_items, sp["test"]) for it in r]
        dev = [it for r in pool.map(dev_items, sp["val"]) for it in r]
    for name, its in (("test", test), ("dev", dev)):
        dump(its, ROOT / f"data/p2/items_sdk_{name}.jsonl")
        real = [it for it in its if not it["control"]]
        print(f"sdk_{name}", len(its), "items,", len(real), "real, objects", sum(len(it["entries"]) for it in real),
              "with changes", sum(bool(it["entries"]) for it in real))


if __name__ == "__main__":
    main()
