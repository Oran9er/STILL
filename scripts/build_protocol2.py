"""Holds out 10 dev scenes from the SceneDiff training split and writes the train / dev / test item files."""
import hashlib
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sds as B  # noqa: E402
from still_common import ROOT, dump, load  # noqa: E402

R_EXTRA = 2
OUT = ROOT / "data/p2"
TRAIN_FILES = ["sds_train", "sds_train12", "sds_train12e", "sds_train12h", "sds_train12he", "sds_train12o",
               "sds_train12oe", "sds_train12x", "sds_hc_train", "sds_hc_traine", "sds_traine", "sdlc_train", "sdsvl_train",
               "sd_train", "sdsv_trainh", "sdlc_trainA", "sdsvl_trainA", "sd_trainA", "sdsv_trainhA", "sds_trainp", "sds_hc_trainp", "ver_train", "ver2_train", "verN_train"]


def dev_scenes():
    base = sorted({it["group"] for it in load(ROOT / "data/items_sds_train.jsonl")})
    half = lambda s: int(hashlib.sha1(s.encode()).hexdigest(), 16) % 2
    key = lambda s: hashlib.sha1(("dev" + s).encode()).hexdigest()
    return sorted(sorted([s for s in base if half(s) == h], key=key)[:5] for h in (0, 1))[0] + \
        sorted(sorted([s for s in base if half(s) == h], key=key)[:5] for h in (0, 1))[1]


def extra(seq):
    seg = B.Safe(open(B.SRC / "data" / seq / "segments.pkl", "rb")).load()
    fr = {v: B.frames_of(seq, v) for v in (1, 2)}
    rng = random.Random("test2" + seq)
    out = []
    for r in range(R_EXTRA):
        out.append(B.item(seq, seg, B.pick(fr[1], B.K, rng), B.pick(fr[2], B.K, rng), f"t{r}", "sds_test"))
        v = 1 + rng.randrange(2)
        out.append(B.control(seq, v, B.pick(fr[v], 2 * B.K, rng), f"d{r}{v}", "sds_test", seg))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    dev = set(dev_scenes())
    print("dev scenes", sorted(dev))
    json.dump(sorted(dev), open(OUT / "dev_scenes.json", "w"))
    for f in TRAIN_FILES:
        p = ROOT / f"data/items_{f}.jsonl"
        if not p.exists():
            print("skip", f)
            continue
        its = load(p)
        keep = [it for it in its if it["group"] not in dev]
        dump(keep, OUT / f"items_{f}.jsonl")
        print(f, len(its), "->", len(keep))
    ids = lambda it: True
    for src, dst in (("sds_train12", "sds_dev"), ("sds_train12e", "sds_deve"), ("sds_hc_train", "sds_hc_dev"),
                     ("sds_hc_traine", "sds_hc_deve")):
        its = [it for it in load(ROOT / f"data/items_{src}.jsonl") if it["group"] in dev and (ids(it) or "hc" in src)]
        for it in its:
            it["set"] = dst
        dump(its, OUT / f"items_{dst}.jsonl")
        print(dst, len(its), "real", sum(not it["control"] for it in its))
    test = load(ROOT / "data/items_sds_test.jsonl")
    with Pool(32) as pool:
        add = [it for r in pool.map(extra, sorted({it["group"] for it in test})) for it in r]
    test += add
    dump(test, OUT / "items_sds_test.jsonl")
    real = [it for it in test if not it["control"]]
    print("p2 sds_test", len(test), "items,", len(real), "real, objects", sum(len(it["entries"]) for it in real))


if __name__ == "__main__":
    main()
