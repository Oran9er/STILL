"""Two-frame SceneDiff items with a binary change label, used as auxiliary data for the holistic baseline."""
import os
import collections
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / "ext/scenediff/scenediff_benchmark"
K = {"test": 12, "val": 40}
NEAR, AREA, RING = 30, 0.005, 4
POOL = sys.argv[1] if len(sys.argv) > 1 else "hi"  # "lo" = the remaining pairs (30-99 inliers or matches on < half a frame)


class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def main():
    split = {}
    for sp in ("val", "test"):
        for n in json.load(open(SRC / f"splits/{sp}_split.json"))["varied"]:
            split[n] = sp
    match = json.load(open(ROOT / "data/sd_match.json"))
    rng = random.Random(0)
    by = {sp: collections.defaultdict(lambda: {"no": [], "yes": []}) for sp in ("val", "test")}
    stat = collections.Counter()
    for seq, r in sorted(match.items()):
        if seq not in split:
            continue
        seg = Safe(open(SRC / "data" / seq / "segments.pkl", "rb")).load()
        frames = {v: {int(o): sorted(int(k) for k in fr) for o, fr in seg[f"video{v}_objects"].items()} for v in (1, 2)}

        def near(v, f):
            return {o for o, ks in frames[v].items() if any(abs(k - f) <= NEAR for k in ks)}
        for row in r["rows"]:
            hi = row["inl"] >= 100 and min(row["cov"]) >= 0.5
            if hi != (POOL == "hi"):
                continue
            f = {1: row["f1"], 2: row["f2"]}
            nr = {1: near(1, f[1]), 2: near(2, f[2])}
            if not nr[1] and not nr[2]:
                lab, objs = "no", []
            else:
                objs = []
                for o in row["obs"]:
                    m = r["meta"][o["oid"]]
                    kind = "moved" if m["in1"] and m["in2"] else "removed" if m["in1"] else "added"
                    if o["area"] < AREA or o["ring"] < RING:
                        continue
                    if kind == "moved" and o["oid"] in nr[3 - o["view"]]:
                        continue
                    objs.append(dict(label=m["label"], kind=kind, view=o["view"], box=o["box"]))
                if not objs:
                    stat[(split[seq], "ambiguous")] += 1
                    continue
                lab = "yes"
            stat[(split[seq], lab)] += 1
            by[split[seq]][seq][lab].append(dict(
                id=f"{seq}|{row['f1']}|{row['f2']}", group=seq, task="bin", domain="scene", gold=lab, objects=objs,
                images=[str(ROOT / f"data/img_sd/{seq}/v1_{row['f1']:04d}.jpg"),
                        str(ROOT / f"data/img_sd/{seq}/v2_{row['f2']:04d}.jpg")], inl=row["inl"]))
    print(sorted(stat.items()))
    pre = "sd" if POOL == "hi" else "sdlo"
    for sp, name in (("test", pre + "_test"), ("val", pre + "_train")):
        out = []
        for seq in sorted(by[sp]):
            for lab in ("no", "yes"):
                c = by[sp][seq][lab]
                rng.shuffle(c)
                s1, s2, pick = set(), set(), []
                for it in c:  # first pairs that reuse neither frame, then fill up
                    if it["images"][0] not in s1 and it["images"][1] not in s2:
                        pick.append(it)
                        s1.add(it["images"][0])
                        s2.add(it["images"][1])
                pick += [it for it in c if it not in pick]
                out += [dict(it, set=name) for it in pick[:K[sp]]]
        with open(ROOT / f"data/items_{name}.jsonl", "w") as fo:
            for it in out:
                fo.write(json.dumps(it) + "\n")
        g = collections.defaultdict(set)
        for it in out:
            g[it["group"]].add(it["gold"])
        print(name, len(out), "yes", sum(o["gold"] == "yes" for o in out), "groups", len(g),
              "with both", sum(len(v) == 2 for v in g.values()),
              collections.Counter(o["kind"] for it in out for o in it["objects"]))


if __name__ == "__main__":
    main()
