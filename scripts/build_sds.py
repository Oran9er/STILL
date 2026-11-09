"""Scene items for the VLM-output protocol: 5 frames from each capture and every changed object with its boxes."""
import collections
import hashlib
import json
import os
import pickle
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sd import rle_decode  # noqa: E402

ROOT = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / "ext/scenediff/scenediff_benchmark"
K, MIN_AREA = 5, 0.003
R_TRAIN = int(os.environ.get("STILL_SDS_R", 4))  # random frame subsets per training scene


class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def name(label):
    s = re.sub(r"_\d+$", "", label.strip()).replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", s) or "object"


def frames_of(seq, v):
    return sorted(int(p.stem.split("_")[1]) for p in (ROOT / "data/img_sd" / seq).glob(f"v{v}_*.jpg"))


def boxes(seg, v, frames):
    """{oid: {frame: (box0-1000, area share)}} for the given annotated frame indices."""
    out = collections.defaultdict(dict)
    for oid, fr in seg[f"video{v}_objects"].items():
        for f in frames:
            r = fr.get(str(f)) if fr else None
            if r is None:
                continue
            m = rle_decode(r["counts"], *r["size"])
            if m.sum() == 0:
                continue
            ys, xs = np.where(m)
            h, w = m.shape
            out[int(oid)][f] = ([round(1000 * xs.min() / w), round(1000 * ys.min() / h),
                                 round(1000 * (xs.max() + 1) / w), round(1000 * (ys.max() + 1) / h)], float(m.mean()))
    return out


def pick(frames, k, rng=None):
    if len(frames) <= k:
        return list(frames)
    if rng is None:
        return [frames[i] for i in np.linspace(0, len(frames) - 1, k).round().astype(int)]
    return sorted(rng.sample(frames, k))


def item(seq, seg, f1, f2, tag, set_name):
    bx = {1: boxes(seg, 1, f1), 2: boxes(seg, 2, f2)}
    sel = {1: f1, 2: f2}
    gt, entries, ignore = [], [], []
    for oid, o in enumerate(seg["objects"]):
        kind = "moved" if o["in_video1"] and o["in_video2"] else "removed" if o["in_video1"] else "added"
        views = {f"{v}:{sel[v].index(f) + 1}": b for v in (1, 2) for f, (b, a) in bx[v].get(oid, {}).items()}
        areas = {f"{v}:{sel[v].index(f) + 1}": a for v in (1, 2) for f, (b, a) in bx[v].get(oid, {}).items()}
        if not views:
            continue
        if max(areas.values()) < MIN_AREA:
            ignore += [{"video": int(k[0]), "frame": int(k[2:]), "bbox_2d": b} for k, b in views.items()]
            continue
        ref = 1 if kind == "removed" else 2 if kind == "added" else (1 if any(k[0] == "1" for k in views) else 2)
        cand = {k: a for k, a in areas.items() if int(k[0]) == ref} or areas
        best = max(cand, key=cand.get)
        gt.append(dict(label=name(o["label"]), kind=kind, boxes=views))
        entries.append({"change": kind, "label": name(o["label"]), "video": int(best[0]), "frame": int(best[2:]),
                        "bbox_2d": views[best]})
    entries.sort(key=lambda e: (e["video"], e["frame"], e["bbox_2d"][1], e["bbox_2d"][0]))
    d = ROOT / "data/img_sd" / seq
    return dict(id=f"{seq}|{tag}", set=set_name, group=seq, task="scene", gold="yes" if entries else "no",
                images=[str(d / f"v1_{f:04d}.jpg") for f in f1] + [str(d / f"v2_{f:04d}.jpg") for f in f2],
                n1=len(f1), n2=len(f2), entries=entries, gt=gt, ignore=ignore, control=False)


def control(seq, v, frames, tag, set_name, seg=None):
    d = ROOT / "data/img_sd" / seq
    a, b = frames[0::2], frames[1::2]
    n = min(len(a), len(b))
    ghost = []
    if seg is not None:  # where the objects that DID change between the captures sit in this single capture
        bx = boxes(seg, v, a[:n] + b[:n])
        for oid, fr in bx.items():
            views = {(f"1:{a.index(f) + 1}" if f in a[:n] else f"2:{b.index(f) + 1}"): bb for f, (bb, ar) in fr.items()
                     if ar >= MIN_AREA}
            if views:
                ghost.append(dict(label=name(seg["objects"][oid]["label"]), boxes=views))
    return dict(ghost=ghost, id=f"{seq}|{tag}", set=set_name, group=seq, task="scene", gold="no",
                images=[str(d / f"v{v}_{f:04d}.jpg") for f in a[:n]] + [str(d / f"v{v}_{f:04d}.jpg") for f in b[:n]],
                n1=n, n2=n, entries=[], gt=[], ignore=[], control=True)


def one(args):
    seq, sp = args
    seg = Safe(open(SRC / "data" / seq / "segments.pkl", "rb")).load()
    fr = {v: frames_of(seq, v) for v in (1, 2)}
    if min(len(fr[1]), len(fr[2])) < 2:
        return []
    rng = random.Random(seq)
    name_ = f"sds_{'test' if sp == 'test' else 'train'}"
    if sp == "test":
        v = 1 + rng.randrange(2)
        return [item(seq, seg, pick(fr[1], K), pick(fr[2], K), "u", name_),
                control(seq, v, pick(fr[v], 2 * K), f"c{v}", name_, seg)]
    out = []
    for r in range(R_TRAIN):
        out.append(item(seq, seg, pick(fr[1], K, rng), pick(fr[2], K, rng), f"r{r}", name_))
        v = 1 + r % 2
        out.append(control(seq, v, pick(fr[v], 2 * K, rng), f"c{r}", name_))
    return out


def main():
    jobs = []
    for sp in ("val", "test"):
        jobs += [(s, sp) for s in json.load(open(SRC / f"splits/{sp}_split.json"))["varied"]
                 if (ROOT / "data/img_sd" / s).exists()]
    with Pool(32) as pool:
        res = pool.map(one, jobs)
    for name_ in ("sds_test", "sds_train"):
        items = [it for r in res for it in r if it["set"] == name_]
        out_name = name_ if name_ == "sds_test" or R_TRAIN == 4 else f"sds_train{R_TRAIN}"
        if name_ == "sds_train" and R_TRAIN != 4:  # data dial: the same file restricted to half of the scenes
            half = [it for it in items if int(hashlib.sha1(it["group"].encode()).hexdigest(), 16) % 2 == 0]
            with open(ROOT / f"data/items_{out_name}h.jsonl", "w") as f:
                for it in half:
                    f.write(json.dumps(it) + "\n")
            print(out_name + "h", len(half), "items from", len({it["group"] for it in half}), "scenes")
        if name_ == "sds_test" and R_TRAIN != 4:
            continue
        with open(ROOT / f"data/items_{out_name}.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        real = [it for it in items if not it["control"]]
        print(name_, len(items), "items;", len(real), "real,", sum(bool(it["entries"]) for it in real), "with changes;",
              "objects", sum(len(it["entries"]) for it in real), collections.Counter(e["change"] for it in real for e in it["entries"]),
              "ignored objects", sum(len({(g["video"], g["frame"]) for g in it["ignore"]}) > 0 for it in real),
              "frames per video", collections.Counter(it["n1"] for it in real).most_common(3))


if __name__ == "__main__":
    main()
