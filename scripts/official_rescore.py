"""STILL under the official protocol: builds verifier queries for every SceneDiff object and writes its masks with the fused scores."""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_verify import Q  # noqa: E402
from still_common import ROOT, dump, load  # noqa: E402

PRED = ROOT / "ext/sd_pred/sdiff"
FR = ROOT / "ext/sd_official_ff"
SRC = ROOT / "ext/scenediff/scenediff_benchmark"


def uniform(n, k=5):
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, min(k, n))})


def subset(n, f, k=5):
    s = uniform(n, k)
    if f not in s:
        s[int(np.argmin([abs(x - f) for x in s]))] = f
    return sorted(set(s))


def build():
    from pycocotools import mask as mu
    sp = json.load(open(SRC / "splits/test_split.json"))
    out = []
    for seq in sp["varied"] + sp["kitchen"]:
        om = pickle.load(open(PRED / seq / "object_masks.pkl", "rb"))
        frames = {v: sorted(str(p) for p in (FR / seq / f"video{v}_frames").glob("*.jpg")) for v in (1, 2)}
        for oid, d in om.items():
            if oid in ("H", "W"):
                continue
            for vk, fr in d.items():
                v = 1 if vk == "video_1" else 2
                w = 3 - v
                for f, m in fr.items():
                    f = int(f)
                    mk = mu.decode(m["mask"])
                    ys, xs = np.where(mk)
                    if len(xs) == 0:
                        continue
                    hh, ww = mk.shape
                    b = [round(1000 * xs.min() / ww), round(1000 * ys.min() / hh), round(1000 * (xs.max() + 1) / ww),
                         round(1000 * (ys.max() + 1) / hh)]
                    if b[2] <= b[0] or b[3] <= b[1]:
                        continue
                    sv, sw = subset(len(frames[v]), f), uniform(len(frames[w]))
                    imgs = {v: [frames[v][i] for i in sv], w: [frames[w][i] for i in sw]}
                    fl = sv.index(f) + 1
                    out.append(dict(id=f"{seq}|o{oid}|{v}:{f}", item=f"{seq}|{v}:{f}", group=seq, task="bin3", gold="same",
                                    images=imgs[1] + imgs[2], n1=len(imgs[1]), n2=len(imgs[2]), obj=oid, kind=None,
                                    query=dict(video=v, frame=fl, bbox_2d=b, score=float(m["cost"]), orig_frame=f),
                                    q=Q.format(v=v, f=fl, b=b, w=w)))
    dump(out, ROOT / "data/p2/items_off_test.jsonl")
    print(len(out), "queries on", len({q["item"] for q in out}), "items,", len({q["group"] for q in out}), "pairs")

MAX_REF = 100  # instances with more reference frames (Lunch_room: 108-109) run out of GPU memory in SceneDiff (a 79 GB
               # allocation); their reference frames are subsampled with stride 2 everywhere


def pas_frames(inst_dir):
    imgs = sorted(x for x in inst_dir.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png"))
    ref, test = [x for x in imgs if "_test_" not in x.name], [x for x in imgs if "_test_" in x.name]
    return (ref[::2] if len(ref) > MAX_REF else ref), test


def build_paslcd():
    from pycocotools import mask as mu
    PL = ROOT / "ext/paslcd/PASLCD"
    pred = ROOT / "ext/sd_pred/paslcd_sdiff"
    out = []
    for inst in sorted(f"{s_.name}/{i.name}" for s_ in PL.iterdir() if s_.is_dir() for i in s_.iterdir() if i.is_dir()):
        key = inst.replace("/", "_")
        om = pickle.load(open(pred / key / "object_masks.pkl", "rb"))
        r_, t_ = pas_frames(PL / inst / "images")
        frames = {1: [str(x) for x in r_], 2: [str(x) for x in t_]}
        for oid, d in om.items():
            if oid in ("H", "W"):
                continue
            for vk, fr in d.items():
                v = 1 if vk == "video_1" else 2
                w = 3 - v
                for f, m in fr.items():
                    f = int(f)
                    mk = mu.decode(m["mask"])
                    ys, xs = np.where(mk)
                    if len(xs) == 0:
                        continue
                    hh, ww = mk.shape
                    b = [round(1000 * xs.min() / ww), round(1000 * ys.min() / hh), round(1000 * (xs.max() + 1) / ww),
                         round(1000 * (ys.max() + 1) / hh)]
                    if b[2] <= b[0] or b[3] <= b[1]:
                        continue
                    sv, sw = subset(len(frames[v]), f), uniform(len(frames[w]))
                    im = {v: [frames[v][i] for i in sv], w: [frames[w][i] for i in sw]}
                    fl = sv.index(f) + 1
                    out.append(dict(id=f"{key}|o{oid}|{v}:{f}", item=f"{key}|{v}:{f}", group=key, task="bin3", gold="same",
                                    images=im[1] + im[2], n1=len(im[1]), n2=len(im[2]), obj=oid, kind=None,
                                    query=dict(video=v, frame=fl, bbox_2d=b, score=float(m["cost"]), orig_frame=f),
                                    q=Q.format(v=v, f=fl, b=b, w=w)))
    dump(out, ROOT / "data/p2/items_pas_test.jsonl")
    print(len(out), "queries on", len({q["item"] for q in out}), "items,", len({q["group"] for q in out}), "instances")


def scores_paslcd(res, a, dst):
    acc = defaultdict(list)
    for r in load(res):
        inst, oid = r["id"].split("|")[0], r["id"].split("|")[1][1:]
        acc[(inst, oid)].append(r["p"][1] * max(r["query"]["score"], 1e-6) ** a)
    out = defaultdict(dict)
    for (inst, oid), v in acc.items():
        out[inst.replace("_Instance", "/Instance")][oid] = float(np.mean(v))
    json.dump(out, open(dst, "w"))
    print("objects scored", sum(len(v) for v in out.values()))


def write(res, a, dst, thr=None):
    sc = {r["id"]: r["p"][1] * max(r["query"]["score"], 1e-6) ** a for r in load(res)}
    dst = Path(dst)
    sp = json.load(open(SRC / "splits/test_split.json"))
    kept = dropped = 0
    for seq in sp["varied"] + sp["kitchen"]:
        om = pickle.load(open(PRED / seq / "object_masks.pkl", "rb"))
        new = {"H": om.get("H"), "W": om.get("W")} if om else {}
        for oid, d in om.items():
            if oid in ("H", "W"):
                continue
            vals, nd = [], {}
            for vk, fr in d.items():
                v = 1 if vk == "video_1" else 2
                nd[vk] = {}
                for f, m in fr.items():
                    s = sc.get(f"{seq}|o{oid}|{v}:{int(f)}", 0.0)
                    nd[vk][f] = dict(m, cost=s)
                    vals.append(s)
            if thr is not None and (not vals or np.mean(vals) < thr):
                dropped += 1
                continue
            kept += 1
            new[oid] = nd
        (dst / seq).mkdir(parents=True, exist_ok=True)
        pickle.dump(new, open(dst / seq / "object_masks.pkl", "wb"))
    print("objects kept", kept, "dropped", dropped, "->", dst)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "build_paslcd":
        build_paslcd()
    elif sys.argv[1] == "scores_paslcd":
        scores_paslcd(sys.argv[2], float(sys.argv[3]), sys.argv[4])
    else:
        thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else None
        write(sys.argv[2], float(sys.argv[3]), sys.argv[4], thr)
