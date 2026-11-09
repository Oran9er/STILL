"""Learned registration: RL items for the refinement policy."""
import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

os.environ["STILL_REF_WIDE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_refine  # noqa: E402
from build_crops import OUT, crop  # noqa: E402
from still_common import ROOT, dump, load  # noqa: E402

P = ROOT / "data/p2"
TAG = os.environ["STILL_RL_TAG"]  # name of the stage-1 localisation run


def one(job):
    q, g, coarse, tgt = job
    it = build_refine.one((q, g, coarse, None, "rl"))
    if it is None:
        return None
    v, f, n1 = q["query"]["video"], q["query"]["frame"], q["n1"]
    src = q["images"][(f - 1) if v == 1 else n1 + f - 1]
    dst = q["images"][(n1 + g - 1) if v == 1 else g - 1]
    qc = OUT / f"{hashlib.sha1(q['id'].encode()).hexdigest()}_a.jpg"
    crop(src, q["query"]["bbox_2d"], qc)
    it.update(dst=dst, qcrop=str(qc), tgt=tgt, label="changed" if q["gold"] == "changed" else "same")
    return it


def main():
    jobs = []
    qs = {q["id"]: q for q in load(P / "items_verN_train.jsonl")}
    allf = json.load(open(P / "pi3_all_verN_train.json"))
    co = json.load(open(P / f"lrn_{TAG}_anchor_verN_train.json"))
    for i, r in co.items():
        q = qs.get(i)
        if q is None or r[0] is None or q["gold"] not in ("changed", "same"):
            continue
        t = allf.get(i, {}).get(str(r[0]))
        if t and t[1] >= 0.3:
            jobs.append((q, r[0], r[1], t[0]))
    mode = sys.argv[1] if len(sys.argv) > 1 else ""  # fresh: KC-3D changed queries from pairs unseen in SFT
    if mode == "xfitB":  # SD-V training queries of the B scenes only (unseen by the half-A model), both labels
        B = set(json.load(open(P / "xfit_split.json"))["B"])
        jobs = [j for j in jobs if j[0]["group"] in B]
        with Pool(32) as pool:
            rows = [x for x in pool.imap(one, jobs, chunksize=32) if x]
        dump(rows, P / "items_rlref_xfitB.jsonl")
        print(len(jobs), "jobs ->", len(rows), "RL refine items (B scenes)", dict(Counter(x["label"] for x in rows)))
        return
    fresh = mode in ("fresh", "fresh2")  # (build_kc3d_fresh.py); fresh2 adds the seed-2 pairs (v2b)
    if fresh:
        jobs = [j for j in jobs if j[0]["gold"] == "same"]  # SD-V unchanged queries only
    sets = {"": [("kc3d_train", "kc3d_train_chg")], "fresh": [("kc3dF_train", "kc3dF_train")],
            "fresh2": [("kc3dF_train", "kc3dF_train"), ("kc3dF2_train", "kc3dF2_train")]}[mode]
    for ks, ls in sets:
        kq = {q["id"]: q for q in load(P / f"items_{ks}.jsonl")}
        ka = json.load(open(P / f"pi3_anchor_{ks}.json"))
        kco = json.load(open(P / f"lrn_{TAG}_anchor_{ls}.json"))
        for i, r in kco.items():
            q, t = kq.get(i), ka.get(i)
            if q is None or r[0] is None or not t or t[0] is None or t[2] < 0.3:
                continue
            jobs.append((q, 1, r[1], t[1]))
    with Pool(32) as pool:
        rows = [x for x in pool.imap(one, jobs, chunksize=32) if x]
    dump(rows, P / f"items_rlref_{mode or 'train'}.jsonl")
    print(len(jobs), "jobs ->", len(rows), "RL refine items", dict(Counter((x["label"], "kc3d" if x["group"].startswith("kc3d") else "sdv") for x in rows)))


if __name__ == "__main__":
    main()
