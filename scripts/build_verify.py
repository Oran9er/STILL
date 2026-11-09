"""Verifier training queries: Grounding DINO boxes labelled same / changed / unseen from the SceneDiff annotations."""
import json
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, iou, load  # noqa: E402

MIN_PTS, MIN_SCORE, TRAIN_N, TRAIN_NN = 5, 0.25, 8000, 16000
import os
ANCHOR = bool(os.environ.get("STILL_VER_ANCHOR"))
QA = ("Look at the object in video {v}, frame {f}, inside the box {b} (0-1000 coordinates). Its place appears in video "
      "{w}, frame {g}, around the box {a}. Answer \"same\" if this object is still at its place in video {w}, \"changed\" "
      "if it is not there any more (it was removed or moved away; for an object of video 2: it was not there before), "
      "or \"unseen\" if its place is not in view in any frame of video {w}. Answer with one word.")
QN = ("Look at the object in video {v}, frame {f}, inside the box {b} (0-1000 coordinates). No frame of video {w} was "
      "matched to its surroundings. Answer \"same\" if this object is still at its place in video {w}, \"changed\" if it is "
      "not there any more (it was removed or moved away; for an object of video 2: it was not there before), or "
      "\"unseen\" if its place is not in view in any frame of video {w}. Answer with one word.")
Q = ("Look at the object in video {v}, frame {f}, inside the box {b} (0-1000 coordinates). Compare it with the frames "
     "of video {w}. Answer \"same\" if this object is still at its place in video {w}, \"changed\" if it is not there "
     "any more (it was removed or moved away; for an object of video 2: it was not there before), or \"unseen\" if its "
     "place is not in view in any frame of video {w}. Answer with one word.")


def sift(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    k, d = cv2.SIFT_create(nfeatures=3000).detectAndCompute(im, None)
    return np.float32([x.pt for x in k]), d, im.shape[::-1]


def matches(args):
    """Inlier point pairs (in 0-1000 coordinates of each image) between two frames."""
    a, b = args
    cv2.setNumThreads(1)
    pa, da, sa = sift(a)
    pb, db, sb = sift(b)
    if da is None or db is None or len(pa) < 10 or len(pb) < 10:
        return (a, b), None
    mm = [x for x, y in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
    if len(mm) < 15:
        return (a, b), None
    A = pa[[m.queryIdx for m in mm]]
    B = pb[[m.trainIdx for m in mm]]
    F, inl = cv2.findFundamentalMat(A, B, cv2.FM_RANSAC, 2.0, 0.999)
    if F is None or inl is None:
        return (a, b), None
    k = inl.ravel().astype(bool)
    A, B = A[k] / np.float32(sa) * 1000, B[k] / np.float32(sb) * 1000
    return (a, b), (A.round(1).tolist(), B.round(1).tolist())


def in_mask(pts, b, pad=0.25):
    if not pts:
        return np.zeros(0, bool)
    p = np.array(pts)
    w, h = b[2] - b[0], b[3] - b[1]
    x0, y0, x1, y1 = b[0] - pad * w, b[1] - pad * h, b[2] + pad * w, b[3] + pad * h
    return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)


def anchor(src, dst, b):
    """Map box b through a similarity transform fitted to matched points src -> dst (0-1000 coordinates)."""
    src, dst = np.float32(src), np.float32(dst)
    T = None
    if len(src) >= 3:
        T, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=15.0)
    if T is None:
        x0, y0 = dst.min(0)
        x1, y1 = dst.max(0)
    else:
        c = np.float32([[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]])
        m = c @ T[:, :2].T + T[:, 2]
        (x0, y0), (x1, y1) = m.min(0), m.max(0)
    return [int(np.clip(round(float(x)), 0, 1000)) for x in (x0, y0, x1, y1)]


def in_box(pts, b, pad=0.25):
    if not pts:
        return 0
    p = np.array(pts)
    w, h = b[2] - b[0], b[3] - b[1]
    x0, y0, x1, y1 = b[0] - pad * w, b[1] - pad * h, b[2] + pad * w, b[3] + pad * h
    return int(((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)).sum())


def queries(it, props, M):
    n1 = it["n1"]
    out = []
    for idx, p in enumerate(it["images"]):
        v, f = (1, idx + 1) if idx < n1 else (2, idx - n1 + 1)
        others = it["images"][n1:] if v == 1 else it["images"][:n1]
        key = f"{v}:{f}"
        for q in props.get(p, []):
            if q[4] < MIN_SCORE:
                continue
            b = q[:4]
            best, gj = 0.0, -1
            for j, g in enumerate(it["gt"]):
                if key in g["boxes"]:
                    x = iou(b, g["boxes"][key])
                    if x > best:
                        best, gj = x, j
            ign = any(g["video"] == v and g["frame"] == f and iou(b, g["bbox_2d"]) >= 0.1 for g in it["ignore"])
            if (0.1 <= best < 0.4) or (best < 0.1 and ign):
                continue
            seen, anc = 0, None
            for g_i, o in enumerate(others):
                m = M.get((p, o)) if (p, o) in M else M.get((o, p))
                if m is None:
                    continue
                pts, opp = (m[0], m[1]) if (p, o) in M else (m[1], m[0])
                k = in_box(pts, b)
                if k > seen:
                    seen = k
                    if ANCHOR and k >= MIN_PTS:
                        sel = in_mask(pts, b)
                        anc = (g_i + 1, anchor(np.array(pts)[sel], np.array(opp)[sel], b))
            gold = "unseen" if seen < MIN_PTS else ("changed" if best >= 0.4 else "same")
            out.append(dict(id=f"{it['id']}|{key}|{','.join(map(str, b))}", group=it["group"], task="bin3", gold=gold,
                            images=it["images"], n1=n1, n2=it["n2"], query=dict(video=v, frame=f, bbox_2d=b, score=q[4]),
                            q=(QA.format(v=v, f=f, b=b, w=3 - v, g=anc[0], a=anc[1]) if anc else QN.format(v=v, f=f, b=b, w=3 - v))
                            if ANCHOR else Q.format(v=v, f=f, b=b, w=3 - v), obj=gj if best >= 0.4 else None, item=it["id"],
                            kind=it["gt"][gj]["kind"] if best >= 0.4 else None))
    return out


def main():
    props = json.load(open(ROOT / "data/props_sds_all.json"))
    out_dir = ROOT / "data"
    if sys.argv[1:] == ["p2"]:
        out_dir = ROOT / "data/p2"
        t = load(out_dir / "items_sds_test.jsonl")
        sets = {"test": [it for it in t if not it["control"]],
                "ctl": [it for it in t if it["control"]] + load(ROOT / "data/items_sds_hc_test.jsonl"),
                "dev": load(out_dir / "items_sds_dev.jsonl") + load(out_dir / "items_sds_hc_dev.jsonl")}
    elif sys.argv[1:] == ["trainN"]:
        tr = load(ROOT / "data/items_sds_train12.jsonl")
        sets = {"real": [it for it in tr if not it["control"]],
                "ctl": [it for it in tr if it["control"]] + load(ROOT / "data/items_sds_hc_train.jsonl")}
    elif sys.argv[1:] == ["ctl"]:
        sets = {"ctl": [it for it in load(ROOT / "data/items_sds_test.jsonl") if it["control"]]
                + load(ROOT / "data/items_sds_hc_test.jsonl")}
    else:
        sets = {"train": [it for it in load(ROOT / "data/items_sds_train12.jsonl") if not it["control"]],
                "test": [it for it in load(ROOT / "data/items_sds_test.jsonl") if not it["control"]]}
    pairs = sorted({(a, b) for its in sets.values() for it in its for a in it["images"][:it["n1"]] for b in it["images"][it["n1"]:]})
    print("frame pairs", len(pairs), flush=True)
    with Pool(40) as pool:
        M = dict(pool.imap_unordered(matches, pairs, chunksize=16))
    rng = random.Random(0)
    if "real" in sets:
        qr = [q for it in sets["real"] for q in queries(it, props, M)]
        qc = [q for it in sets["ctl"] for q in queries(it, props, M)]
        pos = [q for q in qr if q["gold"] == "changed"]
        neg_r = [q for q in qr if q["gold"] != "changed"]
        neg_c = [q for q in qc if q["gold"] != "changed"]
        k = TRAIN_NN // 2
        qs = pos + pos + rng.sample(neg_r, min(k, len(neg_r))) + rng.sample(neg_c, min(k, len(neg_c)))
        rng.shuffle(qs)
        dump(qs, ROOT / f"data/items_verN{'A' if ANCHOR else ''}_train.jsonl")
        print("pool real", dict(Counter(q["gold"] for q in qr)), "pool ctl", dict(Counter(q["gold"] for q in qc)),
              "-> written", len(qs), dict(Counter(q["gold"] for q in qs)), flush=True)
        return
    for name, its in sets.items():
        qs = [q for it in its for q in queries(it, props, M)]
        c = Counter(q["gold"] for q in qs)
        if name == "train":
            by = defaultdict(list)
            for q in qs:
                by[q["gold"]].append(q)
            k = TRAIN_N // 3
            qs = [x for g in ("same", "changed", "unseen") for x in rng.sample(by[g], min(k, len(by[g])))]
            rng.shuffle(qs)
        dump(qs, out_dir / f"items_ver{'A' if ANCHOR else ''}_{name}.jsonl")
        print(name, "all queries", dict(c), "-> written", len(qs), dict(Counter(q["gold"] for q in qs)), flush=True)


if __name__ == "__main__":
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
