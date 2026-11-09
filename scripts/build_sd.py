"""Extracts SceneDiff SD-V frames (540 x 960, every 15th annotated frame) and the per-frame boxes of the changed objects."""
import collections
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / "ext/scenediff/scenediff_benchmark"
OUT = ROOT / "data/img_sd"
STEP, LONG, POS_AREA, MIN_INL, RING_INL = 15, 960, 0.002, 30, 4


class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def rle_decode(counts, h, w):
    """COCO compressed RLE string -> (h, w) uint8 mask."""
    s = counts.encode() if isinstance(counts, str) else counts
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = s[p] - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    m = np.zeros(h * w, np.uint8)
    pos, v = 0, 0
    for c in cnts:
        m[pos:pos + c] = v
        pos += c
        v = 1 - v
    return m.reshape((h, w), order="F")


def n_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    return n


def read_frames(path, annot):
    """Frames keyed by the index in the 30-fps ANNOTATED video (the index space of the masks)."""
    n_o = n_frames(path)
    n_a = n_frames(annot) if Path(annot).exists() else n_o
    want = {min(n_o - 1, round(a * n_o / n_a)): a for a in range(0, n_a, STEP)}
    cap = cv2.VideoCapture(str(path))
    out, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out[want[i]] = fr
        i += 1
    return out


def one(seq):
    cv2.setNumThreads(1)
    d = SRC / "data" / seq
    seg = Safe(open(d / "segments.pkl", "rb")).load()
    vids = {}
    for v in (1, 2):
        cand = sorted(d.glob(f"original_video{v}.*"))
        if not cand:
            return seq, None
        vids[v] = read_frames(cand[0], d / f"video{v}.mp4")
    (OUT / seq).mkdir(parents=True, exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=4000)
    feats, objs = {}, {}
    for v in (1, 2):
        vo = seg[f"video{v}_objects"]
        for fi, fr in vids[v].items():
            h, w = fr.shape[:2]
            sc = LONG / max(h, w)
            small = cv2.resize(fr, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
            p = OUT / seq / f"v{v}_{fi:04d}.jpg"
            cv2.imwrite(str(p), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            kp, des = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
            pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
            masks = {}
            for oid, frames in vo.items():
                r = frames.get(str(fi)) if frames else None
                if r is None:
                    continue
                mh, mw = r["size"]
                m = rle_decode(r["counts"], mh, mw)
                if m.sum() == 0:
                    continue
                m = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
                if m.sum() == 0:
                    continue
                ys, xs = np.where(m)
                masks[int(oid)] = dict(box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                                       area=float(m.mean()), m=m)
            feats[(v, fi)] = (pts, des, small.shape[:2])
            objs[(v, fi)] = masks
    bf = cv2.BFMatcher(cv2.NORM_L2)
    meta = [dict(label=o["label"], in1=bool(o["in_video1"]), in2=bool(o["in_video2"]),
                 deform=o.get("deformability")) for o in seg["objects"]]
    rows = []
    for fi in vids[1]:
        p1, d1, s1 = feats[(1, fi)]
        for fj in vids[2]:
            p2, d2, s2 = feats[(2, fj)]
            if d1 is None or d2 is None or len(p1) < 20 or len(p2) < 20:
                continue
            mm = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
            if len(mm) < MIN_INL:
                continue
            a = p1[[x.queryIdx for x in mm]]
            b = p2[[x.trainIdx for x in mm]]
            F, inl = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 2.0, 0.999)
            if F is None or inl is None:
                continue
            inl = inl.ravel().astype(bool)
            n = int(inl.sum())
            if n < MIN_INL:
                continue
            a, b = a[inl], b[inl]
            # spread of the matched points: fraction of a 4x4 grid touched in each image (co-visibility proxy)
            cov = []
            for pts, (h, w) in ((a, s1), (b, s2)):
                g = set(zip(np.clip((pts[:, 0] / w * 4).astype(int), 0, 3), np.clip((pts[:, 1] / h * 4).astype(int), 0, 3)))
                cov.append(len(g) / 16)
            ob = []
            for v, key, pts in ((1, (1, fi), a), (2, (2, fj), b)):
                for oid, mk in objs[key].items():
                    x0, y0, x1, y1 = mk["box"]
                    ex, ey = 0.5 * (x1 - x0) + 40, 0.5 * (y1 - y0) + 40
                    xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
                    near = (pts[:, 0] > x0 - ex) & (pts[:, 0] < x1 + ex) & (pts[:, 1] > y0 - ey) & (pts[:, 1] < y1 + ey)
                    on = mk["m"][np.clip(yi, 0, mk["m"].shape[0] - 1), np.clip(xi, 0, mk["m"].shape[1] - 1)] > 0
                    ob.append(dict(oid=oid, view=v, box=mk["box"], area=round(mk["area"], 5),
                                   ring=int((near & ~on).sum()), on=int((near & on).sum())))
            rows.append(dict(seq=seq, f1=fi, f2=fj, inl=n, cov=[round(c, 3) for c in cov], obs=ob))
    return seq, dict(meta=meta, rows=rows, scene=seg.get("scene_type"), light=seg.get("light_change"),
                     n1=len(vids[1]), n2=len(vids[2]))


def label(row, meta):
    """-> (changed, objects) or None if ambiguous."""
    if not row["obs"]:
        return 0, []
    good = {}
    for o in row["obs"]:
        if o["area"] >= POS_AREA and o["ring"] >= RING_INL:
            good.setdefault(o["oid"], {})[o["view"]] = o["box"]
    if not good:
        return None
    out = []
    by = {}
    for o in row["obs"]:
        by.setdefault(o["oid"], {})[o["view"]] = o["box"]
    for oid, _ in good.items():
        m = meta[oid]
        kind = "moved" if m["in1"] and m["in2"] else "removed" if m["in1"] else "added"
        out.append(dict(label=m["label"], kind=kind, box1=by[oid].get(1), box2=by[oid].get(2)))
    return 1, out


def main():
    split = {}
    for sp in ("val", "test"):
        for sub, names in json.load(open(SRC / f"splits/{sp}_split.json")).items():
            for n in names:
                split[n] = (sp, sub)
    seqs = sorted(p.name for p in (SRC / "data").iterdir() if (p / "segments.pkl").exists())
    if len(sys.argv) > 1:
        seqs = seqs[:int(sys.argv[1])]
    res = {}
    with Pool(40) as pool:
        for k, (seq, r) in enumerate(pool.imap_unordered(one, seqs)):
            if r is not None:
                res[seq] = r
            if k % 25 == 0:
                print(f"[sd] {k}/{len(seqs)}", flush=True)
    json.dump(res, open(ROOT / "data/sd_match.json", "w"))
    n = collections.Counter()
    with open(ROOT / "data/sd_pairs.jsonl", "w") as fo:
        for seq in sorted(res):
            r = res[seq]
            sp, sub = split.get(seq, ("none", "none"))
            for row in r["rows"]:
                lab = label(row, r["meta"])
                if lab is None:
                    n[(sp, sub, "ambiguous")] += 1
                    continue
                n[(sp, sub, lab[0])] += 1
                fo.write(json.dumps(dict(
                    id=f"{seq}|{row['f1']}|{row['f2']}", seq=seq, split=sp, subset=sub, changed=lab[0], objects=lab[1],
                    images=[str(OUT / seq / f"v1_{row['f1']:04d}.jpg"), str(OUT / seq / f"v2_{row['f2']:04d}.jpg")],
                    inl=row["inl"], cov=row["cov"], n_changed_seq=len(r["meta"]), scene=r["scene"])) + "\n")
    for k in sorted(n, key=str):
        print(k, n[k])


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
