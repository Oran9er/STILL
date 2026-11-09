"""Learned registration: items for refining the place box on a zoomed crop."""
import hashlib
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_crops import OUT, crop  # noqa: E402
from still_common import ROOT, dump, load  # noqa: E402

CTX, CTX_MIN, JIT, MAX_PIX = 3.0, 0.3, 0.5, 448
MASK = bool(__import__("os").environ.get("STILL_REF_MASK"))
WIDE = bool(__import__("os").environ.get("STILL_REF_WIDE"))  # v5
Q = ("Image 1 shows an object (inside the red box) in one capture of a scene. Image 2 is a part of a frame of the "
     "other capture around the same place. Give the box of the object's place in image 2 as {\"bbox_2d\": [x1, y1, x2, "
     "y2]} (0-1000 coordinates of image 2). Answer with the JSON only.")


def region(b, W, H):
    cx, cy = (b[0] + b[2]) / 2000 * W, (b[1] + b[3]) / 2000 * H
    w = max((b[2] - b[0]) / 1000 * W * CTX, CTX_MIN * W)
    h = max((b[3] - b[1]) / 1000 * H * CTX, CTX_MIN * H)
    return [int(max(0, cx - w / 2)), int(max(0, cy - h / 2)), int(min(W, cx + w / 2)), int(min(H, cy + h / 2))]


def to_region(b, R, W, H):
    """frame box (0-1000) -> region box (0-1000 of the region), clipped; also the share of the box inside."""
    px = [b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H]
    c = [max(px[0], R[0]), max(px[1], R[1]), min(px[2], R[2]), min(px[3], R[3])]
    area = max(px[2] - px[0], 1) * max(px[3] - px[1], 1)
    share = max(c[2] - c[0], 0) * max(c[3] - c[1], 0) / area
    rw, rh = R[2] - R[0], R[3] - R[1]
    return [int(round((c[0] - R[0]) / rw * 1000)), int(round((c[1] - R[1]) / rh * 1000)),
            int(round((c[2] - R[0]) / rw * 1000)), int(round((c[3] - R[1]) / rh * 1000))], share


def from_region(b, R, W, H):
    rw, rh = R[2] - R[0], R[3] - R[1]
    px = [R[0] + b[0] / 1000 * rw, R[1] + b[1] / 1000 * rh, R[0] + b[2] / 1000 * rw, R[1] + b[3] / 1000 * rh]
    return [int(max(0, min(1000, round(px[0] / W * 1000)))), int(max(0, min(1000, round(px[1] / H * 1000)))),
            int(max(0, min(1000, round(px[2] / W * 1000)))), int(max(0, min(1000, round(px[3] / H * 1000))))]


def one(job):
    q, g, coarse, tgt, tag = job
    v, f, n1 = q["query"]["video"], q["query"]["frame"], q["n1"]
    src = q["images"][(f - 1) if v == 1 else n1 + f - 1]
    dst = q["images"][(n1 + g - 1) if v == 1 else g - 1]
    h = hashlib.sha1(q["id"].encode()).hexdigest()
    c1 = OUT / f"{h}_a.jpg"
    if MASK:  # v3: query crop cut from the frame with the box greyed out (build_mask.py)
        from build_mask import masked
        c1 = OUT / f"{h}_am.jpg"
        src = masked(src, q["query"]["bbox_2d"], q["id"])
    if WIDE:  # v5: the query crop shows the same 3x context as image 2 (box outlined), to place it by surroundings
        c1 = OUT / f"{h}_aw.jpg"
        if not c1.exists():
            qi = Image.open(src).convert("RGB")
            qW, qH = qi.size
            qb = q["query"]["bbox_2d"]
            QR = region(qb, qW, qH)
            ImageDraw.Draw(qi).rectangle([qb[0] / 1000 * qW, qb[1] / 1000 * qH, qb[2] / 1000 * qW, qb[3] / 1000 * qH], outline="red", width=3)
            cq = qi.crop(tuple(QR))
            sq = min(1.0, MAX_PIX / max(cq.size))
            cq.resize((max(28, int(cq.size[0] * sq)), max(28, int(cq.size[1] * sq)))).save(c1, quality=90)
    else:
        crop(src, q["query"]["bbox_2d"], c1)
    im = Image.open(dst).convert("RGB")
    W, H = im.size
    R = region(coarse, W, H)
    if R[2] - R[0] < 8 or R[3] - R[1] < 8:
        return None
    ref, share = to_region(tgt, R, W, H) if tgt else (None, None)
    if tgt and (share < 0.5 or ref[2] <= ref[0] or ref[3] <= ref[1]):
        return None
    c2 = OUT / f"{h}_r{tag}.jpg"
    if not c2.exists():
        c = im.crop(tuple(R))
        s = min(1.0, MAX_PIX / max(c.size))
        c.resize((max(28, int(c.size[0] * s)), max(28, int(c.size[1] * s)))).save(c2, quality=90)
    return dict(id=q["id"], item=q["item"], group=q["group"], task="ref", gold=q["gold"], kind=q.get("kind"),
                query=q["query"], images=[str(c1), str(c2)], q=Q, ref=ref, frame=g, region=R, size=[W, H])


def main():
    random.seed(0)
    if sys.argv[1] == "train":  # train [set]: default verN_train; e.g. kc3d_train
        tset = sys.argv[2] if len(sys.argv) > 2 else "verN_train"
        qs = {q["id"]: q for q in load(ROOT / f"data/p2/items_{tset}.jsonl")}
        anc = json.load(open(ROOT / f"data/p2/pi3_anchor_{tset}.json"))
        jobs = []
        for i, q in qs.items():
            r = anc.get(i)
            if not r or r[0] is None or r[2] < 0.3 or r[1][2] <= r[1][0] or r[1][3] <= r[1][1]:
                continue
            b = r[1]
            w, h = b[2] - b[0], b[3] - b[1]
            s = random.uniform(0.7, 1.4)
            cx, cy = (b[0] + b[2]) / 2 + random.uniform(-JIT, JIT) * w, (b[1] + b[3]) / 2 + random.uniform(-JIT, JIT) * h
            coarse = [cx - s * w / 2, cy - s * h / 2, cx + s * w / 2, cy + s * h / 2]
            jobs.append((q, r[0], coarse, b, "t"))
        with Pool(32) as pool:
            rows = [r for r in pool.imap(one, jobs, chunksize=32) if r]
        dump(rows, ROOT / f"data/p2/items_ref{'W' if WIDE else ''}_{tset}.jsonl")
        print("train", len(jobs), "anchored queries ->", len(rows), "refine items", flush=True)
        return
    _, s, coarse_file, tag = sys.argv[1:5]
    qs = {q["id"]: q for q in load(ROOT / f"data/p2/items_{s}.jsonl")}
    co = json.load(open(ROOT / coarse_file))
    jobs = [(qs[i], r[0], r[1], None, tag) for i, r in co.items() if i in qs and r[0] is not None]
    with Pool(32) as pool:
        rows = [r for r in pool.imap(one, jobs, chunksize=32) if r]
    dump(rows, ROOT / f"data/p2/items_ref{'W' if WIDE else ''}_{tag}_{s}.jsonl")
    print(s, len(co), "coarse answers ->", len(rows), "refine items", flush=True)


if __name__ == "__main__":
    main()
