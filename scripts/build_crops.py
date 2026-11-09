"""Place-view inputs: the object crop and the registered-place crop, with the place question."""
import os
import hashlib
import re
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, load  # noqa: E402

OUT = ROOT / "data/crops"
GROW, MIN_SIDE, MAX_PIX = 1.6, 0.15, 384
Q = ("Image 1 shows an object (inside the red box) in one capture of a scene. Image 2 shows the same place, marked by "
     "the red box, in the other capture. Answer \"same\" if this object is still at this place in image 2, or "
     "\"changed\" if it is not there (it was removed or moved away; if image 1 is from the later capture: it was not "
     "there before). Answer with one word.")
UNREG = bool(os.environ.get("STILL_CROP_UNREG"))
PI3 = bool(os.environ.get("STILL_CROP_PI3"))
PI3_MIN = 0.3
PI3_ANC = {}
ANC = re.compile(r"video \d, frame (\d+), around the box \[([\d, ]+)\]")


def crop(path, b, out):
    if out.exists():
        return
    im = Image.open(path).convert("RGB")
    W, H = im.size
    cx, cy = (b[0] + b[2]) / 2000 * W, (b[1] + b[3]) / 2000 * H
    w = max((b[2] - b[0]) / 1000 * W * GROW, MIN_SIDE * W)
    h = max((b[3] - b[1]) / 1000 * H * GROW, MIN_SIDE * H)
    x0, y0, x1, y1 = max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)
    ImageDraw.Draw(im).rectangle([b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H], outline="red", width=3)
    c = im.crop((int(x0), int(y0), int(x1), int(y1)))
    s = min(1.0, MAX_PIX / max(c.size))
    c = c.resize((max(28, int(c.size[0] * s)), max(28, int(c.size[1] * s))))
    c.save(out, quality=90)


def one(q):
    if PI3:
        r = PI3_ANC.get(q["id"])
        if not r or r[0] is None or r[2] < PI3_MIN or r[1][2] <= r[1][0] or r[1][3] <= r[1][1]:
            return None
        g, a = r[0], r[1]
    else:
        m = ANC.search(q["q"])
        if not m:
            return None
        g, a = int(m.group(1)), [int(t) for t in m.group(2).split(",")]
    v, f, b = q["query"]["video"], q["query"]["frame"], q["query"]["bbox_2d"]
    src = q["images"][(f - 1) if v == 1 else q["n1"] + f - 1]
    dst = q["images"][(q["n1"] + g - 1) if v == 1 else g - 1]
    h = hashlib.sha1(q["id"].encode()).hexdigest()
    sfx = "bu" if UNREG else ("bp" if os.environ.get("STILL_ANCHOR_PREFIX", "pi3") == "pi3" else "bl" + os.environ["STILL_ANCHOR_PREFIX"][4:]) if PI3 else "b"
    c1, c2 = OUT / f"{h}_a.jpg", OUT / f"{h}_{sfx}.jpg"
    if UNREG:
        a = b
    crop(src, b, c1)
    crop(dst, a, c2)
    return dict(id=q["id"], item=q["item"], group=q["group"], task="crop", gold="changed" if q["gold"] == "changed" else "same",
                gold3=q["gold"], obj=q["obj"], kind=q["kind"], query=q["query"], anchor=dict(frame=g, bbox_2d=a),
                images=[str(c1), str(c2)], q=Q)


def main():
    OUT.mkdir(exist_ok=True)
    srcs = (("train", "verN_train"), ("dev", "ver_dev"), ("test", "ver_test"), ("ctl", "ver_ctl")) if PI3 else \
        (("train", "verNA_train"), ("dev", "verA_dev"), ("test", "verA_test"), ("ctl", "verA_ctl"))
    if os.environ.get("STILL_CROP_SETS") == "train":  # verifier training queries only
        srcs = (("train", "verN_train"),)
    elif os.environ.get("STILL_CROP_SETS") == "vscd":
        srcs = (("vscd_dev", "vscd_dev"), ("vscd_test", "vscd_test"), ("vscd_real", "vscd_real"))
    elif os.environ.get("STILL_CROP_SETS") == "kc3d":
        srcs = (("kc3d_train", "kc3d_train"),)
    elif os.environ.get("STILL_CROP_SETS") == "pas":  # PASLCD
        srcs = (("pas_test", "pas_test"),)
    elif os.environ.get("STILL_CROP_SETS") == "rc3d":
        srcs = (("vsd_rc3d", "vsd_rc3d"),)
    elif os.environ.get("STILL_CROP_SETS") == "off":  # official protocol
        srcs = (("off_test", "off_test"),)
    elif os.environ.get("STILL_CROP_SETS") == "vsd_sdk":  # SD-K
        srcs = (("vsd_sdk_dev", "vsd_sdk_dev"), ("vsd_sdk_test", "vsd_sdk_test"))
    elif os.environ.get("STILL_CROP_SETS") == "vsd":  # SceneDiff-proposal queries (build_sdiff_queries.py)
        srcs = (("vsd_dev", "vsd_dev"), ("vsd_test", "vsd_test"), ("vsd_ctl", "vsd_ctl"))
    for name, src in srcs:
        qs = load(ROOT / f"data/p2/items_{src}.jsonl")
        if PI3:
            import json as _j
            PI3_ANC.clear()
            pre = os.environ.get("STILL_ANCHOR_PREFIX", "pi3")  # "lrn_<tag>" = the model's own predicted places
            PI3_ANC.update(_j.load(open(ROOT / f"data/p2/{pre}_anchor_{src}.json")))
        with Pool(32) as pool:
            rows = [r for r in pool.imap(one, qs, chunksize=64) if r]
        tagc = "U" if UNREG else ("L" + os.environ["STILL_ANCHOR_PREFIX"][4:] if os.environ.get("STILL_ANCHOR_PREFIX", "pi3") != "pi3" else "P") if PI3 else ""
        dump(rows, ROOT / f"data/p2/items_crop{tagc}_{name}.jsonl")
        from collections import Counter
        print(name, len(qs), "queries ->", len(rows), "anchored", dict(Counter(r["gold3"] for r in rows)), flush=True)


if __name__ == "__main__":
    main()
