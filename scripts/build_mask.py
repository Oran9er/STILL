"""Learned registration: masked query frames, so that the place must be found from the surroundings."""
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, load  # noqa: E402

OUT = ROOT / "data/masked"
NOTE = (" The object's own box is greyed out in that frame: find its place from the surroundings.")


def masked(src, b, key):
    """the frame src with box b (0-1000) filled grey; cached by key."""
    out = OUT / f"{hashlib.sha1(key.encode()).hexdigest()}.jpg"
    if not out.exists():
        im = Image.open(src).convert("RGB")
        W, H = im.size
        ImageDraw.Draw(im).rectangle([b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H], fill=(128, 128, 128))
        im.save(out, quality=90)
    return str(out)


def loc_one(it):
    v, f, b = it["query"]["video"], it["query"]["frame"], it["query"]["bbox_2d"]
    k = (f - 1) if v == 1 else it["n1"] + f - 1
    ims = list(it["images"])
    ims[k] = masked(ims[k], b, it["id"])
    return dict(it, images=ims, q=it["q"].replace(" Find the same place", NOTE + " Find the same place"))


def ref_one(it):
    from build_crops import crop
    q = it["query"]
    src = masked_src[it["id"]]
    c1 = OUT / f"{hashlib.sha1(it['id'].encode()).hexdigest()}_am.jpg"
    crop(masked(src, q["bbox_2d"], it["id"]), q["bbox_2d"], c1)
    return dict(it, images=[str(c1), it["images"][1]])


masked_src = {}


def main():
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "loc":
        for s in sys.argv[2:]:
            items = load(ROOT / f"data/p2/items_loc_{s}.jsonl")
            with Pool(32) as pool:
                rows = list(pool.imap(loc_one, items, chunksize=32))
            dump(rows, ROOT / f"data/p2/items_locM_{s}.jsonl")
            print(s, len(rows), "masked loc items", flush=True)
        return
    qs = {q["id"]: q for q in load(ROOT / "data/p2/items_verN_train.jsonl")}
    for i, q in qs.items():
        v, f = q["query"]["video"], q["query"]["frame"]
        masked_src[i] = q["images"][(f - 1) if v == 1 else q["n1"] + f - 1]
    items = load(ROOT / "data/p2/items_ref_verN_train.jsonl")
    with Pool(32) as pool:
        rows = list(pool.imap(ref_one, items, chunksize=32))
    dump(rows, ROOT / "data/p2/items_refM_verN_train.jsonl")
    print("ref", len(rows), "masked ref items", flush=True)


if __name__ == "__main__":
    main()
