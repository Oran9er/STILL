"""KC-3D synthetic two-view pairs as scene items."""
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump  # noqa: E402

SRC = ROOT / "ext/kc3d"
N = 3000


class Safe(pickle.Unpickler):
    def find_class(self, m, n):
        import builtins
        if m == "builtins" and n in ("dict", "list"):
            return getattr(builtins, n)
        raise pickle.UnpicklingError(f"blocked {m}.{n}")


def comps(mask_path):
    m = (np.array(Image.open(mask_path)) > 0).astype(np.uint8)
    H, W = m.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= 0.003 * H * W:
            out.append([round(1000 * x / W), round(1000 * y / H), round(1000 * (x + w) / W), round(1000 * (y + h) / H)])
    return out


def main():
    split = Safe(open(SRC / "data_split.pkl", "rb")).load()["train"]
    rng = random.Random(0)
    pick = rng.sample(split, N)
    out = []
    for s in pick:
        gt = []
        for v, mk in ((1, s["mask1"]), (2, s["mask2"])):
            for b in comps(SRC / mk):
                gt.append(dict(label="object", kind="moved", boxes={f"{v}:1": b}))
        if not gt:
            continue
        key = Path(s["image1"]).stem.rsplit("_", 1)[0]
        out.append(dict(id=f"kc3d|{key}", set="kc3d_train", group=f"kc3d_{key}", task="scene", gold="yes",
                        images=[str(SRC / s["image1"]), str(SRC / s["image2"])], n1=1, n2=1, entries=[], gt=gt,
                        ignore=[], control=False))
    (ROOT / "data/kc3d").mkdir(exist_ok=True)
    dump(out, ROOT / "data/kc3d/items_kc3d_train.jsonl")
    print(len(out), "pairs,", sum(len(it["gt"]) for it in out), "changed-object views")


if __name__ == "__main__":
    main()
