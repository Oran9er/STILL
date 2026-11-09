"""AP50 on RC-3D (present view, absent view and both)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import iou, load  # noqa: E402


def ap50(rows, views=(1, 2)):
    cand, n = [], 0
    for r in rows:
        for v in views:
            gt = r["gt"][0]["boxes"].get(f"{v}:1")
            if gt is None:
                continue
            n += 1
            used = False
            for p in sorted([p for p in r["pred"] if p["video"] == v], key=lambda p: -p.get("conf", 0.0)):
                hit = (not used) and iou(p["bbox_2d"], gt) >= 0.5
                used |= hit
                cand.append((p.get("conf", 0.0), int(hit)))
    cand.sort(key=lambda c: -c[0])
    h = np.array([c[1] for c in cand], float)
    if not len(h):
        return 0.0
    tp, fp = np.cumsum(h), np.cumsum(1 - h)
    prec, rec = tp / (tp + fp), tp / max(n, 1)
    pint = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * pint))


def main():
    rows = load(sys.argv[1])
    print(json.dumps(dict(file=Path(sys.argv[1]).stem, pairs=len(rows), ap50_both=round(ap50(rows), 4),
                          ap50_img1=round(ap50(rows, (1,)), 4), ap50_img2=round(ap50(rows, (2,)), 4))))


if __name__ == "__main__":
    main()
