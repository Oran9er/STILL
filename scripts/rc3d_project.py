"""Adds the registered place of every RC-3D detection in the other image, so that AP50 can be computed over both images."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, load  # noqa: E402

anc = json.load(open(ROOT / "data/p2/pi3_anchor_vsd_rc3d.json"))
rows = load(sys.argv[1])
added = 0
for r in rows:
    extra = []
    for p in r["pred"]:
        a = anc.get(f"{r['id']}|{p['video']}:{p['frame']}|{','.join(map(str, p['bbox_2d']))}")
        if a and a[0] is not None and a[2] >= 0.3:
            extra.append(dict(p, video=3 - p["video"], frame=a[0], bbox_2d=a[1], projected=True))
    r["pred"] = r["pred"] + extra
    added += len(extra)
dump(rows, sys.argv[2])
print("projected entries added", added)
