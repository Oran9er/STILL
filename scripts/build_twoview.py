"""Two-frame versions of the SD-V training queries (query frame and its registered place frame)."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_verify import Q  # noqa: E402
from still_common import ROOT, dump, load  # noqa: E402

anc = json.load(open(ROOT / "data/p2/pi3_anchor_verN_train.json"))
out = []
for q in load(ROOT / "data/p2/items_verN_train.jsonl"):
    a = anc.get(q["id"])
    if not a or a[0] is None or a[2] < 0.3:
        continue
    v, f, b = q["query"]["video"], q["query"]["frame"], q["query"]["bbox_2d"]
    src = q["images"][(f - 1) if v == 1 else q["n1"] + f - 1]
    dst = q["images"][(q["n1"] + a[0] - 1) if v == 1 else a[0] - 1]
    imgs = [src, dst] if v == 1 else [dst, src]
    out.append(dict(q, id=q["id"] + "|2v", images=imgs, n1=1, n2=1, query=dict(q["query"], frame=1),
                    gold="same" if q["gold"] == "unseen" else q["gold"],
                    q=Q.format(v=v, f=1, b=b, w=3 - v)))
dump(out, ROOT / "data/p2/items_ver2v_train.jsonl")
print(len(out), "two-view queries", dict(Counter(x["gold"] for x in out)))
