"""RC-3D detections from a run's verifier scores."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, load  # noqa: E402

n = sys.argv[1]
a = json.load(open(ROOT / f"logs/fuse_{n}sd.json"))["a"]
items = {it["id"]: it for it in load(ROOT / "data/rc3d/items_rc3d.jsonl")}
by = collections.defaultdict(list)
for r in load(ROOT / f"res/{n}rc_tv_test.jsonl"):
    q = r["query"]
    by[r["item"]].append(dict(change="moved", video=q["video"], frame=q["frame"], bbox_2d=q["bbox_2d"],
                              conf=r["p"][1] * max(q["score"], 1e-6) ** a))
dump([dict(it, pred=by.get(i, [])) for i, it in items.items()], ROOT / f"res/{n}_rc3d_Vg.jsonl")
