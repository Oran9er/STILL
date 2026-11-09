"""Evaluates SD-K at the operating point chosen on the SD-V dev split."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, load, scene_ap, scene_metrics  # noqa: E402

n = sys.argv[1]
op = json.load(open(ROOT / f"logs/fuse_{n}sd.json"))
a, thr = op["a"], op["thr"]
by = collections.defaultdict(list)
for r in load(ROOT / f"res/{n}sdk_tv_test.jsonl"):
    q = r["query"]
    by[r["item"]].append(dict(change="removed" if q["video"] == 1 else "added", video=q["video"], frame=q["frame"],
                              bbox_2d=q["bbox_2d"], conf=r["p"][1] * max(q["score"], 1e-6) ** a))
R = [dict(id=it["id"], group=it["group"], gt=it["gt"], ignore=it["ignore"], control=it["control"], pred=by.get(it["id"], []))
     for it in load(ROOT / "data/p2/items_sdk_test.jsonl")]
dump(R, ROOT / f"res/{n}sdk_fuse_sys_test.jsonl")
cut = [dict(r, pred=[p for p in r["pred"] if p["conf"] >= thr]) for r in R]
m = scene_metrics(cut)
print(json.dumps(dict(run=n, set="sdk_test", a=a, thr=thr, ap=scene_ap(R)["ap"], f1=m["f1"], p=m["precision"],
                      r=m["recall"], control_spec=round(sum(not r["pred"] for r in cut if r["control"]) /
                                                        max(sum(r["control"] for r in cut), 1), 4))))
