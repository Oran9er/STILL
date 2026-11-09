"""Final score s = sqrt(P_scene * P_place) * c^a; the exponent a and the threshold are chosen on dev."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, load, scene_ap, scene_metrics  # noqa: E402

A_GRID = (0.0, 0.5, 1.0, 2.0)


def main():
    tag, sdk = sys.argv[1], "--sdk" in sys.argv
    if sdk:
        sets = {"dev": ([f"res/{tag}_tv_dev.jsonl"], "data/p2/items_sdk_dev.jsonl"),
                "test": ([f"res/{tag}_tv_test.jsonl"], "data/p2/items_sdk_test.jsonl")}
    else:
        sets = {"dev": ([f"res/{tag}_tv_dev.jsonl"], "data/p2/items_sds_dev.jsonl"),
                "test": ([f"res/{tag}_tv_test.jsonl", f"res/{tag}_tv_ctl.jsonl"], "data/p2/items_sds_test.jsonl"),
                "hc": ([f"res/{tag}_tv_test.jsonl", f"res/{tag}_tv_ctl.jsonl"], "data/items_sds_hc_test.jsonl")}
    raw = {k: [r for f in fs for r in load(ROOT / f)] for k, (fs, _) in sets.items()}
    items = {k: load(ROOT / it) for k, (_, it) in sets.items()}

    def rows(k, a):
        by = collections.defaultdict(list)
        for r in raw[k]:
            q = r["query"]
            by[r["item"]].append(dict(change="removed" if q["video"] == 1 else "added", video=q["video"], frame=q["frame"],
                                      bbox_2d=q["bbox_2d"], conf=r["p"][1] * max(q["score"], 1e-6) ** a))
        return [dict(id=it["id"], group=it["group"], gt=it["gt"], ignore=it["ignore"], control=it["control"],
                     pred=by.get(it["id"], [])) for it in items[k]]
    cut = lambda R, t: [dict(r, pred=[p for p in r["pred"] if p["conf"] >= t]) for r in R]
    best = None
    for a in A_GRID:
        d = rows("dev", a)
        cs = np.array(sorted(p["conf"] for r in d for p in r["pred"]))
        for t in np.quantile(cs, np.linspace(0, 0.99, 60)):
            f = scene_metrics(cut(d, t))["f1"]
            if best is None or f > best[0]:
                best = (f, a, float(t))
    f, a, t = best
    out = {}
    for k in sets:
        R = rows(k, a)
        dump(R, ROOT / f"res/{tag}_fuse_sys_{k}.jsonl")
        m, ap = scene_metrics(cut(R, t)), scene_ap(R)
        out[k] = dict(ap=ap["ap"], f1max=ap["f1max"], f1=m["f1"], p=m["precision"], r=m["recall"], spec=m["pair_spec"])
    print(json.dumps(dict(tag=tag, a=a, thr=t, dev_f1=round(f, 4), **{k: v for k, v in out.items()})))


if __name__ == "__main__":
    main()
