"""Scores a verifier system on real items together with the controls (AP, F1, control specificity)."""
import os
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, load, scene_ap, scene_metrics  # noqa: E402


def main():
    import re
    tag = re.sub(r"_(ver|verA|ver2|crop|cropU|cropP|tv)_(test|dev|ctl)$", "", Path(sys.argv[1]).stem)
    by = defaultdict(list)
    for f in dict.fromkeys(sys.argv[1:3]):  # the dev call passes the same file twice: read it once (bug fixed 10-16)
        for r in load(f):
            q = r["query"]
            by[r["item"]].append(dict(change="removed" if q["video"] == 1 else "added", video=q["video"],
                                      frame=q["frame"], bbox_2d=q["bbox_2d"], conf=r["p"][1]))
    thrs = [float(x) for x in sys.argv[3:]] or [0.5, 0.7, 0.9]
    import os
    sets = (("test", ["items_sds_test.jsonl"]), ("hc", ["items_sds_hc_test.jsonl"]))
    if os.environ.get("VER_P2"):
        sets = (("test", ["p2/items_sds_test.jsonl"]), ("hc", ["items_sds_hc_test.jsonl"])) if not Path(sys.argv[1]).stem.endswith("_dev") \
            else (("dev", ["p2/items_sds_dev.jsonl"]),)
    for name, fns in sets:
        rows = [dict(id=it["id"], group=it["group"], gt=it["gt"], ignore=it["ignore"], control=it["control"],
                     pred=by.get(it["id"], [])) for fn in fns for it in load(ROOT / "data" / fn)]
        dump(rows, ROOT / f"res/{tag}_sys_{name}{'2' if os.environ.get('VER_P2') else ''}.jsonl")
        a = scene_ap(rows)
        for t in thrs:
            rt = [dict(r, pred=[p for p in r["pred"] if p["conf"] >= t]) for r in rows]
            m = scene_metrics(rt)
            ctl = [r for r in rt if r["control"]]
            print(json.dumps(dict(system=tag, set=name, n=len(rows), ap=a["ap"], f1max=a["f1max"], thr=t, f1=m["f1"],
                                  p=m["precision"], r=m["recall"],
                                  control_spec=round(sum(not r["pred"] for r in ctl) / max(len(ctl), 1), 4),
                                  control_pred_per_item=round(sum(len(r["pred"]) for r in ctl) / max(len(ctl), 1), 2))))


if __name__ == "__main__":
    main()
