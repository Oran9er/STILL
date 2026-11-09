"""Turns SceneDiff change regions into verifier queries."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_verify import Q  # noqa: E402
from still_common import ROOT, dump, iou, load  # noqa: E402


def queries(it, preds):
    out, seen = [], set()
    for p in preds:
        v, f, b = p["video"], p["frame"], p["bbox_2d"]
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        qid = f"{it['id']}|{v}:{f}|{','.join(map(str, b))}"
        if qid in seen:
            continue
        seen.add(qid)
        key = f"{v}:{f}"
        best, gj = 0.0, -1
        for j, g in enumerate(it["gt"]):
            if key in g["boxes"]:
                x = iou(b, g["boxes"][key])
                if x > best:
                    best, gj = x, j
        ign = any(g["video"] == v and g["frame"] == f and iou(b, g["bbox_2d"]) >= 0.1 for g in it["ignore"])
        gold = "changed" if best >= 0.4 else ("same" if best < 0.1 and not ign else "skip_gold")
        out.append(dict(id=qid, group=it["group"], task="bin3", gold=gold, images=it["images"], n1=it["n1"], n2=it["n2"],
                        query=dict(video=v, frame=f, bbox_2d=b, score=p["conf"]), q=Q.format(v=v, f=f, b=b, w=3 - v),
                        obj=gj if best >= 0.4 else None, item=it["id"], kind=it["gt"][gj]["kind"] if best >= 0.4 else None))
    return out


def main():
    if sys.argv[1:] == ["rc3d"]:  # RC-3D two-view pairs
        items = {it["id"]: it for it in load(ROOT / "data/rc3d/items_rc3d.jsonl")}
        qs = [q for r in load(ROOT / "res/SDIFF_rc3d_Vg.jsonl") for q in queries(items[r["id"]], r["pred"])]
        dump(qs, ROOT / "data/p2/items_vsd_rc3d.jsonl")
        print("rc3d", len(qs), "queries on", len({q["item"] for q in qs}), "items", dict(Counter(q["gold"] for q in qs)))
        return
    if sys.argv[1:] == ["sdk"]:  # SD-K kitchen test / dev
        sets = {"sdk_dev": [("sdk_dev", "data/p2/items_sdk_dev.jsonl")], "sdk_test": [("sdk_test", "data/p2/items_sdk_test.jsonl")]}
        for name, srcs in sets.items():
            qs = []
            for s, f in srcs:
                items = {it["id"]: it for it in load(ROOT / f)}
                for r in load(ROOT / f"res/SDIFF_{s}_Vg.jsonl"):
                    qs += queries(items[r["id"]], r["pred"])
            dump(qs, ROOT / f"data/p2/items_vsd_{name}.jsonl")
            print(name, len(qs), "queries on", len({q["item"] for q in qs}), "items", dict(Counter(q["gold"] for q in qs)))
        return
    sets = {"dev": [("sds_dev", "data/p2/items_sds_dev.jsonl")],
            "test": [("sds_test", "data/p2/items_sds_test.jsonl")],
            "ctl": [("sds_hc_test", "data/items_sds_hc_test.jsonl")]}
    for name, srcs in sets.items():
        qs = []
        for s, f in srcs:
            items = {it["id"]: it for it in load(ROOT / f)}
            for r in load(ROOT / f"res/SDIFF_{s}_Vg.jsonl"):
                it = items[r["id"]]
                if name == "test" or name == "dev" or it["control"]:
                    qs += queries(it, r["pred"])
        dump(qs, ROOT / f"data/p2/items_vsd_{name}.jsonl")
        print(name, len(qs), "queries on", len({q["item"] for q in qs}), "items", dict(Counter(q["gold"] for q in qs)))


if __name__ == "__main__":
    main()
