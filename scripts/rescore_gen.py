"""Re-scores saved generations of the holistic baseline."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, iou, list_metrics, load, parse_list, parse_scene, scene_ap, scene_metrics  # noqa: E402


def main():
    for f in sys.argv[1:]:
        rows = load(f)
        scene = "gt" in rows[0]
        for r in rows:
            old_pred = r.get("pred") or []
            r["pred"] = (parse_scene if scene else parse_list)(r["text"])
            if scene and r["pred"] and len(old_pred) == len(r["pred"]) and all("conf" in p for p in old_pred):
                for p, q in zip(r["pred"], old_pred):  # keep the confidences stored at generation time
                    p["conf"] = q["conf"]
        fn = scene_metrics if scene else list_metrics
        m3, m5 = fn(rows, 0.3), fn(rows, 0.5)
        old = json.load(open(f.replace(".jsonl", ".json")))
        summ = dict(tag=old["tag"], set=old["set"], arm=old["arm"], task=old["task"], n=len(rows), f1_03=m3["f1"],
                    p_03=m3["precision"], r_03=m3["recall"], f1_05=m5["f1"], kind_acc=m3["kind_acc"],
                    pair_sens=m3["pair_sens"], pair_spec=m3["pair_spec"], unparsable=m3["unparsable"])
        if scene:
            summ["recall_by_kind"] = m3["recall_by_kind"]
            summ["control_spec"] = round(sum(not r["pred"] for r in rows if r["control"]) / max(sum(r["control"] for r in rows), 1), 4)
            for lab, sel in (("retail", lambda g: g.startswith("store")), ("other", lambda g: not g.startswith("store"))):
                m = scene_metrics([r for r in rows if sel(r["group"]) and not r["control"]], 0.3)
                summ[f"{lab}_f1"] = m["f1"]
                summ[f"{lab}_pr"] = [m["precision"], m["recall"]]
            items = {}  # protocol v2 sets live in data/p2 and are supersets of the old ones
            for src in (ROOT / f"data/items_{old['set']}.jsonl", ROOT / f"data/p2/items_{old['set']}.jsonl"):
                if src.exists():
                    items.update({it["id"]: it for it in load(src)})
            gh = hit = npred = 0  # controls: do the predictions land on the objects that changed BETWEEN the captures?
            for r in rows:
                if not r["control"]:
                    continue
                ghosts = items.get(r["id"], {}).get("ghost", [])
                gh += len(ghosts)
                npred += len(r["pred"] or [])
                for g in ghosts:
                    hit += any(f"{p['video']}:{p['frame']}" in g["boxes"]
                               and iou(p["bbox_2d"], g["boxes"][f"{p['video']}:{p['frame']}"]) >= 0.3 for p in (r["pred"] or []))
            summ["control_ghost_recall"] = round(hit / max(gh, 1), 4)
            summ["control_pred_per_item"] = round(npred / max(sum(r["control"] for r in rows), 1), 2)
            if any("conf" in p for r in rows for p in (r["pred"] or [])):
                summ.update({k: v for k, v in scene_ap(rows).items() if k in ("ap", "f1max")})
            summ["n_pred_per_item"] = round(sum(len(r["pred"] or []) for r in rows if not r["control"]) / max(sum(not r["control"] for r in rows), 1), 2)
        json.dump(summ, open(f.replace(".jsonl", ".json"), "w"), indent=1)
        print(json.dumps(summ))


if __name__ == "__main__":
    main()
