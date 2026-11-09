"""Grounded versions of the two-frame items (list of changes with boxes) for the holistic baseline."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, load  # noqa: E402

W, H = 540, 960


def norm(box):
    x0, y0, x1, y1 = box
    return [round(1000 * x0 / W), round(1000 * y0 / H), round(1000 * x1 / W), round(1000 * y1 / H)]


def name(label):
    s = re.sub(r"_\d+$", "", label.strip()).replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", s) or "object"


def main():
    match = json.load(open(ROOT / "data/sd_match.json"))
    rows = {(seq, r["f1"], r["f2"]): (r, m["meta"]) for seq, m in match.items() for r in m["rows"]}
    for pre in ("sd", "sdlo"):
        for sp in ("test", "train"):
            out = []
            for it in load(ROOT / f"data/items_{pre}_{sp}h.jsonl"):  # the h files carry the homography H (arm P)
                seq, f1, f2 = it["id"].split("|")
                row, meta = rows[(seq, int(f1), int(f2))]
                keep = {(o["view"], tuple(o["box"])) for o in it["objects"]}
                ents = sorted(({"change": o["kind"], "image": o["view"], "bbox_2d": norm(o["box"]), "label": name(o["label"])}
                               for o in it["objects"]), key=lambda e: (e["image"], e["bbox_2d"][1], e["bbox_2d"][0]))
                ign = [{"image": o["view"], "bbox_2d": norm(o["box"])} for o in row["obs"]
                       if (o["view"], tuple(o["box"])) not in keep]
                out.append(dict(it, set=f"{pre}l_{sp}", task="list", entries=ents, ignore=ign))
            dump(out, ROOT / f"data/items_{pre}l_{sp}.jsonl")
            if sp == "train":  # clean targets only: pairs whose visible changed objects are all counted
                dump([o for o in out if not o["ignore"]], ROOT / f"data/items_{pre}lc_{sp}.jsonl")
            print(f"{pre}l_{sp}", len(out), "entries", sum(len(o["entries"]) for o in out),
                  "items with ignore regions", sum(bool(o["ignore"]) for o in out),
                  "clean yes items", sum(bool(o["entries"]) and not o["ignore"] for o in out))


def same_capture():
    out = [dict(it, set="sdsvl_train", task="list", entries=[], ignore=[]) for it in load(ROOT / "data/items_sdsv_trainh.jsonl")]
    dump(out, ROOT / "data/items_sdsvl_train.jsonl")
    print("sdsvl_train", len(out))


if __name__ == "__main__":
    same_capture()
    main()
