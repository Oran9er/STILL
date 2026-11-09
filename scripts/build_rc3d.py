"""RC-3D pairs as scene items (one image per capture)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump  # noqa: E402

SRC = ROOT / "ext/rc3d"


def main():
    out = []
    for part in ("part1", "part2", "part3", "part4"):
        d = json.load(open(SRC / part / "coco_annotations.json"))
        ims, ans = d["images"][0::3], d["images"][2::3]
        a1, a2 = d["annotations"][0::3], d["annotations"][2::3]
        for i1, i2, b1, b2 in zip(ims, ans, a1, a2):
            assert b1["image_id"] == i1["id"] and b2["image_id"] == i2["id"]
            box = lambda b, im: [round(1000 * b["bbox"][0] / im["width"]), round(1000 * b["bbox"][1] / im["height"]),
                                 round(1000 * (b["bbox"][0] + b["bbox"][2]) / im["width"]),
                                 round(1000 * (b["bbox"][1] + b["bbox"][3]) / im["height"])]
            g = Path(i1["file_name"]).stem.rsplit("_", 1)[0]
            out.append(dict(id=f"{part}/{g}", set="rc3d", group=f"{part}/{g}", task="scene", gold="yes",
                            images=[str(SRC / part / i1["file_name"]), str(SRC / part / i2["file_name"])], n1=1, n2=1,
                            entries=[], gt=[dict(label="object", kind="moved", boxes={"1:1": box(b1, i1), "2:1": box(b2, i2)})],
                            ignore=[], control=False))
    (ROOT / "data/rc3d").mkdir(exist_ok=True)
    dump(out, ROOT / "data/rc3d/items_rc3d.jsonl")
    print(len(out), "pairs")


if __name__ == "__main__":
    main()
