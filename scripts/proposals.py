"""Class-agnostic Grounding DINO proposals for every frame of the scene items."""
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, iou, load  # noqa: E402

DEV = os.environ.get("STILL_PROPS_DEV", "cuda")  # "cpu" when every GPU is busy
PROMPT = "object. bottle. box. bag. chair. cup. toy. can. book. shoe. container. food. tool. device."


def detect(proc, model, paths, props):
    for k in range(0, len(paths), 8):
        ims = [Image.open(p).convert("RGB") for p in paths[k:k + 8]]
        enc = proc(images=ims, text=[PROMPT] * len(ims), return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**enc)
        res = proc.post_process_grounded_object_detection(out, enc.input_ids, threshold=0.2, text_threshold=0.2,
                                                          target_sizes=[im.size[::-1] for im in ims])
        for p, im, r in zip(paths[k:k + 8], ims, res):
            w, h = im.size
            props[p] = [[round(1000 * b[0] / w), round(1000 * b[1] / h), round(1000 * b[2] / w), round(1000 * b[3] / h), round(float(s), 3)]
                        for b, s in zip(r["boxes"].tolist(), r["scores"].tolist())]
    return props


def main():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    name = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(DEV).eval()
    if sys.argv[1] == "--missing":
        out = ROOT / "data/props_sds_all.json"
        props = json.load(open(out))
        paths = sorted({p for f in sys.argv[2:] for it in load(f) for p in it["images"]} - set(props))
        print("missing frames", len(paths), flush=True)
        json.dump(detect(proc, model, paths, props), open(out, "w"))
        print("frames now", len(props))
        return
    items = [it for it in load(sys.argv[1]) if not it["control"]]
    if len(sys.argv) > 2:
        items = items[:int(sys.argv[2])]
    paths = sorted({p for it in items for p in it["images"]})
    props = detect(proc, model, paths, {})
    name_ = Path(sys.argv[1]).stem.replace("items_", "")
    json.dump(props, open(ROOT / f"data/props_{name_}.json", "w"))
    # recall of changed objects: a view of the object is covered by a proposal with IoU >= 0.3 / 0.5
    hit3 = hit5 = n = 0
    for it in items:
        for g in it["gt"]:
            n += 1
            best = 0.0
            for key, b in g["boxes"].items():
                v, f = int(key[0]), int(key[2:])
                p = it["images"][(f - 1) if v == 1 else it["n1"] + f - 1]
                best = max([best] + [iou(b, q[:4]) for q in props.get(p, [])])
            hit3 += best >= 0.3
            hit5 += best >= 0.5
    print(json.dumps(dict(items=len(items), frames=len(paths), proposals_per_frame=round(sum(map(len, props.values())) / max(len(props), 1), 1),
                          changed_objects=n, recall_iou03=round(hit3 / max(n, 1), 3), recall_iou05=round(hit5 / max(n, 1), 3))))


if __name__ == "__main__":
    main()
