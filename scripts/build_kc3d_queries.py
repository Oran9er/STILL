"""Verifier queries on KC-3D: Grounding DINO boxes and change-mask components."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_verify import Q  # noqa: E402
from still_common import ROOT, dump, iou, load  # noqa: E402

props = json.load(open(ROOT / "data/props_sds_all.json"))
rng = random.Random(0)
pos, neg = [], []
for it in load(ROOT / "data/kc3d/items_kc3d_train.jsonl"):
    for v in (1, 2):
        img = it["images"][v - 1]
        gts = [g["boxes"][f"{v}:1"] for g in it["gt"] if f"{v}:1" in g["boxes"]]
        cand = [(g, "changed") for g in gts]
        for p in props.get(img, []):
            if p[4] < 0.25:
                continue
            b = p[:4]
            best = max([iou(b, g) for g in gts] + [0.0])
            if best >= 0.4:
                cand.append((b, "changed"))
            elif best < 0.1:
                cand.append((b, "same"))
        seen = set()
        for b, lab in cand:
            k = tuple(b)
            if k in seen or b[2] <= b[0] or b[3] <= b[1]:
                continue
            seen.add(k)
            q = dict(id=f"{it['id']}|{v}:1|{','.join(map(str, b))}", group=it["group"], task="bin3", gold=lab,
                     images=it["images"], n1=1, n2=1, query=dict(video=v, frame=1, bbox_2d=b, score=1.0),
                     q=Q.format(v=v, f=1, b=b, w=3 - v), obj=None, item=it["id"], kind="moved" if lab == "changed" else None)
            (pos if lab == "changed" else neg).append(q)
neg = rng.sample(neg, min(len(neg), 3 * len(pos)))
qs = pos + neg
rng.shuffle(qs)
dump(qs, ROOT / "data/p2/items_kc3d_train.jsonl")
print(len(qs), "queries", dict(Counter(q["gold"] for q in qs)))
