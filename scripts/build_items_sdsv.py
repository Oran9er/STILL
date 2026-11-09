"""Same-capture two-frame pairs (nothing changed) for the holistic baseline."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump  # noqa: E402

PER_VIDEO = 4


def main():
    val = json.load(open(ROOT / "ext/scenediff/scenediff_benchmark/splits/val_split.json"))["varied"]
    rng = random.Random(0)
    out = []
    for seq in val:
        d = ROOT / "data/img_sd" / seq
        for v in (1, 2):
            fr = sorted(int(p.stem.split("_")[1]) for p in d.glob(f"v{v}_*.jpg"))
            cand = [(a, b) for a in fr for b in fr if 30 <= b - a <= 90]
            rng.shuffle(cand)
            for a, b in cand[:PER_VIDEO]:
                a, b = (a, b) if rng.random() < 0.5 else (b, a)
                out.append(dict(id=f"{seq}|sv{v}|{a}|{b}", set="sdsv_train", group=seq, task="bin", domain="scene",
                                gold="no", objects=[], images=[str(d / f"v{v}_{a:04d}.jpg"), str(d / f"v{v}_{b:04d}.jpg")]))
    dump(out, ROOT / "data/items_sdsv_train.jsonl")
    print("sdsv_train", len(out))


if __name__ == "__main__":
    main()
