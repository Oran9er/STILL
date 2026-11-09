"""Learned registration: training and test items for predicting the place frame and box."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, dump, load  # noqa: E402

MIN_SHARE = 0.3
Q = ("Look at the object in video {v}, frame {f}, inside the box {b} (0-1000 coordinates). Find the same place in "
     "video {w}: answer with the frame of video {w} that shows it best and the box of that place there, as "
     "{{\"frame\": <number>, \"bbox_2d\": [x1, y1, x2, y2]}}, or {{\"frame\": null}} if the place is not in view in "
     "any frame of video {w}. Answer with the JSON only.")


def main():
    for name in sys.argv[1:]:
        qs = load(ROOT / f"data/p2/items_{name}.jsonl")
        anc = json.load(open(ROOT / f"data/p2/pi3_anchor_{name}.json"))
        out, seen = [], set()
        for q in qs:
            if q["id"] in seen:
                continue
            seen.add(q["id"])
            v, f, b = q["query"]["video"], q["query"]["frame"], q["query"]["bbox_2d"]
            r = anc.get(q["id"])
            ok = r and r[0] is not None and r[2] >= MIN_SHARE and r[1][2] > r[1][0] and r[1][3] > r[1][1]
            tgt = {"frame": r[0], "bbox_2d": r[1]} if ok else {"frame": None}
            out.append(dict(id=q["id"], item=q["item"], group=q["group"], task="loc", images=q["images"], n1=q["n1"],
                            n2=q["n2"], query=q["query"], gold=q["gold"], kind=q.get("kind"),
                            q=Q.format(v=v, f=f, b=b, w=3 - v), loc=tgt))
        dump(out, ROOT / f"data/p2/items_loc_{name}.jsonl")
        print(name, len(out), "queries; target in view:", sum(o["loc"]["frame"] is not None for o in out),
              dict(Counter(o["gold"] for o in out)))


if __name__ == "__main__":
    main()
