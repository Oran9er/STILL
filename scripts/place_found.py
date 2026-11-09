"""Learned registration: how often a learned place matches the pi^3 place, by change type."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, iou, load  # noqa: E402

P3 = json.load(open(ROOT / "data/p2/pi3_anchor_vsd_test.json"))
Q = {r["id"]: r for r in load(ROOT / "data/p2/items_vsd_test.jsonl")}
for name in sys.argv[1:]:
    R = json.load(open(ROOT / f"data/p2/{name}_vsd_test.json"))
    st = collections.defaultdict(collections.Counter)
    for i, q in Q.items():
        if q["gold"] not in ("changed", "same"):
            continue
        t = P3.get(i)
        if not t or t[0] is None or t[2] < 0.3:
            continue
        k = "unchanged" if q["gold"] == "same" else "changed"
        r = R.get(i)
        st[k]["n"] += 1
        if r and r[0] is not None and r[0] == t[0]:
            st[k]["frame"] += 1
            st[k]["found"] += iou(r[1], t[1]) >= 0.5
    print(name, " | ".join(f"{k}: found {c['found'] / c['n']:.3f}, same frame {c['frame'] / c['n']:.3f} (n {c['n']})"
                           for k, c in sorted(st.items())))
