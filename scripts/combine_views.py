"""Combines the scene-view and place-view answers of each query: sqrt(P_scene * P_place)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import dump, load  # noqa: E402

crop = {r["id"]: r["p"][1] for r in load(sys.argv[2])}
rows = load(sys.argv[1])
n = 0
for r in rows:
    pc = crop.get(r["id"])
    if pc is not None:
        n += 1
        r["p"] = [r["p"][0], math.sqrt(r["p"][1] * pc), r["p"][2]]
dump(rows, sys.argv[3])
print(json.dumps(dict(out=sys.argv[3], queries=len(rows), with_place_view=n)))
