"""SIFT places for the registration ablation, in the pi3_anchor.py output format."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_verify import MIN_PTS, anchor, in_box, in_mask, matches  # noqa: E402
from still_common import ROOT, load  # noqa: E402


def main():
    for s in sys.argv[1:]:
        qs = load(ROOT / f"data/p2/items_{s}.jsonl")
        pairs, todo = {}, set()
        for q in qs:
            v, f, n1 = q["query"]["video"], q["query"]["frame"], q["n1"]
            src = q["images"][(f - 1) if v == 1 else n1 + f - 1]
            for o in (q["images"][n1:] if v == 1 else q["images"][:n1]):
                todo.add((src, o))
        with Pool(48) as pool:
            M = dict(pool.imap_unordered(matches, sorted(todo), chunksize=8))
        out = {}
        for q in qs:
            v, f, n1, b = q["query"]["video"], q["query"]["frame"], q["n1"], q["query"]["bbox_2d"]
            src = q["images"][(f - 1) if v == 1 else n1 + f - 1]
            best = (0, None)
            for g_i, o in enumerate(q["images"][n1:] if v == 1 else q["images"][:n1]):
                m = M.get((src, o))
                if m is None:
                    continue
                k = in_box(m[0], b)
                if k > best[0]:
                    sel = in_mask(m[0], b)
                    best = (k, [g_i + 1, anchor(np.array(m[0])[sel], np.array(m[1])[sel], b), k])
            if best[0] >= MIN_PTS:
                out[q["id"]] = best[1]
        json.dump(out, open(ROOT / f"data/p2/reg_sift_anchor_{s}.json", "w"))
        print(s, len(qs), "queries,", len(todo), "frame pairs ->", len(out), "anchored", flush=True)


if __name__ == "__main__":
    main()
