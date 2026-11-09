"""Scene bootstrap of F1 when every run uses its own threshold."""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import load, scene_metrics  # noqa: E402


def counts(spec):
    path, _, t = spec.partition(":")
    t = float(t) if t else None
    by = collections.defaultdict(lambda: np.zeros(3))
    for r in load(path):
        pred = r["pred"]
        if t is not None and pred:
            pred = [p for p in pred if p.get("conf", 1.0) >= t]
        m = scene_metrics([dict(r, pred=pred)])
        by[r["group"]] += (m["tp"], m["fp"], m["fn"])
    return by


def f1(c):
    tp, fp, fn = c
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def main():
    fams = [[counts(s) for s in arg.split(",")] for arg in sys.argv[1:3]]
    groups = sorted(fams[0][0])
    rng = np.random.default_rng(0)
    mean = lambda fam, gs: np.mean([f1(sum(run[g] for g in gs)) for run in fam])
    d = []
    for _ in range(2000):
        gs = rng.choice(groups, len(groups))
        d.append(mean(fams[1], gs) - mean(fams[0], gs))
    a, b = mean(fams[0], groups), mean(fams[1], groups)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"A {a:.3f}  B {b:.3f}  B - A {b - a:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({len(groups)} scenes)")


if __name__ == "__main__":
    main()
