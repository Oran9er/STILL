"""Scene bootstrap of AP and F1 for paired comparisons.

boot_fast.py A1,A2,... B1,B2,...
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import iou, load, parse_scene  # noqa: E402


def cands(rows):
    """per group: (array of conf, array of hit), n_gt"""
    by = defaultdict(lambda: ([], [], 0))
    for r in rows:
        pred = r.get("pred")
        if "text" in r:  # holistic outputs: re-parse, keep stored confidences
            new = parse_scene(r["text"]) or []
            if pred and len(pred) == len(new):
                for p, q in zip(new, pred):
                    if "conf" in q:
                        p["conf"] = q["conf"]
            pred = new
        c, h, n = by[r["group"]]
        used = [False] * len(r["gt"])
        for p in sorted(pred or [], key=lambda p: -p.get("conf", 0.0)):
            key = f"{p['video']}:{p['frame']}"
            best, bj = 0.0, -1
            for j, g in enumerate(r["gt"]):
                if key in g["boxes"]:
                    v = iou(p["bbox_2d"], g["boxes"][key])
                    if v > best:
                        best, bj = v, j
            if best >= 0.3:
                if used[bj]:
                    continue
                used[bj] = True
                c.append(p.get("conf", 0.0)); h.append(1)
            elif any(g["video"] == p["video"] and g["frame"] == p["frame"] and iou(p["bbox_2d"], g["bbox_2d"]) >= 0.3
                     for g in r["ignore"]):
                continue
            else:
                c.append(p.get("conf", 0.0)); h.append(0)
        by[r["group"]] = (c, h, n + len(r["gt"]))
    return {g: (np.array(c, float), np.array(h, float), n) for g, (c, h, n) in by.items()}


def score(cs, groups, thr=None):
    c = np.concatenate([cs[g][0] for g in groups])
    h = np.concatenate([cs[g][1] for g in groups])
    n = sum(cs[g][2] for g in groups)
    o = np.argsort(-c, kind="mergesort")
    h = h[o]
    tp = np.cumsum(h)
    fp = np.cumsum(1 - h)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(n, 1)
    # all-point interpolated AP and best F1 over thresholds, as in still_common.scene_ap
    pint = np.maximum.accumulate(prec[::-1])[::-1]
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * pint))
    f1 = float(np.max(2 * prec * rec / np.maximum(prec + rec, 1e-9))) if len(h) else 0.0
    keep = c[o] >= (thr if thr is not None else -1)
    t, f = h[keep].sum(), (1 - h[keep]).sum()
    f1t = float(2 * t / max(2 * t + f + (n - t), 1))
    return ap, f1, f1t


def main():
    # a file may carry its own threshold for F1 @ thr: path:thr (overrides BOOT_THR_A / BOOT_THR_B)
    specs = [[(f.rsplit(":", 1)[0], float(f.rsplit(":", 1)[1])) if ":" in f else (f, None) for f in s.split(",")]
             for s in sys.argv[1:]]
    fams = [[cands(load(f)) for f, _ in sp] for sp in specs]
    groups = sorted(set.intersection(*[set(c) for fam in fams for c in fam]))
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(groups), len(groups)) for _ in range(2000)]
    est, boot = [], []
    thrs = [os.environ.get("BOOT_THR_A"), os.environ.get("BOOT_THR_B")]
    for fam, sp, th in zip(fams, specs, thrs):
        th = float(th) if th else None
        ths = [t if t is not None else th for _, t in sp]
        est.append(np.mean([score(c, groups, t) for c, t in zip(fam, ths)], 0))
        boot.append(np.array([np.mean([score(c, [groups[i] for i in d], t) for c, t in zip(fam, ths)], 0) for d in draws]))
    for s, e, b in zip(sys.argv[1:], est, boot):
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        print(f"{len(s.split(','))} runs: AP {e[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]  F1max {e[1]:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]  "
              f"F1@thr {e[2]:.3f} [{lo[2]:.3f}, {hi[2]:.3f}]  ({len(groups)} scenes)")
    if len(fams) == 2:
        d = boot[1] - boot[0]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        e = est[1] - est[0]
        print(f"difference B - A: AP {e[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}] p(<=0)={np.mean(d[:, 0] <= 0):.4f}  "
              f"F1max {e[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}] p(<=0)={np.mean(d[:, 1] <= 0):.4f}  "
              f"F1@thr {e[2]:+.3f} [{lo[2]:+.3f}, {hi[2]:+.3f}] p(<=0)={np.mean(d[:, 2] <= 0):.4f}")


if __name__ == "__main__":
    main()
