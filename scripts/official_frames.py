"""Frames for the official SceneDiff protocol, read from the original videos with OpenCV."""
import os
import json
import sys
from multiprocessing import Pool
from pathlib import Path

SRC = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1])) / "ext/scenediff/scenediff_benchmark"
OUT = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1])) / "ext/sd_official"


def count(path):
    """Frame count by grabbing (CAP_PROP_FRAME_COUNT is unreliable), as build_sd.n_frames."""
    import cv2
    c = cv2.VideoCapture(str(path))
    n = 0
    while c.grab():
        n += 1
    c.release()
    return n


def n_annot(d, k, v, n_o):
    """30-fps frame count: the annotated video's if present, else the original's duration at 30 fps."""
    import cv2
    a = d / f"video{k}.mp4"
    if a.exists():
        return count(a)
    c = cv2.VideoCapture(str(v))
    fps = c.get(cv2.CAP_PROP_FPS) or 30.0
    c.release()
    return max(1, round(n_o * 30.0 / fps))


def one(seq):
    import cv2
    cv2.setNumThreads(1)
    d = SRC / "data" / seq
    origs = sorted(x for x in d.iterdir() if x.suffix.lower() in (".mp4", ".mov", ".avi"))[:2]
    for k, v in enumerate(origs, 1):
        dst = OUT / seq / f"video{k}_frames"
        if dst.exists() and any(dst.iterdir()):
            continue
        dst.mkdir(parents=True, exist_ok=True)
        n_o = count(v)
        n_a = n_annot(d, k, v, n_o)
        want = {min(n_o - 1, round(a * n_o / n_a)): j for j, a in enumerate(range(0, n_a, 30))}
        c = cv2.VideoCapture(str(v))
        i = 0
        while True:
            ok, fr = c.read()
            if not ok or i > max(want):
                break
            if i in want:
                cv2.imwrite(str(dst / f"{want[i]:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            i += 1
        c.release()
    return seq, [len(list((OUT / seq / f"video{k}_frames").glob("*.jpg"))) for k in (1, 2)]


def main():
    split = sys.argv[1]
    sp = json.load(open(SRC / f"splits/{split}_split.json"))
    seqs = sp["varied"] + sp["kitchen"]
    with Pool(24) as pool:
        res = pool.map(one, seqs)
    n = [a + b for _, (a, b) in res]
    print(len(res), "pairs, frames per pair min / median / max", min(n), sorted(n)[len(n) // 2], max(n))


if __name__ == "__main__":
    main()
