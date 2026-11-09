"""Frames for the official SceneDiff protocol, decoded with ffmpeg as in the official code."""
import os
import json
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

SRC = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1])) / "ext/scenediff/scenediff_benchmark"
OUT = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1])) / "ext/sd_official_ff"


def one(seq):
    d = SRC / "data" / seq
    vids = sorted(x for x in d.iterdir() if x.suffix.lower() in (".mp4", ".mov", ".avi"))[:2]
    for k, v in enumerate(vids, 1):
        dst = OUT / seq / f"video{k}_frames"
        if dst.exists() and any(dst.iterdir()):
            continue
        dst.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            subprocess.run(["ffmpeg", "-loglevel", "quiet", "-threads", "2", "-i", str(v), "-q:v", "2", "-r", "30",
                            "-start_number", "0", f"{tmp}/%05d.jpg"], check=True)
            for i, f in enumerate(sorted(Path(tmp).glob("*.jpg"))[::30]):
                shutil.move(str(f), dst / f"{i:05d}.jpg")
    return seq, [len(list((OUT / seq / f"video{k}_frames").glob("*.jpg"))) for k in (1, 2)]


def main():
    sp = json.load(open(SRC / f"splits/{sys.argv[1]}_split.json"))
    OUT.mkdir(exist_ok=True)
    with Pool(24) as pool:
        res = pool.map(one, sp["varied"] + sp["kitchen"])
    n = [a + b for _, (a, b) in res]
    print(len(res), "pairs, frames per pair min / median / max", min(n), sorted(n)[len(n) // 2], max(n))


if __name__ == "__main__":
    main()
