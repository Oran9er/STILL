"""First-vs-second-half controls: the two halves of one capture, in which nothing changed."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sds import K, MIN_AREA, ROOT, SRC, Safe, boxes, frames_of, name, pick  # noqa: E402


def one(args):
    seq, sp = args
    seg = Safe(open(SRC / "data" / seq / "segments.pkl", "rb")).load()
    out = []
    d = ROOT / "data/img_sd" / seq
    for v in (1, 2):
        fr = frames_of(seq, v)
        if len(fr) < 2 * K:
            continue
        a, b = pick(fr[:len(fr) // 2], K), pick(fr[len(fr) // 2:], K)
        if v == 2:  # second capture: also use the reverse temporal order, so that order carries no cue
            a, b = b, a
        bx = boxes(seg, v, a + b)
        ghost = []
        for oid, f in bx.items():
            views = {(f"1:{a.index(x) + 1}" if x in a else f"2:{b.index(x) + 1}"): bb for x, (bb, ar) in f.items() if ar >= MIN_AREA}
            if views:
                ghost.append(dict(label=name(seg["objects"][oid]["label"]), boxes=views))
        out.append(dict(id=f"{seq}|h{v}", set=f"sds_hc_{sp}", group=seq, task="scene", gold="no",
                        images=[str(d / f"v{v}_{x:04d}.jpg") for x in a] + [str(d / f"v{v}_{x:04d}.jpg") for x in b],
                        n1=K, n2=K, entries=[], gt=[], ignore=[], control=True, ghost=ghost))
    return out


def main():
    for sp, name_ in (("test", "test"), ("val", "train")):
        seqs = [s for s in json.load(open(SRC / f"splits/{sp}_split.json"))["varied"] if (ROOT / "data/img_sd" / s).exists()]
        with Pool(32) as pool:
            items = [it for r in pool.map(one, [(s, name_) for s in seqs]) for it in r]
        with open(ROOT / f"data/items_sds_hc_{name_}.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"sds_hc_{name_}", len(items), "items from", len({it['group'] for it in items}), "scenes; ghost objects",
              sum(len(it["ghost"]) for it in items))


if __name__ == "__main__":
    main()
