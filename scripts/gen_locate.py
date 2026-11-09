"""Learned registration: inference of the place frame and box."""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import content, encode, images, iou, load  # noqa: E402


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None, None
    f, b = d.get("frame"), d.get("bbox_2d")
    if f is None or not isinstance(b, list) or len(b) != 4:
        return None, None
    b = [int(max(0, min(1000, x))) for x in b]
    return (int(f), b) if b[2] > b[0] and b[3] > b[1] else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    items = load(a.items)
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        items = items[k::n]
    out, agree = {}, []
    for b0 in range(0, len(items), a.bs):
        batch = items[b0:b0 + a.bs]
        texts = [proc.apply_chat_template([{"role": "user", "content": content(it)}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False) for it in batch]
        enc = encode(proc, texts, [im for it in batch for im in images(it)], "V", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=40, do_sample=False)
        for it, ids in zip(batch, g[:, enc["input_ids"].shape[1]:]):
            f, b = parse(proc.tokenizer.decode(ids, skip_special_tokens=True))
            if f is not None:  # valid-frame decoding: a frame of the other capture
                f = min(max(f, 1), it["n2"] if it["query"]["video"] == 1 else it["n1"])
            out[it["id"]] = [f, b, 1.0 if f is not None else 0.0]
            t = it["loc"]
            agree.append((f is None) == (t["frame"] is None) and (f is None or (f == t["frame"], iou(b, t["bbox_2d"]))))
        if (b0 // a.bs) % 50 == 0:
            print(f"[loc] {b0}/{len(items)}", flush=True)
    dst = a.out.replace(".json", f"_s{a.shard.split('/')[0]}.json") if a.shard else a.out
    json.dump(out, open(dst, "w"))
    vis = [x for x in agree if isinstance(x, tuple)]
    print(json.dumps(dict(n=len(out), in_view_agree=round(sum(bool(x) for x in agree) / max(len(agree), 1), 4),
                          same_frame=round(sum(x[0] for x in vis) / max(len(vis), 1), 4),
                          median_iou_same_frame=sorted(x[1] for x in vis if x[0])[len([1 for x in vis if x[0]]) // 2] if any(x[0] for x in vis) else None)))


if __name__ == "__main__":
    main()
