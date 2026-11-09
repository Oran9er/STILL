"""Learned registration: inference of the refined place box."""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_refine import from_region  # noqa: E402
from still_common import content, encode, images, iou, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--coarse", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pi3", default="")
    ap.add_argument("--shard", default="")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    items = load(a.items)
    coarse = json.load(open(a.coarse))
    if a.shard:  # shard k keeps its refine items plus every k-th coarse answer that has no refine item
        k, n = map(int, a.shard.split("/"))
        every = {it["id"] for it in items}
        items = items[k::n]
        ids = {it["id"] for it in items}
        coarse = {i: r for j, (i, r) in enumerate(sorted(coarse.items())) if i in ids or (i not in every and j % n == k)}
    out = {i: list(r) for i, r in coarse.items()}
    bad = 0
    for b0 in range(0, len(items), a.bs):
        batch = items[b0:b0 + a.bs]
        texts = [proc.apply_chat_template([{"role": "user", "content": content(it)}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False) for it in batch]
        enc = encode(proc, texts, [im for it in batch for im in images(it)], "V", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=32, do_sample=False)
        for it, ids_ in zip(batch, g[:, enc["input_ids"].shape[1]:]):
            m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", proc.tokenizer.decode(ids_, skip_special_tokens=True))
            if not m:
                bad += 1
                continue
            b = [min(1000, int(x)) for x in m.groups()]
            if b[2] <= b[0] or b[3] <= b[1]:
                bad += 1
                continue
            W, H = it["size"]
            out[it["id"]] = [it["frame"], from_region(b, it["region"], W, H), 1.0]
        if (b0 // a.bs) % 50 == 0:
            print(f"[ref] {b0}/{len(items)}", flush=True)
    dst = a.out.replace(".json", f"_s{a.shard.split('/')[0]}.json") if a.shard else a.out
    json.dump(out, open(dst, "w"))
    summ = dict(n=len(items), unparsed=bad)
    if a.pi3:
        p3 = json.load(open(a.pi3))
        pc, pr = [], []
        for it in items:
            t = p3.get(it["id"])
            if t and t[0] is not None and t[2] >= 0.3 and t[0] == it["frame"]:
                pc.append(iou(coarse[it["id"]][1], t[1]))
                pr.append(iou(out[it["id"]][1], t[1]))
        med = lambda x: round(sorted(x)[len(x) // 2], 3) if x else None
        summ.update(same_frame=len(pc), iou_coarse_median=med(pc), iou_refined_median=med(pr),
                    iou_coarse_ge05=round(sum(x >= 0.5 for x in pc) / max(len(pc), 1), 3),
                    iou_refined_ge05=round(sum(x >= 0.5 for x in pr) / max(len(pr), 1), 3))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
