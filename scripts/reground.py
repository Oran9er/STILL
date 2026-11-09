"""Re-grounds the boxes of a holistic InternVL run with the model's native grounding prompt."""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, load, scene_ap, scene_metrics  # noqa: E402

PB = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{l}</ref>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    items = {it["id"]: it for it in load(ROOT / a.items)}
    rows = load(ROOT / f"res/{a.tag}_{a.set}_Vg.jsonl")
    jobs = []
    for ri, r in enumerate(rows):
        it = items[r["id"]]
        for pi, p in enumerate(r["pred"] or []):
            v, f = p["video"], p["frame"]
            if not (1 <= f <= (it["n1"] if v == 1 else it["n2"])):
                continue
            jobs.append((ri, pi, it["images"][(f - 1) if v == 1 else it["n1"] + f - 1], p.get("label") or "object"))
    print(len(rows), "rows,", len(jobs), "entries", flush=True)
    new = {}
    for b in range(0, len(jobs), a.bs):
        batch = jobs[b:b + a.bs]
        texts, ims = [], []
        for _, _, path, l in batch:
            im = Image.open(path).convert("RGB")
            s = min(1.0, (448 * 1024 / (im.width * im.height)) ** 0.5)
            ims.append(im.resize((round(im.width * s), round(im.height * s))))
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PB.format(l=l)}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = proc(text=texts, images=ims, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=48, do_sample=False)
        for (ri, pi, _, l), seq in zip(batch, o[:, enc["input_ids"].shape[1]:]):
            t = proc.tokenizer.decode(seq, skip_special_tokens=True)
            m = re.search(r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", t)
            if m:
                new[(ri, pi)] = [float(x) for x in m.groups()]
        if (b // a.bs) % 50 == 0:
            print(f"[reground] {b}/{len(jobs)} {t[:80]!r}", flush=True)
    for ri, r in enumerate(rows):
        r["pred"] = [dict(p, bbox_2d=new[(ri, pi)]) for pi, p in enumerate(r["pred"] or []) if (ri, pi) in new] \
            if r["pred"] is not None else None
    stem = f"{a.tag}G_{a.set}_Vg"
    with open(ROOT / f"res/{stem}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    m = scene_metrics(rows)
    summ = dict(tag=a.tag + "G", set=a.set, n=len(rows), f1_03=m["f1"], p_03=m["precision"], r_03=m["recall"],
                pair_spec=m["pair_spec"], recall_by_kind=m["recall_by_kind"], regrounded=len(new), entries=len(jobs),
                **{k: v for k, v in scene_ap(rows).items() if k in ("ap", "f1max")})
    json.dump(summ, open(ROOT / f"res/{stem}.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
