"""Learned registration: RL of the refinement policy with an IoU and a verification reward."""
import argparse
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_refine import from_region  # noqa: E402
from still_common import MAXPIX, ROOT, content, encode, images, iou, load  # noqa: E402

LORA_TARGETS = (r"model\.language_model\.layers\.\d+\.(mlp\.(gate_proj|up_proj|down_proj)|"
                r"self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))")
QC = ("Image 1 shows an object (inside the red box) in one capture of a scene. Image 2 shows the same place, marked by "
      "the red box, in the other capture. Answer \"same\" if this object is still at this place in image 2, or "
      "\"changed\" if it is not there (it was removed or moved away; if image 1 is from the later capture: it was not "
      "there before). Answer with one word.")  # build_crops.Q
GROW, MIN_SIDE, MAX_PIX = 1.6, 0.15, 384  # build_crops.crop


def place_crop(path, b):
    """build_crops.crop in memory, then still_common.load_image's resizing."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    cx, cy = (b[0] + b[2]) / 2000 * W, (b[1] + b[3]) / 2000 * H
    w = max((b[2] - b[0]) / 1000 * W * GROW, MIN_SIDE * W)
    h = max((b[3] - b[1]) / 1000 * H * GROW, MIN_SIDE * H)
    x0, y0, x1, y1 = max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)
    ImageDraw.Draw(im).rectangle([b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H], outline="red", width=3)
    c = im.crop((int(x0), int(y0), int(x1), int(y1)))
    s = min(1.0, MAX_PIX / max(c.size))
    c = c.resize((max(28, int(c.size[0] * s)), max(28, int(c.size[1] * s))))
    s = min(1.0, math.sqrt(MAXPIX / (c.size[0] * c.size[1])))
    wq, hq = max(32, round(c.size[0] * s / 32) * 32), max(32, round(c.size[1] * s / 32) * 32)
    return c if (wq, hq) == c.size else c.resize((wq, hq), Image.BICUBIC)


def parse_box(text):
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if not m:
        return None
    b = [min(1000, int(x)) for x in m.groups()]
    return b if b[2] > b[0] and b[3] > b[1] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train", default=str(ROOT / "data/p2/items_rlref_train.jsonl"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--items", type=int, default=4, help="items per GPU per step")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--hit", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--gate", type=float, default=0.3)
    ap.add_argument("--gate-same", action="store_true", help="also gate the verification term for unchanged items (v1-v2b)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed * 100 + rank)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=LORA_TARGETS))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
    data = load(a.train)
    pools = {"changed": [x for x in data if x["label"] == "changed"], "same": [x for x in data if x["label"] == "same"]}
    rng = random.Random(a.seed * 1000 + rank)
    tok = proc.tokenizer
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    eos_t = torch.tensor(eos, device=dev)
    wid = [sorted({tok(w, add_special_tokens=False)["input_ids"][0], tok(w.capitalize(), add_special_tokens=False)["input_ids"][0]})
           for w in ("same", "changed", "unseen")]
    if rank == 0:
        print(f"[rl] {len(pools['changed'])} changed + {len(pools['same'])} unchanged items; G {a.gen}; world {world}; "
              f"hit {a.hit} lam {a.lam} gate {a.gate}", flush=True)

    def verify(qcrop, crops, label):
        """P(label) of the frozen verifier (adapter off) for each place crop."""
        qim = images({"task": "crop", "images": [qcrop]})[0]
        txt = proc.apply_chat_template([{"role": "user", "content": content({"task": "crop", "q": QC})}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ims = [x for c in crops for x in (qim, c)]
        enc = encode(proc, [txt] * len(crops), ims, "V", padding=True).to(dev)
        with torch.no_grad(), model.disable_adapter():
            lg = model(**enc).logits[:, -1].float()
        pr = torch.softmax(lg, -1)
        pw = torch.stack([pr[:, i].sum(-1) for i in wid], -1)
        pw = pw / pw.sum(-1, keepdim=True).clamp(min=1e-9)
        return pw[:, 1] if label == "changed" else pw[:, 0]

    log = defaultdict(list)
    t0 = time.time()
    for step in range(a.steps):
        batch = [rng.choice(pools["changed" if k % 2 == 0 else "same"]) for k in range(a.items)]
        opt.zero_grad(set_to_none=True)
        n_tok = 0
        for it in batch:
            txt = proc.apply_chat_template([{"role": "user", "content": content(it)}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            enc = encode(proc, [txt], images(it), "V").to(dev)
            n_p = enc["input_ids"].shape[1]
            model.eval()
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=a.temp, top_p=1.0, top_k=0, max_new_tokens=32,
                                     num_return_sequences=a.gen, eos_token_id=eos, pad_token_id=tok.pad_token_id)
            comp = out[:, n_p:]
            texts = tok.batch_decode(comp, skip_special_tokens=True)
            W, H = it["size"]
            r = torch.full((a.gen,), -0.5, device=dev)
            ious, gated, boxes = [0.0] * a.gen, [], {}
            for k, t in enumerate(texts):
                b = parse_box(t)
                if b is None:
                    continue
                fb = from_region(b, it["region"], W, H)
                if fb[2] <= fb[0] or fb[3] <= fb[1]:
                    continue
                ious[k] = iou(fb, it["tgt"])
                r[k] = ious[k] + a.hit * (ious[k] >= 0.5)
                if a.lam > 0 and (ious[k] >= a.gate or (it["label"] == "same" and not a.gate_same)):
                    # for an unchanged object a wrong place can only read "changed" (penalised), so no gate is needed
                    gated.append(k)
                    boxes[k] = fb
            if gated:
                pv = verify(it["qcrop"], [place_crop(it["dst"], boxes[k]) for k in gated], it["label"])
                for j, k in enumerate(gated):
                    r[k] += a.lam * pv[j]
                log[f"pver_{it['label']}"].append(float(pv.mean()))
            adv = (r - r.mean()) / (r.std() + 0.1)
            log[f"iou_{it['label']}"].append(sum(ious) / a.gen)
            log[f"hit_{it['label']}"].append(sum(x >= 0.5 for x in ious) / a.gen)
            log["bad"].append(float((r <= -0.5).float().mean()))
            log["r"].append(float(r.mean()))
            if adv.abs().max() < 1e-6:
                continue
            model.train()
            cmask = torch.ones_like(comp)
            for i in range(comp.shape[0]):
                hit = (comp[i].unsqueeze(-1) == eos_t).any(-1).nonzero()
                if len(hit):
                    cmask[i, hit[0, 0] + 1:] = 0
            for c0 in range(0, comp.shape[0], a.chunk):
                cc, cm, ad = comp[c0:c0 + a.chunk], cmask[c0:c0 + a.chunk], adv[c0:c0 + a.chunk]
                if ad.abs().max() < 1e-6:
                    continue
                n = cc.shape[0]
                ids = torch.cat([enc["input_ids"].repeat(n, 1), cc], 1)
                att = torch.cat([enc["attention_mask"].repeat(n, 1), cm], 1)
                o = model(input_ids=ids, attention_mask=att, pixel_values=enc["pixel_values"].repeat(n, 1),
                          image_grid_thw=enc["image_grid_thw"].repeat(n, 1),
                          mm_token_type_ids=(ids == model.config.image_token_id).to(torch.int32))
                lp = torch.log_softmax(o.logits[:, n_p - 1:-1].float(), -1).gather(-1, cc.unsqueeze(-1)).squeeze(-1)
                loss = -(ad[:, None] * lp * cm).sum()
                n_tok += int(cm.sum())
                loss.backward()
                del o, lp, loss
        tot = torch.tensor([float(n_tok)], device=dev)
        dist.all_reduce(tot)
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            p.grad /= max(tot.item(), 1.0)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 10 == 0 or step < 3:
            keys = sorted(["r", "bad", "iou_changed", "iou_same", "hit_changed", "hit_same", "pver_changed", "pver_same"])
            vals = torch.tensor([[sum(log[k]), len(log[k])] for k in keys], device=dev)
            dist.all_reduce(vals)
            if rank == 0:
                s = " ".join(f"{k} {v[0].item() / v[1].item():.3f}" for k, v in zip(keys, vals) if v[1] > 0)
                print(f"[rl] step {step + 1} {s} ({time.time() - t0:.0f}s)", flush=True)
            log.clear()
    if rank == 0:
        out = ROOT / "runs" / a.tag
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "final")
        proc.tokenizer.padding_side = "right"
        proc.save_pretrained(out / "final")
        print(f"[rl] saved {out / 'final'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
