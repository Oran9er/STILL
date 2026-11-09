"""Holistic VLM baseline: generates the list of changes for each scene item and scores it."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import (ROOT, align_meta, attach_codiff, attach_conf, content, cope_generate, encode,  # noqa: E402
                          entry_confidences, images, list_metrics, load, parse_list, parse_scene, scene_ap,
                          scene_metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--arm", default="V", choices=["V", "S", "F", "P", "C"])
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--maxpix", type=int, default=448 * 1024)
    ap.add_argument("--adapter", default="", help="LoRA adapter saved by still_rl4.py --save-every; merged into --model")
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").to("cuda").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    core = attach_codiff(model, weights=args.model).to("cuda").to(torch.bfloat16) if args.arm == "C" else None
    items = [it for it in load(args.items) if it["task"] in ("list", "scene")]
    scene = items[0]["task"] == "scene"
    if args.limit:
        items = items[:args.limit]
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        items = items[k::n]
    name = Path(args.items).stem.replace("items_", "")
    rows = []
    for b in range(0, len(items), args.bs):
        batch = items[b:b + args.bs]
        texts, ims = [], []
        for it in batch:
            texts.append(proc.apply_chat_template([{"role": "user", "content": content(it, args.arm)}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False))
            ims += images(it, args.arm, args.maxpix)
        enc = encode(proc, texts, ims, args.arm, padding=True).to("cuda")
        kw = dict(max_new_tokens=args.max_new, do_sample=False, output_scores=scene, return_dict_in_generate=True)
        if args.arm == "C":
            meta = [align_meta(it) for it in batch]
            core._codiff_ctx = ([m[0] for m in meta], [m[1] for m in meta])
        with torch.no_grad():
            if args.arm == "P":
                meta = [align_meta(it) for it in batch]
                with cope_generate(model, [m[0] for m in meta], [m[1] for m in meta]):
                    res = model.generate(**enc, **kw)
            else:
                res = model.generate(**enc, **kw)
        out = res.sequences
        n_p = enc["input_ids"].shape[1]
        lps = None
        if scene:  # log-probability of every generated token (greedy), for per-entry confidences
            try:
                sc = torch.stack(res.scores, 1).float().log_softmax(-1)  # (batch, new tokens, vocab)
                lps = sc.gather(-1, out[:, n_p:n_p + sc.shape[1]].unsqueeze(-1)).squeeze(-1).cpu()
            except Exception as ex:  # confidences are optional; never fail the evaluation because of them
                print(f"[gen] no scores: {ex!r}", flush=True)
        for bi, (it, o) in enumerate(zip(batch, out[:, n_p:])):
            text = proc.tokenizer.decode(o, skip_special_tokens=True).strip()
            pred = parse_scene(text) if scene else parse_list(text)
            if scene and lps is not None and pred:
                try:
                    keep = [k for k, t in enumerate(o.tolist()) if t not in proc.tokenizer.all_special_ids and k < lps.shape[1]]
                    pieces = [proc.tokenizer.decode([o[k]]) for k in keep]
                    attach_conf(text, pred, entry_confidences(text, pieces, [float(lps[bi, k]) for k in keep]))
                except Exception as ex:
                    print(f"[gen] no confidence for {it['id']}: {ex!r}", flush=True)
            rows.append(dict(id=it["id"], group=it["group"], gold=it["gold"], entries=it["entries"], ignore=it["ignore"],
                             text=text, pred=pred,
                             **({"gt": it["gt"], "control": it["control"]} if scene else {})))
        if (b // args.bs) % 20 == 0:
            print(f"[gen] {b}/{len(items)} {rows[-1]['text'][:120]!r}", flush=True)
    stem = f"{args.tag}_{name}_{args.arm}g" + (f"_s{args.shard.split('/')[0]}" if args.shard else "")
    with open(ROOT / "res" / f"{stem}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fn = scene_metrics if scene else list_metrics
    m3, m5 = fn(rows, 0.3), fn(rows, 0.5)
    summ = dict(tag=args.tag, set=name, arm=args.arm, task="scene" if scene else "list", n=len(rows), f1_03=m3["f1"],
                p_03=m3["precision"], r_03=m3["recall"], f1_05=m5["f1"], kind_acc=m3["kind_acc"],
                pair_sens=m3["pair_sens"], pair_spec=m3["pair_spec"], unparsable=m3["unparsable"])
    if scene:
        summ["recall_by_kind"] = m3["recall_by_kind"]
        summ["control_spec"] = round(sum(not r["pred"] for r in rows if r["control"]) / max(sum(r["control"] for r in rows), 1), 4)
        summ.update({k: v for k, v in scene_ap(rows).items() if k in ("ap", "f1max")})
    json.dump(summ, open(ROOT / "res" / f"{stem}.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
