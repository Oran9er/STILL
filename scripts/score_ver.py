"""Scores verifier queries: P(same / changed / unseen) at the first answer token.

score_ver.py --model M --tag T --items data/p2/items_vsd_test.jsonl [--shard k/n] [--bs 8] [--full]
"""
import copy
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import ROOT, content, encode, images, load  # noqa: E402

WORDS = ("same", "changed", "unseen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--shard", default="")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="one full forward per query (no prefix sharing)")
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    tok = proc.tokenizer
    ids = [sorted({tok(w, add_special_tokens=False)["input_ids"][0], tok(w.capitalize(), add_special_tokens=False)["input_ids"][0]}) for w in WORDS]
    items = load(args.items)
    if args.limit:
        items = items[:args.limit]
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        items = items[k::n]
    def full(batch):
        texts = [proc.apply_chat_template([{"role": "user", "content": content(it)}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False) for it in batch]
        enc = encode(proc, texts, [im for it in batch for im in images(it)], "V", padding=True).to("cuda")
        with torch.no_grad():
            return model(**enc).logits[:, -1].float()

    def shared(group):
        """group: queries of one scene item (same images)."""
        mark = "\u0001QUESTION\u0001"
        it0 = dict(group[0], q=mark)
        text = proc.apply_chat_template([{"role": "user", "content": content(it0)}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        pre, post = text.split(mark)
        enc = encode(proc, [pre], images(group[0]), "V").to("cuda")
        with torch.no_grad():
            out = model(**enc, use_cache=True)
        cache, n0 = out.past_key_values, enc["input_ids"].shape[1]
        delta = model.model.rope_deltas.view(-1)[0].item()
        logits = []
        for b in range(0, len(group), args.bs):
            sub = group[b:b + args.bs]
            suf = [tok(it["q"] + post, add_special_tokens=False)["input_ids"] for it in sub]
            L = max(map(len, suf))
            ids = torch.full((len(sub), L), tok.pad_token_id, dtype=torch.long)
            am = torch.zeros((len(sub), n0 + L), dtype=torch.long)
            am[:, :n0] = 1
            for i, x in enumerate(suf):
                ids[i, :len(x)] = torch.tensor(x)
                am[i, n0:n0 + len(x)] = 1
            t = (n0 + torch.arange(L)).view(1, 1, -1).expand(1, len(sub), -1)
            pos = torch.cat([t, (t + delta).expand(3, -1, -1)], 0)
            c = copy.deepcopy(cache)
            for layer in c.layers:
                for name in ("keys", "values"):
                    v = getattr(layer, name, None)
                    if isinstance(v, torch.Tensor):
                        setattr(layer, name, v.repeat_interleave(len(sub), 0))
                for name in ("conv_states", "recurrent_states"):
                    d = getattr(layer, name, None)
                    if isinstance(d, dict):
                        for k, v in d.items():
                            if isinstance(v, torch.Tensor):
                                d[k] = v.repeat_interleave(len(sub), 0)
            with torch.no_grad():
                lg = model(input_ids=ids.cuda(), attention_mask=am.cuda(), position_ids=pos.cuda(), past_key_values=c,
                           use_cache=True).logits.float()
            logits.append(lg[torch.arange(len(sub)), torch.tensor([len(x) - 1 for x in suf])])
            del c
        return torch.cat(logits, 0)

    def probs(lg):
        pr = torch.softmax(lg, -1)
        pw = torch.stack([pr[:, i].sum(-1) for i in ids], -1)
        return (pw / pw.sum(-1, keepdim=True).clamp(min=1e-9)).cpu()

    groups = {}
    for it in items:
        groups.setdefault(it["item"], []).append(it)
    use_full = args.full or items[0]["task"] == "crop"  # crop pairs have their own images per query
    if not use_full:
        g0 = list(groups.values())[0][:args.bs]
        try:
            a, b = probs(shared(g0)), probs(full(g0))
            diff = (a - b).abs().max().item()
        except Exception as e:  # any failure of the shared path -> full forwards
            print(f"[ver] prefix sharing raised {type(e).__name__}: {e}", flush=True)
            diff = float("inf")
        print(f"[ver] prefix-sharing self-check on {len(g0)} queries: max |dp| = {diff:.4f}", flush=True)
        use_full = diff > 0.05  # bf16 noise between batch layouts is about 0.01-0.02 (seen 0.006 / 0.022)
        if use_full:
            print("[ver] self-check failed -> full forwards", flush=True)
    rows, done = [], 0
    for gi, group in enumerate(groups.values()):
        if use_full:
            pw = torch.cat([probs(full(group[b:b + args.bs])) for b in range(0, len(group), args.bs)], 0)
        else:
            pw = probs(shared(group))
        for it, q in zip(group, pw.tolist()):
            rows.append(dict(id=it["id"], item=it["item"], group=it["group"], gold=it["gold"], obj=it["obj"], kind=it["kind"],
                             query=it["query"], p=[round(x, 5) for x in q]))
        done += len(group)
        if gi % 20 == 0:
            print(f"[ver] {done}/{len(items)}", flush=True)
    stem = f"{args.tag}_{Path(args.items).stem.replace('items_', '')}" + (f"_s{args.shard.split('/')[0]}" if args.shard else "")
    with open(ROOT / "res" / f"{stem}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pred = [WORDS[max(range(3), key=lambda i: r["p"][i])] for r in rows]
    rec = {w: round(sum(p == w for p, r in zip(pred, rows) if r["gold"] == w) / max(sum(r["gold"] == w for r in rows), 1), 3) for w in WORDS}
    print(json.dumps(dict(tag=args.tag, n=len(rows), acc=round(sum(p == r["gold"] for p, r in zip(pred, rows)) / len(rows), 4),
                          recall=rec, pred=dict(Counter(pred)))))


if __name__ == "__main__":
    main()
