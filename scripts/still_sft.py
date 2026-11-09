"""Full-parameter FSDP fine-tuning of the verifier.

python -m torch.distributed.run --nproc_per_node 4 scripts/still_sft.py --model M --out runs/NAME --train a.jsonl b.jsonl --epochs 1
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from still_common import aligned_position_ids, align_meta, attach_codiff, content, encode, images, load, target  # noqa: E402

WRAP = {"qwen3_vl": ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"],
        "qwen2_5_vl": ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"],
        "qwen3_5": ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"],
        "internvl": ["Qwen3DecoderLayer", "InternVLVisionLayer"]}  # InternVL3.5-8B-HF


def to_bf16(folder):
    from safetensors.torch import load_file, save_file
    for f in sorted(Path(folder).glob("*.safetensors")):
        sd = load_file(str(f))
        tmp = str(f) + ".bf16.tmp"  # write next to it, then replace: a failed write (e.g. disk full) keeps the fp32 file
        save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}, tmp,
                  metadata={"format": "pt"})
        os.replace(tmp, str(f))


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L], lab[i, :L], att[i, :L] = b["input_ids"], b["labels"], 1
    out = {"input_ids": ids, "labels": lab, "attention_mask": att}
    if "mm_token_type_ids" in batch[0]:
        mt = torch.zeros((len(batch), n), dtype=batch[0]["mm_token_type_ids"].dtype)
        for i, b in enumerate(batch):
            mt[i, :len(b["mm_token_type_ids"])] = b["mm_token_type_ids"]
        out["mm_token_type_ids"] = mt
    for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if k in batch[0]:
            out[k] = torch.cat([b[k] for b in batch])
    for k in ("align_H", "align_sz"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


class Items(Dataset):
    def __init__(self, items, proc, arm, maxpix):
        self.items, self.proc, self.arm, self.maxpix = items, proc, arm, maxpix

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        user = [{"role": "user", "content": content(it, self.arm)}]
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": target(it)}]}]
        p_txt = self.proc.apply_chat_template(user, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        f_txt = self.proc.apply_chat_template(full, tokenize=False, enable_thinking=False)
        ims = images(it, self.arm, self.maxpix)
        enc = encode(self.proc, [f_txt], ims, self.arm)
        n_prompt = encode(self.proc, [p_txt], ims, self.arm)["input_ids"].shape[1]
        ids = enc["input_ids"][0]
        labels = ids.clone()
        labels[:n_prompt] = -100
        if self.proc.tokenizer.decode(ids[-1:]) == "\n":
            labels[-1] = -100
        out = {"input_ids": ids, "labels": labels}
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if k in enc:
                out[k] = enc[k]
        if "mm_token_type_ids" in enc:
            out["mm_token_type_ids"] = enc["mm_token_type_ids"][0]
        if self.arm in ("P", "C"):
            H, sz = align_meta(it)
            out["align_H"], out["align_sz"] = torch.tensor(H, dtype=torch.float32), torch.tensor(sz, dtype=torch.float32)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--arm", default="V", choices=["V", "A", "B", "F", "P", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--maxpix", type=int, default=448 * 1024)
    ap.add_argument("--new-lr", type=float, default=1e-4, help="learning rate of newly added modules (CoDiff, arm C)")
    ap.add_argument("--balance", action="store_true", help="resample each file to equal yes / no counts per epoch")
    args = ap.parse_args()
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
    rng = random.Random(args.seed)
    cases = []
    for f in args.train:
        f, mult = (f.split("@")[0], float(f.split("@")[1])) if "@" in f else (f, 1.0)  # path@N: N times the epochs
        base = load(f)
        ep_f = args.epochs * mult
        reps = int(ep_f) + (1 if ep_f % 1 else 0)
        if args.balance:
            y, n = [b for b in base if b["gold"] == "yes"], [b for b in base if b["gold"] == "no"]
            m = max(len(y), len(n))
            ep = [rng.sample(y, len(y)) + rng.choices(y, k=m - len(y)) + rng.sample(n, len(n)) + rng.choices(n, k=m - len(n))
                  for _ in range(reps)]
        else:
            ep = [list(base) for _ in range(reps)]
        tot = int(round(len(ep[0]) * ep_f))
        flat = [c for e in ep for c in rng.sample(e, len(e))][:tot]
        print(f"[sft] {f}: {len(base)} items -> {len(flat)} samples", flush=True)
        cases += flat
    rng.shuffle(cases)
    proc = AutoProcessor.from_pretrained(args.model)
    world = int(os.environ.get("WORLD_SIZE", 1))
    accum = max(1, round(16 / world))
    print(f"[sft] {len(cases)} samples, effective batch {accum * world}, arm {args.arm}", flush=True)
    mtype = AutoConfig.from_pretrained(args.model).model_type
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa")
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=1, gradient_accumulation_steps=accum, num_train_epochs=1,
        max_steps=args.max_steps, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03,
        weight_decay=0.0, bf16=True, logging_steps=20, save_strategy="no", report_to=[], seed=args.seed,
        data_seed=args.seed, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4, remove_unused_columns=False, ddp_timeout=1800,
        fsdp="full_shard auto_wrap" if world > 1 else "",
        fsdp_config={"transformer_layer_cls_to_wrap": WRAP[mtype], "use_orig_params": True} if world > 1 else None)

    rope_fn = getattr(model.model, "get_rope_index", None)  # pure function of the token ids; safe to call outside the FSDP wrapper
    core = attach_codiff(model) if args.arm == "C" else None

    class STrainer(Trainer):
        def create_optimizer(self, *a, **k):
            if self.optimizer is None and core is not None:
                named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
                new = [p for n, p in named if "codiff" in n]
                assert new, "CoDiff parameters not found"
                self.optimizer = torch.optim.AdamW([{"params": [p for n, p in named if "codiff" not in n], "lr": args.lr},
                                                    {"params": new, "lr": args.new_lr}], lr=args.lr, weight_decay=0.0)
                print(f"[sft] CoDiff: {sum(p.numel() for p in new) / 1e6:.1f} M new parameters at lr {args.new_lr}", flush=True)
                return self.optimizer
            return super().create_optimizer(*a, **k)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            if "align_H" in inputs:
                Hs, sz = inputs.pop("align_H"), inputs.pop("align_sz")
                if core is not None:  # arm C: counterpart features injected into the visual tokens
                    core._codiff_ctx = (Hs, sz)
                else:  # arm P: image 2 positioned in the frame of image 1
                    inputs["position_ids"] = aligned_position_ids(rope_fn, inputs, Hs, sz)
            logits = model(**inputs).logits[:, :-1]
            tgt = labels[:, 1:]
            bi, ti = (tgt != -100).nonzero(as_tuple=True)
            ce = torch.nn.functional.cross_entropy(logits[bi, ti].float(), tgt[bi, ti], reduction="none")
            n = torch.zeros(tgt.shape[0], device=ce.device).index_add_(0, bi, torch.ones_like(ce))
            tot = torch.zeros(tgt.shape[0], device=ce.device).index_add_(0, bi, ce)
            return (tot / n.clamp(min=1)).mean()

    ds = Items(cases, proc, args.arm, args.maxpix)
    tr = STrainer(model=model, args=targs, train_dataset=ds, data_collator=lambda b: collate(b, proc.tokenizer.pad_token_id))
    tr.model_accepts_loss_kwargs = False
    tr.train()
    tr.accelerator.wait_for_everyone()
    final = Path(args.out) / "final"
    if world > 1:
        tr.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    tr.save_model(str(final))
    if tr.accelerator.is_main_process:
        proc.save_pretrained(final)
        to_bf16(final)
        json.dump({"args": vars(args), "log": tr.state.log_history}, open(Path(args.out) / "log_history.json", "w"), indent=1)
        print(f"[sft] saved {final}", flush=True)


if __name__ == "__main__":
    main()
