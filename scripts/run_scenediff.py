"""Runs SceneDiff on the 5 + 5-frame items and stores its change regions as proposals."""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

CODE = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1])) / "ext/scene_diff_code"
ROOT = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.chdir(CODE)  # the official code resolves checkpoints/ and submodules/ relative to its root
    sys.path.insert(0, str(CODE))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    from pycocotools import mask as mask_utils
    import modules.semantic_model as sm
    from modules.scenediff import SceneDiff

    cfg = yaml.safe_load(open(CODE / "configs/scenediff_config.yml"))
    cfg["visualization"] = dict(vis_pc=False, save_detections=False, save_gt_visualization=False)
    if os.environ.get("STILL_DINO") == "v3hf":
        from transformers import DINOv3ViTModel
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        def load(self):
            if self.model is None:
                self.model = DINOv3ViTModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m").to(self.device).eval()
                print("[run_scenediff] DINOv3 ViT-H+/16 (Hugging Face release)", flush=True)

        def feat(img_np, model):
            H, W = img_np.shape[:2]
            x = torch.tensor(img_np).float().permute(2, 0, 1)[None]
            h, w = math.ceil(H / 16) * 16, math.ceil(W / 16) * 16
            x = (F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) - mean) / std
            with torch.no_grad():
                out = model(pixel_values=x.to("cuda")).last_hidden_state
            t = out[:, 1 + model.config.num_register_tokens:]
            t = t.reshape(1, h // 16, w // 16, -1).permute(0, 3, 1, 2)
            return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0]

        sm.SemanticModel._load_dinov3_model = load
        sm.predict_dinov3_feat = feat
    elif os.environ.get("STILL_DINO") == "v2":
        cfg["models"]["dinov3"]["feature_dim"] = 1024
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def load(self):
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(self.device).eval()
            print("[run_scenediff] provisional: DINOv2 ViT-L/14 in place of DINOv3", flush=True)

        def feat(img_np, model):
            H, W = img_np.shape[:2]
            x = torch.tensor(img_np).float()
            x = x.permute(2, 0, 1) if x.shape[-1] == 3 else x
            x = x / 255.0 if x.max() > 1.5 else x
            x = (x - mean) / std
            h, w = math.ceil(H / 14) * 14, math.ceil(W / 14) * 14
            x = F.interpolate(x[None], size=(h, w), mode="bilinear", align_corners=False).to("cuda")
            with torch.no_grad():
                t = model.forward_features(x)["x_norm_patchtokens"]
            t = t.reshape(1, h // 14, w // 14, -1).permute(0, 3, 1, 2)
            return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0]

        sm.SemanticModel._load_dinov3_model = load
        sm.predict_dinov3_feat = feat
    if os.environ.get("STILL_GPU_FRAC"):  # cap memory for runs on the shared light-job GPUs (<= 20 GB of 80)
        torch.cuda.set_per_process_memory_fraction(float(os.environ["STILL_GPU_FRAC"]))
    model = SceneDiff(cfg, device="cuda")
    # The official code caches SAM masks at <images>/../sam_masks.pkl, i.e. ONE file for all our items (they share
    # data/img_sd/). Redirect the cache to one file per item outside the data tree.
    import hashlib
    import modules.mask_model as mm
    cache_dir = ROOT / "ext/sdiff_cache"
    cache_dir.mkdir(exist_ok=True)
    orig = mm.MaskModel.load_or_generate_masks

    def per_item(self, file_list, H, W, cache_path=None):
        key = hashlib.sha1("|".join(map(str, file_list)).encode()).hexdigest()
        return orig(self, file_list, H, W, cache_dir / f"{key}_{H}x{W}.pkl")
    mm.MaskModel.load_or_generate_masks = per_item

    items = [json.loads(l) for l in open(ROOT / a.items)]
    if a.limit:
        items = items[:a.limit]
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        items = items[k::n]
    name = Path(a.items).stem.replace("items_", "")
    out_path = ROOT / "res" / (f"{a.tag}_{name}_Vg" + (f"_s{a.shard.split('/')[0]}" if a.shard else "") + ".jsonl")
    rows = []
    for i, it in enumerate(items):
        pred = []
        with tempfile.TemporaryDirectory() as tmp:
            try:
                om = model.process_scene(it["images"], it["n1"], it["n2"], Path(tmp))
            except Exception as e:  # e.g. no change region found
                print(f"[run_scenediff] {it['id']}: {type(e).__name__}: {e}", flush=True)
                om = {}
        H, W = om.get("H"), om.get("W")
        for oid, d in om.items():
            if oid in ("H", "W"):
                continue
            kind = "moved" if ("video_1" in d and "video_2" in d) else ("removed" if "video_1" in d else "added")
            for vk, frames in d.items():
                v = 1 if vk == "video_1" else 2
                for f, m in frames.items():
                    mk = mask_utils.decode(m["mask"])
                    ys, xs = np.where(mk)
                    if len(xs) == 0:
                        continue
                    hh, ww = mk.shape
                    pred.append(dict(change=kind, label="object", video=v, frame=int(f) + 1,
                                     bbox_2d=[round(1000 * xs.min() / ww), round(1000 * ys.min() / hh),
                                              round(1000 * (xs.max() + 1) / ww), round(1000 * (ys.max() + 1) / hh)],
                                     conf=float(m["cost"])))
        rows.append(dict(id=it["id"], group=it["group"], gt=it["gt"], ignore=it["ignore"], control=it["control"],
                         pred=pred))
        print(f"[run_scenediff] {i + 1}/{len(items)} {it['id']}: {len(pred)} entries", flush=True)
        torch.cuda.empty_cache()
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[run_scenediff] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
