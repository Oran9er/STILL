"""SceneDiff predictions under the official protocol (object_masks.pkl per test pair)."""
import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

CODE = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1])) / "ext/scene_diff_code"
FR = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1])) / "ext/sd_official_ff"
SRC = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1])) / "ext/scenediff/scenediff_benchmark"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    os.chdir(CODE)
    sys.path.insert(0, str(CODE))
    import torch
    import torch.nn.functional as F
    import yaml
    from transformers import DINOv3ViTModel
    import modules.semantic_model as sm
    from modules.scenediff import SceneDiff
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def load(self):
        if self.model is None:
            self.model = DINOv3ViTModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m").to(self.device).eval()

    def feat(img_np, model):
        H, W = img_np.shape[:2]
        x = torch.tensor(img_np).float().permute(2, 0, 1)[None]
        h, w = math.ceil(H / 16) * 16, math.ceil(W / 16) * 16
        x = (F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) - mean) / std
        with torch.no_grad():
            t = model(pixel_values=x.to("cuda")).last_hidden_state[:, 1 + model.config.num_register_tokens:]
        t = t.reshape(1, h // 16, w // 16, -1).permute(0, 3, 1, 2)
        return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0]
    sm.SemanticModel._load_dinov3_model = load
    sm.predict_dinov3_feat = feat
    cfg = yaml.safe_load(open(CODE / "configs/scenediff_config.yml"))
    cfg["visualization"] = dict(vis_pc=False, save_detections=False, save_gt_visualization=False)
    model = SceneDiff(cfg, device="cuda")
    sp = json.load(open(SRC / "splits/test_split.json"))
    seqs = sp["varied"] + sp["kitchen"]
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        seqs = seqs[k::n]
    for i, seq in enumerate(seqs):
        dst = out / seq / "object_masks.pkl"
        if dst.exists():
            continue
        f1 = sorted((FR / seq / "video1_frames").glob("*.jpg"))
        f2 = sorted((FR / seq / "video2_frames").glob("*.jpg"))
        try:
            om = model.process_scene([str(x) for x in f1 + f2], len(f1), len(f2), out / "_work")
        except Exception as e:
            print(f"[official] {seq}: {type(e).__name__}: {e}", flush=True)
            om = {}
        dst.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(om, open(dst, "wb"))
        print(f"[official] {i + 1}/{len(seqs)} {seq}: {len([k for k in om if k not in ('H', 'W')])} objects", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
