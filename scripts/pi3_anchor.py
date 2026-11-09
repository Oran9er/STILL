"""Registers each query box in the other capture with pi^3: lift to 3D, visibility test, best-visible frame and quantile box."""
import os
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

CODE = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1])) / "ext/scene_diff_code"
ROOT = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1]))


def fit_k(lp, conf):
    """lp: (H, W, 3) camera points, conf mask -> fx, fy, cx, cy (pixel units of this map)."""
    import torch
    H, W = lp.shape[:2]
    v, u = torch.meshgrid(torch.arange(H, device=lp.device, dtype=torch.float32) + 0.5,
                          torch.arange(W, device=lp.device, dtype=torch.float32) + 0.5, indexing="ij")
    m = conf & (lp[..., 2] > 1e-4)
    x, y = (lp[..., 0] / lp[..., 2])[m], (lp[..., 1] / lp[..., 2])[m]
    A = torch.stack([x, torch.ones_like(x)], 1)
    fx, cx = torch.linalg.lstsq(A, u[m][:, None]).solution[:, 0].tolist()
    A = torch.stack([y, torch.ones_like(y)], 1)
    fy, cy = torch.linalg.lstsq(A, v[m][:, None]).solution[:, 0].tolist()
    return fx, fy, cx, cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--shard", default="")
    a = ap.parse_args()
    import os
    os.chdir(CODE)
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(CODE / "submodules/Pi3"))
    import torch
    from pi3.models.pi3 import Pi3
    from utils import load_images_as_tensor_from_list
    if os.environ.get("STILL_GPU_FRAC"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["STILL_GPU_FRAC"]))
    model = Pi3.from_pretrained("yyfz233/Pi3").to("cuda").eval()
    by = defaultdict(list)
    for l in open(ROOT / a.queries):
        q = json.loads(l)
        by[q["item"]].append(q)
    items = sorted(by)
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        items = items[k::n]
    out = {}
    for ii, key in enumerate(items):
        qs = by[key]
        imgs, n1 = qs[0]["images"], qs[0]["n1"]
        x = load_images_as_tensor_from_list(imgs)[None].to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            r = model(x)
        pts = r["points"][0].float()  # (N, H, W, 3) world
        lp = r["local_points"][0].float()  # (N, H, W, 3) camera
        conf = torch.sigmoid(r["conf"][0, ..., 0].float()) > 0.1
        T = r["camera_poses"][0].float()  # (N, 4, 4) camera-to-world
        N, H, W = pts.shape[:3]
        K = [fit_k(lp[i], conf[i]) for i in range(N)]
        Tinv = torch.linalg.inv(T)
        for q in qs:
            v, f, b = q["query"]["video"], q["query"]["frame"], q["query"]["bbox_2d"]
            i = (f - 1) if v == 1 else n1 + f - 1
            x0, y0 = int(b[0] / 1000 * W), int(b[1] / 1000 * H)
            x1, y1 = max(x0 + 1, int(round(b[2] / 1000 * W))), max(y0 + 1, int(round(b[3] / 1000 * H)))
            P = pts[i, y0:y1, x0:x1][conf[i, y0:y1, x0:x1]]
            others = range(n1, N) if v == 1 else range(0, n1)
            best = (0.0, None, None)
            for j in others:
                if len(P) < 10:
                    break
                Pc = (Tinv[j, :3, :3] @ P.T).T + Tinv[j, :3, 3]
                z = Pc[:, 2]
                fx, fy, cx, cy = K[j]
                u = fx * Pc[:, 0] / z.clamp(min=1e-6) + cx
                w = fy * Pc[:, 1] / z.clamp(min=1e-6) + cy
                inside = (z > 1e-4) & (u >= 0) & (u < W) & (w >= 0) & (w < H)
                if inside.sum() == 0:
                    continue
                ui, wi = u[inside].long().clamp(0, W - 1), w[inside].long().clamp(0, H - 1)
                dz = lp[j, wi, ui, 2]
                vis = z[inside] <= dz * 1.05
                share = float(vis.sum()) / len(P)
                if share > best[0]:
                    uu, ww = u[inside][vis], w[inside][vis]
                    qx = torch.quantile(uu, torch.tensor([0.05, 0.95], device=uu.device)).tolist()
                    qy = torch.quantile(ww, torch.tensor([0.05, 0.95], device=ww.device)).tolist()
                    box = [round(1000 * qx[0] / W), round(1000 * qy[0] / H), round(1000 * qx[1] / W), round(1000 * qy[1] / H)]
                    best = (share, (j - n1 + 1) if v == 1 else j + 1, [min(1000, max(0, c)) for c in box])
            out[q["id"]] = [best[1], best[2], round(best[0], 4)]
        if ii % 20 == 0:
            print(f"[pi3] {ii + 1}/{len(items)} items, {len(out)} queries", flush=True)
    name = Path(a.queries).stem.replace("items_", "")
    dst = ROOT / (f"data/p2/pi3_anchor_{name}" + (f"_s{a.shard.split('/')[0]}" if a.shard else "") + ".json")
    json.dump(out, open(dst, "w"))
    print(f"[pi3] wrote {dst} ({len(out)} queries)", flush=True)


if __name__ == "__main__":
    main()
