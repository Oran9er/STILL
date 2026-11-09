"""Shared helpers: item files, image loading, prompts and the scene-level metrics."""
import json
import math
import os
from pathlib import Path

from PIL import Image

ROOT = Path(os.environ.get("STILL_ROOT", Path(__file__).resolve().parents[1]))
MAXPIX = 448 * 1024  # per-image pixel budget (about 450 visual tokens at 32 px per token)

Q_BIN = {
    "scene": ("The two images show the same scene at two different times. The camera position, framing and lighting may "
              "differ between them. Ignoring differences that are only due to viewpoint, framing or lighting, has any "
              "object in the scene been added, removed or moved between the first and the second image? "
              "Answer yes or no."),
}


F_HEAD = ("The video has exactly two frames: its first frame is the first image and its second frame is the second "
          "image. ")


Q_LIST = ("The two images show the same scene at two different times. The camera position, framing and lighting may "
          "differ between them. List every object that was added, removed or moved between the first and the second "
          "image, ignoring differences that are only due to viewpoint, framing or lighting. Answer with a JSON list. "
          "Each entry is {\"change\": \"added\" | \"removed\" | \"moved\", \"image\": 1 or 2, \"bbox_2d\": [x1, y1, x2, y2], "
          "\"label\": \"<object>\"}, where \"image\" is the image in which the object is visible and the box is given in "
          "that image in 0-1000 coordinates. Answer [] if nothing changed.")


SCENE_PIX = int(os.environ.get("STILL_SCENE_PIX", 256 * 1024))  # per-frame pixel budget for multi-frame items (384 x 672 for SceneDiff frames, 252 tokens)
Q_SCENE = ("Between time 1 and time 2 some objects in the scene may have been added, removed or moved. List every such "
           "object. Ignore differences that come only from the camera position, framing or lighting. Answer with a JSON "
           "list. Each entry is {\"change\": \"added\" | \"removed\" | \"moved\", \"label\": \"<object>\", \"video\": 1 or 2, "
           "\"frame\": <frame number>, \"bbox_2d\": [x1, y1, x2, y2]}: one frame in which the object is visible (a frame of "
           "time 1 for a removed object, of time 2 for an added object, either for a moved object) and its box in that "
           "frame in 0-1000 coordinates. Answer [] if nothing changed.")


Q_SCENE_E = ("Between time 1 and time 2 some objects in the scene may have been added, removed or moved. List every such "
             "object for which the change can be verified, that is, the place of the object is also in view in a frame "
             "of the other time. An object that is merely out of view at the other time has not changed. Ignore "
             "differences that come only from the camera position, framing or lighting. Answer with a JSON list. Each "
             "entry is {\"change\": \"added\" | \"removed\" | \"moved\", \"label\": \"<object>\", \"video\": 1 or 2, "
             "\"frame\": <frame number>, \"bbox_2d\": [x1, y1, x2, y2], \"check\": {\"video\": <the other video>, "
             "\"frame\": <frame number>}}: one frame in which the object is visible with its box in 0-1000 coordinates, "
             "and as \"check\" the frame of the other time in which the place of the object is seen. If the object "
             "itself is visible at the other time (it was moved), add its box there to \"check\" as \"bbox_2d\". "
             "Answer [] if nothing changed.")


def load(path):
    return [json.loads(l) for l in open(path)]


def dump(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_image(path, maxpix=MAXPIX):
    im = Image.open(path).convert("RGB")
    s = min(1.0, math.sqrt(maxpix / (im.size[0] * im.size[1])))
    w, h = (max(32, round(im.size[0] * s / 32) * 32), max(32, round(im.size[1] * s / 32) * 32))
    return im if (w, h) == im.size else im.resize((w, h), Image.BICUBIC)


def question(it):
    if it["task"] == "bin":
        return it.get("q") or Q_BIN[it.get("domain", "scene")]
    if it["task"] == "list":
        return Q_LIST
    if it["task"] == "scene":
        return Q_SCENE_E if it.get("fmt") == "ev" else Q_SCENE
    if it["task"] in ("covis", "bin3", "crop", "loc", "ref"):  # per-object queries (build_rscan_c.py, build_verify.py, build_crops.py, build_locate.py)
        return it["q"]
    raise ValueError(it["task"])


def target(it):
    if it["task"] == "bin":
        return it["gold"]
    if it["task"] == "covis":
        return it["a"]
    if it["task"] in ("bin3", "crop"):
        return it["gold"]
    if it["task"] == "loc":
        return json.dumps(it["loc"])
    if it["task"] == "ref":  # box refinement
        return json.dumps({"bbox_2d": it["ref"]})
    return json.dumps(it["entries_ev"] if it.get("fmt") == "ev" else it["entries"])


def scene_content(it):
    """Multi-frame layout: a text label before every frame, so that the answer can name frames."""
    c = [{"type": "text", "text": f"The same scene was recorded twice along different camera paths. "
                                  f"Time 1 (before), {it['n1']} frames:"}]
    pr = it.get("props")  # proposals-in-prompt rival (build_sds_props.py): detector boxes listed per frame
    lab = lambda v, k: f"\nVideo {v}, frame {k + 1}:" if pr is None else (
        f"\nVideo {v}, frame {k + 1} (detected objects: {'; '.join(map(str, pr[k if v == 1 else it['n1'] + k])) or 'none'}):")
    for k in range(it["n1"]):
        c += [{"type": "text", "text": lab(1, k)}, {"type": "image"}]
    c.append({"type": "text", "text": f"\nTime 2 (after), {it['n2']} frames:"})
    for k in range(it["n2"]):
        c += [{"type": "text", "text": lab(2, k)}, {"type": "image"}]
    return c + [{"type": "text", "text": "\n" + question(it)}]


def crop_content(it):
    """Place-registered crop pair: the object's crop in its own capture, then the same place in the other."""
    return [{"type": "text", "text": "Image 1, the object in one capture of a scene:"}, {"type": "image"},
            {"type": "text", "text": "\nImage 2, the same place in the other capture:"}, {"type": "image"},
            {"type": "text", "text": "\n" + question(it)}]


def ref_content(it):
    """Box refinement: the query crop, then a zoomed part of the other capture's frame."""
    return [{"type": "text", "text": "Image 1, the object in one capture of a scene:"}, {"type": "image"},
            {"type": "text", "text": "\nImage 2, part of a frame of the other capture:"}, {"type": "image"},
            {"type": "text", "text": "\n" + question(it)}]


def content(it, arm="V"):
    """arm V: both images; A: first image only; B: second image only; S: images swapped; P: both images, with the M-RoPE positions of image 2 moved into the frame of image 1 (aligned_position_ids); C: both images, default positions, counterpart features injected by the CoDiff module (attach_codiff); F: the pair as ONE two-frame video (Qwen's temporal patch embedding fuses the two frames at pixel level)."""
    if it["task"] == "crop":
        return crop_content(it)
    if it["task"] == "ref":
        return ref_content(it)
    if it["task"] in ("scene", "covis", "bin3", "loc"):
        return scene_content(it)
    if arm == "F":
        return [{"type": "video"}, {"type": "text", "text": F_HEAD + question(it)}]
    n = {"V": 2, "S": 2, "A": 1, "B": 1, "P": 2, "C": 2}[arm]
    return [{"type": "image"} for _ in range(n)] + [{"type": "text", "text": question(it)}]


def images(it, arm="V", maxpix=MAXPIX):
    p = it["images"]
    if it["task"] in ("crop", "ref"):
        return [load_image(x, maxpix) for x in p]
    if it["task"] in ("scene", "covis", "bin3", "loc"):
        return [load_image(x, min(maxpix, SCENE_PIX)) for x in p]
    p = {"V": p, "S": p[::-1], "A": p[:1], "B": p[1:], "F": p, "P": p, "C": p}[arm]
    ims = [load_image(x, maxpix) for x in p]
    if arm == "F" and ims[1].size != ims[0].size:
        ims[1] = ims[1].resize(ims[0].size, Image.BICUBIC)
    return ims


def encode(proc, texts, ims, arm, **kw):
    """Processor call for a batch: ims is the flat image list (two per item for arm F)."""
    if arm == "F":
        vids = [[ims[i], ims[i + 1]] for i in range(0, len(ims), 2)]
        return proc(videos=vids, text=texts, add_special_tokens=False, return_tensors="pt", do_sample_frames=False, **kw)
    return proc(images=ims, text=texts, add_special_tokens=False, return_tensors="pt", **kw)


def auc(pos, neg):
    """Mann-Whitney AUC of scores (pos should be larger)."""
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bin_metrics(rows):
    """rows: dicts with gold ('yes'/'no'), p (P(yes)), group."""
    import numpy as np
    pos = [r["p"] for r in rows if r["gold"] == "yes"]
    neg = [r["p"] for r in rows if r["gold"] == "no"]
    by = {}
    for r in rows:
        by.setdefault(r["group"], {"yes": [], "no": []})[r["gold"]].append(r["p"])
    w = [(auc(g["yes"], g["no"]), len(g["yes"]) * len(g["no"])) for g in by.values() if g["yes"] and g["no"]]
    sens = float(np.mean([p > 0.5 for p in pos])) if pos else float("nan")
    spec = float(np.mean([p <= 0.5 for p in neg])) if neg else float("nan")
    return dict(n=len(rows), n_pos=len(pos), n_neg=len(neg), auc=round(auc(pos, neg), 4),
                auc_within=round(float(np.mean([a for a, _ in w])), 4) if w else None, n_groups_within=len(w),
                sens=round(sens, 4), spec=round(spec, 4), bal_acc=round((sens + spec) / 2, 4))


def align_meta(it):
    """(H, sizes) for arm P: H maps image-2 pixels to image-1 pixels (identity if the item has none, i.e."""
    w1, h1 = Image.open(it["images"][0]).size
    w2, h2 = Image.open(it["images"][1]).size
    H = it.get("H") or [[w1 / w2, 0.0, 0.0], [0.0, h1 / h2, 0.0], [0.0, 0.0, 1.0]]
    return H, [w1, h1, w2, h2]


def aligned_position_ids(rope_fn, enc, Hs, sizes, token_px=32):
    """M-RoPE position ids with image 2 placed in the coordinate frame of image 1 (arm P, "CoPE")."""
    base, _ = rope_fn(enc["input_ids"], mm_token_type_ids=enc["mm_token_type_ids"],
                      image_grid_thw=enc["image_grid_thw"], attention_mask=enc.get("attention_mask"))
    return align_positions(base, enc["mm_token_type_ids"], enc["image_grid_thw"], Hs, sizes, token_px)


def align_positions(base, mm_token_type_ids, image_grid_thw, Hs, sizes, token_px=32):
    """The alignment step of aligned_position_ids on given base positions (3, B, L)."""
    import torch
    enc = {"mm_token_type_ids": mm_token_type_ids, "image_grid_thw": image_grid_thw}
    pos = base.float()
    grids = enc["image_grid_thw"]
    for b in range(pos.shape[1]):
        idx = (enc["mm_token_type_ids"][b] == 1).nonzero().flatten()
        cut = (idx[1:] - idx[:-1] > 1).nonzero().flatten()
        assert len(cut) == 1, "arm P needs exactly two images per item"
        a, c = idx[:cut[0] + 1], idx[cut[0] + 1:]
        (h1, w1), (h2, w2) = [(int(g[1]) // 2, int(g[2]) // 2) for g in (grids[2 * b], grids[2 * b + 1])]
        assert len(a) == h1 * w1 and len(c) == h2 * w2
        s1 = base[1, b, a[0]].float()
        W1o, H1o, W2o, H2o = [float(v) for v in sizes[b]]
        ii, jj = torch.meshgrid(torch.arange(h2, device=pos.device), torch.arange(w2, device=pos.device), indexing="ij")
        x2 = (jj.flatten().float() + 0.5) * token_px * (W2o / (w2 * token_px))
        y2 = (ii.flatten().float() + 0.5) * token_px * (H2o / (h2 * token_px))
        H = torch.as_tensor(Hs[b], dtype=torch.float32, device=pos.device)
        q = H @ torch.stack([x2, y2, torch.ones_like(x2)])
        den = torch.where(q[2].abs() < 1e-6, torch.full_like(q[2], 1e-6), q[2])
        x1, y1 = q[0] / den, q[1] / den
        behind = q[2] <= 0  # beyond the horizon of the homography: push far outside
        wq = (x1 / (W1o / (w1 * token_px))) / token_px - 0.5
        hq = (y1 / (H1o / (h1 * token_px))) / token_px - 0.5
        wq = torch.where(behind, torch.full_like(wq, 2.0 * w1), wq).clamp(-float(w1), 2.0 * w1)
        hq = torch.where(behind, torch.full_like(hq, 2.0 * h1), hq).clamp(-float(h1), 2.0 * h1)
        pos[1, b, c] = s1 + hq
        pos[2, b, c] = s1 + wq
    return pos


class cope_generate:
    """Context manager for generation in arm P: wraps model.model.get_rope_index so that the prompt positions are the aligned ones (the deltas for the generated text tokens are unchanged)."""

    def __init__(self, model, Hs, sizes):
        self.m, self.Hs, self.sizes = model.model, Hs, sizes

    def __enter__(self):
        self.orig = self.m.get_rope_index

        def wrapped(input_ids, mm_token_type_ids=None, image_grid_thw=None, **kw):
            base, deltas = self.orig(input_ids, mm_token_type_ids=mm_token_type_ids, image_grid_thw=image_grid_thw, **kw)
            n = base.shape[1] // len(self.Hs)  # beams / repeats per item
            Hs = [h for h in self.Hs for _ in range(n)]
            sz = [s for s in self.sizes for _ in range(n)]
            return align_positions(base, mm_token_type_ids, image_grid_thw, Hs, sz), deltas
        self.m.get_rope_index = wrapped

    def __exit__(self, *a):
        self.m.get_rope_index = self.orig


def json_entries(text):
    """List of dicts from a model answer."""
    import re
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            raw = json.loads(m.group(0))
            if isinstance(raw, list):
                return [e for e in raw if isinstance(e, dict)]
        except Exception:
            pass
    if "[" not in text:
        return None
    out = []
    for a, b in object_spans(text):  # top-level objects, nested braces allowed
        try:
            o = json.loads(text[a:b])
        except Exception:
            continue
        if isinstance(o, dict):
            out.append(o)
    return out if out or re.search(r"\[\s*\]", text) else None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def parse_list(text):
    """Model output -> list of valid entries (dicts with image in {1, 2} and a 4-number box); None if unparsable."""
    raw = json_entries(text)
    if raw is None:
        return None
    out = []
    for e in raw:
        b = e.get("bbox_2d")
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b) and e.get("image") in (1, 2):
            out.append({"change": e.get("change"), "image": e["image"], "bbox_2d": [float(v) for v in b]})
    return out


def list_metrics(rows, thr=0.3):
    """rows: dicts with entries (gold), ignore, pred (parsed list or None)."""
    tp = fp = fn = kind_ok = bad = 0
    yes = [0, 0]
    no = [0, 0]
    for r in rows:
        pred = r["pred"]
        if pred is None:
            bad += 1
            pred = []
        gt = list(r["entries"])
        used = [False] * len(gt)
        for p in pred:
            best, bj = 0.0, -1
            for j, g in enumerate(gt):
                if not used[j] and g["image"] == p["image"]:
                    v = iou(p["bbox_2d"], g["bbox_2d"])
                    if v > best:
                        best, bj = v, j
            if best >= thr:
                used[bj] = True
                tp += 1
                kind_ok += p["change"] == gt[bj]["change"]
            elif any(g["image"] == p["image"] and iou(p["bbox_2d"], g["bbox_2d"]) >= thr for g in r["ignore"]):
                pass
            else:
                fp += 1
        fn += used.count(False)
        c = yes if r["entries"] else no
        c[0] += bool(pred)
        c[1] += 1
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(n=len(rows), thr=thr, precision=round(prec, 4), recall=round(rec, 4),
                f1=round(2 * prec * rec / max(prec + rec, 1e-9), 4), kind_acc=round(kind_ok / max(tp, 1), 4),
                tp=tp, fp=fp, fn=fn, unparsable=bad, pair_sens=round(yes[0] / max(yes[1], 1), 4),
                pair_spec=round(1 - no[0] / max(no[1], 1), 4))


# ---------------------------------------------------------------------------------------------------------------
# CoDiff: every visual token receives the feature of its counterpart in the other image.

def map_tokens(H, size_a, size_b, grid_a, grid_b, token_px=32):
    """For every token of image a (grid_a = (rows, cols)) its real-valued (row, col) in the token grid of image b."""
    import torch
    ha, wa = grid_a
    hb, wb = grid_b
    dev = H.device
    ii, jj = torch.meshgrid(torch.arange(ha, device=dev), torch.arange(wa, device=dev), indexing="ij")
    xa = (jj.flatten().float() + 0.5) * (size_a[0] / wa)
    ya = (ii.flatten().float() + 0.5) * (size_a[1] / ha)
    q = H.float() @ torch.stack([xa, ya, torch.ones_like(xa)])
    den = torch.where(q[2].abs() < 1e-6, torch.full_like(q[2], 1e-6), q[2])
    col = (q[0] / den) / (size_b[0] / wb) - 0.5
    row = (q[1] / den) / (size_b[1] / hb) - 0.5
    ok = (q[2] > 0) & (row > -0.5) & (row < hb - 0.5) & (col > -0.5) & (col < wb - 0.5)
    return torch.stack([row, col], 1), ok


def make_codiff(d, r=1024):
    import torch
    from torch import nn

    class CoDiff(nn.Module):
        """token <- token + MLP([own, counterpart, own - counterpart, valid]); the last layer starts at zero."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.fc1 = nn.Linear(3 * d + 1, r)
            self.fc2 = nn.Linear(r, d)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def sample(self, src, grid_src, rc, ok):
            """Bilinear sample of the token grid src (rows*cols, d) at real-valued (row, col) positions rc (n, 2)."""
            hs, ws = grid_src
            g = torch.stack([(rc[:, 1] + 0.5) / ws * 2 - 1, (rc[:, 0] + 0.5) / hs * 2 - 1], -1).clamp(-2, 2)
            f = src.reshape(1, hs, ws, -1).permute(0, 3, 1, 2).float()
            o = torch.nn.functional.grid_sample(f, g.reshape(1, -1, 1, 2).float(), mode="bilinear",
                                                padding_mode="zeros", align_corners=False)
            return o[0, :, :, 0].t().to(src.dtype) * ok[:, None].to(src.dtype)

        def forward(self, own, other, grid_own, grid_other, rc, ok):
            a = self.norm(own)
            b = self.sample(self.norm(other), grid_other, rc, ok)
            m = ok[:, None].to(own.dtype)
            x = torch.cat([a, b, (a - b) * m, m], -1)
            return own + self.fc2(torch.nn.functional.gelu(self.fc1(x.to(self.fc1.weight.dtype)))).to(own.dtype)

    return CoDiff()


def attach_codiff(model, r=1024, weights=None):
    """Adds model.model.codiff and wraps get_image_features: when model.model._codiff_ctx = (Hs, sizes) is set, the visual tokens of the two images of every item are updated with their counterparts (both directions)."""
    import torch
    core = model.model
    d = model.config.text_config.hidden_size
    core.add_module("codiff", make_codiff(d, r))
    if weights is not None:
        from safetensors.torch import load_file
        sd = {}
        for f in sorted(Path(weights).glob("*.safetensors")):
            sd.update({k.split("codiff.", 1)[1]: v for k, v in load_file(str(f)).items() if "codiff." in k})
        assert sd, f"no CoDiff weights in {weights}"
        core.codiff.load_state_dict(sd)
    core._codiff_ctx = None
    orig = core.get_image_features

    def wrapped(pixel_values, image_grid_thw=None, **kw):
        out = orig(pixel_values, image_grid_thw, **kw)
        ctx = core._codiff_ctx
        if ctx is None:
            return out
        Hs, sizes = ctx
        emb = list(out.pooler_output)
        assert len(emb) == 2 * len(Hs), "CoDiff needs exactly two images per item"
        for b in range(len(Hs)):
            e1, e2 = emb[2 * b], emb[2 * b + 1]
            g1, g2 = [(int(g[1]) // 2, int(g[2]) // 2) for g in (image_grid_thw[2 * b], image_grid_thw[2 * b + 1])]
            H = torch.as_tensor(Hs[b], dtype=torch.float32, device=e1.device)
            w1, h1, w2, h2 = [float(v) for v in sizes[b]]
            rc21, ok21 = map_tokens(H, (w2, h2), (w1, h1), g2, g1)  # image-2 tokens in image 1
            rc12, ok12 = map_tokens(torch.linalg.inv(H), (w1, h1), (w2, h2), g1, g2)  # image-1 tokens in image 2
            emb[2 * b] = core.codiff(e1, e2, g1, g2, rc12, ok12)
            emb[2 * b + 1] = core.codiff(e2, e1, g2, g1, rc21, ok21)
        out.pooler_output = tuple(emb) if isinstance(out.pooler_output, tuple) else emb
        return out
    core.get_image_features = wrapped
    return core


def parse_scene(text):
    """Model output -> list of valid scene entries (video in {1, 2}, integer frame, 4-number box); None if unparsable."""
    raw = json_entries(text)
    if raw is None:
        return None
    out = []
    for e in raw:
        b = e.get("bbox_2d")
        if (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
                and e.get("video") in (1, 2) and isinstance(e.get("frame"), int)):
            out.append({"change": e.get("change"), "label": e.get("label"), "video": e["video"], "frame": e["frame"],
                        "bbox_2d": [float(v) for v in b]})
            c = e.get("check")
            if isinstance(c, dict) and c.get("video") in (1, 2) and isinstance(c.get("frame"), int):
                cb = c.get("bbox_2d")
                out[-1]["check"] = {"video": c["video"], "frame": c["frame"]}
                if isinstance(cb, list) and len(cb) == 4 and all(isinstance(v, (int, float)) for v in cb):
                    out[-1]["check"]["bbox_2d"] = [float(v) for v in cb]
    return out


def scene_metrics(rows, thr=0.3):
    """rows: dicts with gt (objects with boxes per "<video>:<frame>"), ignore, control, pred (parsed list or None)."""
    tp = fp = fn = kind_ok = bad = 0
    chg, quiet = [0, 0], [0, 0]
    per = {"added": [0, 0], "removed": [0, 0], "moved": [0, 0]}
    for r in rows:
        pred = r["pred"]
        if pred is None:
            bad += 1
            pred = []
        used = [False] * len(r["gt"])
        for p in pred:
            key = f"{p['video']}:{p['frame']}"
            best, bj = 0.0, -1
            for j, g in enumerate(r["gt"]):
                if key in g["boxes"]:
                    v = iou(p["bbox_2d"], g["boxes"][key])
                    if v > best:
                        best, bj = v, j
            if best >= thr:
                if not used[bj]:
                    used[bj] = True
                    tp += 1
                    kind_ok += p["change"] == r["gt"][bj]["kind"]
            elif any(g["video"] == p["video"] and g["frame"] == p["frame"] and iou(p["bbox_2d"], g["bbox_2d"]) >= thr
                     for g in r["ignore"]):
                pass
            else:
                fp += 1
        fn += used.count(False)
        for g, u in zip(r["gt"], used):
            per[g["kind"]][0] += u
            per[g["kind"]][1] += 1
        c = chg if r["gt"] else quiet
        c[0] += bool(pred)
        c[1] += 1
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(n=len(rows), thr=thr, precision=round(prec, 4), recall=round(rec, 4),
                f1=round(2 * prec * rec / max(prec + rec, 1e-9), 4), kind_acc=round(kind_ok / max(tp, 1), 4),
                tp=tp, fp=fp, fn=fn, unparsable=bad, pair_sens=round(chg[0] / max(chg[1], 1), 4),
                pair_spec=round(1 - quiet[0] / max(quiet[1], 1), 4),
                recall_by_kind={k: round(a / max(b, 1), 3) for k, (a, b) in per.items()})


def scene_metrics_ev(rows, thr=0.3):
    """Evidence-citing variant (items from build_sds_ev.py): required = objects with evidence (ev not None), ignore = ignore_ev."""
    tp = fp = fn = kind_ok = check_ok = bad = 0
    for r in rows:
        pred = r["pred"]
        if pred is None:
            bad += 1
            pred = []
        gts = [(g, e) for g, e in zip(r["gt"], r["ev"]) if e is not None]
        used = [False] * len(gts)
        for p in pred:
            key = f"{p['video']}:{p['frame']}"
            best, bj = 0.0, -1
            for j, (g, e) in enumerate(gts):
                if key in g["boxes"]:
                    v = iou(p["bbox_2d"], g["boxes"][key])
                    if v > best:
                        best, bj = v, j
            if best >= thr:
                if not used[bj]:
                    used[bj] = True
                    tp += 1
                    g, e = gts[bj]
                    kind_ok += p["change"] == g["kind"]
                    c = p.get("check")
                    if c:
                        ck = f"{c['video']}:{c['frame']}"
                        ok = ck in e
                        if ok and "bbox_2d" in c and ck in g["boxes"]:
                            ok = iou(c["bbox_2d"], g["boxes"][ck]) >= thr
                        check_ok += ok
            elif any(g["video"] == p["video"] and g["frame"] == p["frame"] and iou(p["bbox_2d"], g["bbox_2d"]) >= thr
                     for g in r["ignore_ev"]):
                pass
            else:
                fp += 1
        fn += used.count(False)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(n=len(rows), precision=round(prec, 4), recall=round(rec, 4), f1=round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                kind_acc=round(kind_ok / max(tp, 1), 4), check_acc=round(check_ok / max(tp, 1), 4), tp=tp, fp=fp, fn=fn,
                kind_ok=kind_ok, check_ok=check_ok, unparsable=bad)


def object_spans(text):
    """Character spans [start, end) of the top-level {...} objects of a JSON-list answer (nested braces allowed; an object cut off at the end of the text is skipped)."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                out.append((start, i + 1))
    return out


def entry_confidences(text, pieces, logps):
    """One confidence per top-level object of `text`: exp(mean token log-probability) over the tokens of the object and of the separator before it."""
    import math
    full = "".join(pieces)
    off = full.find(text[:20]) if text else -1
    if off < 0:
        return None
    starts, pos = [], 0
    for pc in pieces:
        starts.append(pos - off)
        pos += len(pc)
    out, prev_end = [], 0
    for a, b in object_spans(text):
        lp = [l for st, pc, l in zip(starts, pieces, logps) if st + len(pc) > prev_end and st < b]
        out.append(math.exp(sum(lp) / len(lp)) if lp else 0.0)
        prev_end = b
    return out


def attach_conf(text, pred, confs):
    """Adds "conf" to the parsed scene entries: parse_scene keeps the valid objects in order, so the k-th valid top-level object gets the k-th valid confidence."""
    if pred is None or confs is None:
        return False
    spans = object_spans(text)
    if len(spans) != len(confs):
        return False
    valid = []
    for (a, b), c in zip(spans, confs):
        try:
            e = json.loads(text[a:b])
        except Exception:
            continue
        if parse_scene("[" + text[a:b] + "]"):
            valid.append(c)
    if len(valid) != len(pred):
        return False
    for e, c in zip(pred, valid):
        e["conf"] = round(c, 5)
    return True


def scene_ap(rows, thr=0.3, ev=False):
    """Threshold-free scores for scene results whose entries carry "conf": average precision over all entries ranked by confidence (duplicates of an already found object and hits on ignore regions are skipped), and the best F1 over confidence thresholds."""
    cand, n_gt = [], 0
    for ri, r in enumerate(rows):
        gts = [g for g, e in zip(r["gt"], r["ev"]) if e is not None] if ev else r["gt"]
        ign = r["ignore_ev"] if ev else r["ignore"]
        n_gt += len(gts)
        used = [False] * len(gts)
        for p in sorted(r["pred"] or [], key=lambda p: -p.get("conf", 0.0)):
            key = f"{p['video']}:{p['frame']}"
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if key in g["boxes"]:
                    v = iou(p["bbox_2d"], g["boxes"][key])
                    if v > best:
                        best, bj = v, j
            if best >= thr:
                if used[bj]:
                    continue
                used[bj] = True
                cand.append((p.get("conf", 0.0), 1))
            elif any(g["video"] == p["video"] and g["frame"] == p["frame"] and iou(p["bbox_2d"], g["bbox_2d"]) >= thr for g in ign):
                continue
            else:
                cand.append((p.get("conf", 0.0), 0))
    cand.sort(key=lambda c: -c[0])
    tp = fp = 0
    prec, rec, best_f1 = [], [], 0.0
    for c, y in cand:
        tp += y
        fp += 1 - y
        prec.append(tp / (tp + fp))
        rec.append(tp / max(n_gt, 1))
        best_f1 = max(best_f1, 2 * tp / max(2 * tp + fp + (n_gt - tp), 1))
    ap, prev_r = 0.0, 0.0
    for i in range(len(prec)):  # all-point interpolation
        pmax = max(prec[i:])
        ap += (rec[i] - prev_r) * pmax
        prev_r = rec[i]
    return dict(ap=round(ap, 4), f1max=round(best_f1, 4), n_gt=n_gt, n_pred=len(cand))
