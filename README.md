# STILL: Is It Still There?

Code for **"Is It Still There? Place-Registered Verification Makes VLMs Reliable Scene-Change Detectors"**
(Jinjie Xie, Baihua Li, Qinggang Meng, Loughborough University).

Multi-image VLMs are poor at telling which objects changed between two visits to a scene. They can say what a frame
shows, but they cannot find where an object was in the other capture, so they report objects that only one viewpoint
shows and miss most removed and moved objects. STILL turns change detection into a set of questions asked at the
place where each object was. Feed-forward 3D reconstruction ($\pi^3$) registers every change hypothesis to its place in
the other capture, and one VLM answers, from a scene view and a local place view, whether the object is still there.
The verifier is trained on questions of exactly this form.

On the SceneDiff benchmark (official protocol) STILL raises obj/sc AP from 24.0 to 37.8 on SD-V and from 11.6 to 20.6
on the egocentric kitchen subset SD-K, which it never sees in training.

## Setup

Two Python environments are used.

1. **Verifier** (training and scoring of Qwen3.5-9B / Qwen3-VL-8B): Python 3.11 and `requirements.txt`.
2. **SceneDiff** (SceneDiff proposals, $\pi^3$ registration, official evaluation): follow the setup of the
   [SceneDiff repository](https://github.com/yuqunw/scene_diff) and clone it into `ext/scene_diff_code`.
   DINOv3 weights come from Hugging Face and need access approval.

Copy `scripts/env.example.sh` to `scripts/env.sh` and set the two Python executables, the base models and the GPUs.

### Data

Everything external lives in `ext/`.

| Path | Source |
|---|---|
| `ext/scenediff/scenediff_benchmark` | SceneDiff benchmark, Hugging Face dataset `yuqun/SceneDiff` |
| `ext/scene_diff_code` | `git clone https://github.com/yuqunw/scene_diff` |
| `ext/kc3d` | KC-3D synthetic pairs from [CYWS-3D](https://github.com/ragavsachdeva/CYWS-3D) (with `data_split.pkl`) |
| `ext/rc3d/part1-4` | RC-3D test pairs from CYWS-3D |

Models are loaded from Hugging Face: `Qwen/Qwen3.5-9B`, `Qwen/Qwen3-VL-8B-Instruct`,
`IDEA-Research/grounding-dino-tiny`, `yyfz233/Pi3` and `facebook/dinov3-vith16plus-pretrain-lvd1689m`.

`data/dev_scenes.json` lists the 10 SD-V scenes we held out from training to choose the operating point.
`build_protocol2.py` selects the same scenes.

## Running

```bash
./run/01_prepare_data.sh          # items, SceneDiff proposals, pi^3 places and place-view crops
./run/02_train.sh STILL_k0 0      # verifier, one epoch on 4 GPUs (about 9 h on A100)
./run/03_evaluate.sh STILL_k0     # SD-V, SD-K and RC-3D
./run/04_official.sh STILL_k0     # SceneDiff official protocol
```

The paper reports three runs (seeds 0, 1, 2). Results go to `res/`, logs to `logs/`.

- `logs/fuse_<run>sd.json` holds the SD-V operating point and the dev and test scores under the VLM-output protocol.
- `sdk_at_sdv_op.py` prints SD-K at that operating point.
- `rc3d_ap.py` prints RC-3D AP50.
- `04_official.sh` writes the official metrics to `ext/sd_pred/eval_*.json`.

`boot_fast.py` and `boot_f1runs.py` give scene-bootstrap intervals for paired comparisons.

## Code

All scripts are in `scripts/` and read and write paths relative to the repository root (`STILL_ROOT`).

| Step | Scripts |
|---|---|
| Items and controls | `build_sd.py`, `build_sds.py`, `build_sds_hc.py`, `build_protocol2.py`, `build_sdk.py`, `build_rc3d.py`, `build_kc3d.py` |
| Proposals | `run_scenediff.py` (SceneDiff change regions), `proposals.py` (Grounding DINO) |
| Training queries | `build_verify.py`, `build_twoview.py`, `build_kc3d_queries.py`, `build_sdiff_queries.py` |
| Place registration | `pi3_anchor.py`, `build_crops.py` |
| Verifier | `still_sft.py` (training), `score_ver.py` (scoring), `combine_views.py` |
| Scoring and metrics | `fuse_sdiff.py`, `sdk_at_sdv_op.py`, `ver_pooled.py`, `rc3d_from_tv.py`, `rc3d_project.py`, `rc3d_ap.py`, `boot_fast.py`, `boot_f1runs.py`, `still_common.py` |
| Official protocol | `official_frames_ff.py`, `official_frames.py`, `run_official.py`, `official_rescore.py` |
| Holistic VLM baselines | `still_gen.py`, `rescore_gen.py`, `reground.py`, `build_items_sd.py`, `build_items_sdl.py`, `build_items_sdsv.py` |
| Registration ablation | `sift_anchor.py`; `STILL_CROP_UNREG=1` in `build_crops.py` keeps the box unmapped |
| Learned registration (Sec. 6) | `build_locate.py`, `gen_locate.py`, `build_refine.py`, `gen_refine.py`, `build_mask.py`, `build_rl_ref.py`, `still_rl_ref.py`, `place_found.py` |

The holistic baseline fine-tunes the same base model with `still_sft.py --arm V` on the scene items
(`items_sds_train.jsonl`, three passes) together with the two-frame SceneDiff items from `build_items_sd.py`,
`build_items_sdl.py` and `build_items_sdsv.py`, and is scored with `still_gen.py`. For the Grounding DINO proposer, build the queries with `build_verify.py p2`, register them with
`pi3_anchor.py` and crop them with `build_crops.py` (`STILL_CROP_PI3=1`).

## Citation

```bibtex
@article{xie2026still,
  title   = {Is It Still There? Place-Registered Verification Makes {VLMs} Reliable Scene-Change Detectors},
  author  = {Xie, Jinjie and Li, Baihua and Meng, Qinggang},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

We build on [SceneDiff](https://github.com/yuqunw/scene_diff), [$\pi^3$](https://github.com/yyfz/Pi3),
[CYWS-3D](https://github.com/ragavsachdeva/CYWS-3D) and the Qwen models.

## License

MIT.
