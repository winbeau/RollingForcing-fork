# Inference Benchmark Scripts

## `infer_bench.sh` — Batch Inference for Comparison Experiments

### Quick Start

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_bench.sh

# 4 GPUs in parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 bash scripts/infer_bench.sh --num_gpus 4

# Custom settings
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=29511 bash scripts/infer_bench.sh \
    --num_gpus 8 \
    --num_output_frames 120 \
    --output_dir videos/my_experiment
```

`infer_bench.sh` does not choose GPUs by itself. GPU visibility is controlled externally via `CUDA_VISIBLE_DEVICES`, and `--num_gpus` only controls the number of worker processes (`torchrun --nproc_per_node`).

For multi-GPU runs, `MASTER_PORT` is also controlled externally and passed through to `torchrun --master_port`. If unset, the script uses `29501`.

If the visible GPU count does not match `--num_gpus`, the script exits before launching inference.

### Script Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/rolling_forcing_dmd.yaml` | Config file path |
| `--checkpoint` | `checkpoints/rolling_forcing_dmd.pt` | Model checkpoint path |
| `--prompts` | `prompts/MovieGenVideoBench_num32.txt` | Prompt file (one prompt per line) |
| `--output_dir` | `videos/MovieGenVideoBench_num32` | Output directory |
| `--num_gpus` | `1` | Number of worker processes for inference; must match the number of GPUs visible through `CUDA_VISIBLE_DEVICES` |
| `--num_output_frames` | `120` | Number of **latent** frames to generate (see below) |

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | unset | Controls which GPUs are visible to the script; visible GPU count must equal `--num_gpus` |
| `MASTER_PORT` | `29501` | Passed to `torchrun --master_port` for multi-GPU runs |

### Output Structure

```
videos/MovieGenVideoBench_num32/
├── prompts.csv          # index,prompt mapping (index starting from 0)
├── video_000.mp4
├── video_001.mp4
├── ...
└── video_031.mp4
```

`prompts.csv` format:

```csv
index,prompt
0,"A stylish woman strolls down a bustling Tokyo street..."
1,"A stunning mid-afternoon landscape photograph..."
```

Video filenames (`video_XXX.mp4`) correspond 1:1 with the CSV index.

---

## Video Generation Specifications

### Resolution & Frame Rate

| Parameter | Value |
|---|---|
| Width | **832** px |
| Height | **480** px |
| Aspect ratio | 832:480 = 26:15 (~16:9) |
| Frame rate (FPS) | **16** fps |
| Video codec | H.264 (via `torchvision.io.write_video`) |
| Pixel value range | [0, 255] uint8 |

### Frame Count & Duration

The `--num_output_frames` argument specifies the number of **latent** frames. The VAE temporal stride is **4**, so the relationship between latent frames and pixel frames is:

```
pixel_frames = (latent_frames - 1) * 4 + 1
duration     = pixel_frames / 16 fps
```

| `--num_output_frames` (latent) | Pixel frames | Duration |
|---|---|---|
| 21 (default of inference.py) | 81 | ~5.1 s |
| 42 | 165 | ~10.3 s |
| 63 | 249 | ~15.6 s |
| 84 | 333 | ~20.8 s |
| **120** (script default) | **477** | **~29.8 s** |

Constraint: `num_output_frames` must be divisible by `num_frame_per_block` (= **3**). Valid values: 21, 24, 27, ..., 120, ...

### Latent Space

| Parameter | Value |
|---|---|
| Latent channels | 16 |
| Latent spatial size | 60 x 104 |
| VAE stride (t, h, w) | (4, 8, 8) |
| Latent shape per sample | `[num_output_frames, 16, 60, 104]` |
| Patch size | (1, 2, 2) |

Spatial relationship: 480 / 8 = 60, 832 / 8 = 104.

### VAE

| Parameter | Value |
|---|---|
| Checkpoint | `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` |
| Normalization mean | 16-dim per-channel vector (see `wan/configs/`) |
| Normalization std | 16-dim per-channel vector (see `wan/configs/`) |
| Output range | [-1, 1] (clamped), then scaled to [0, 255] |

---

## Model Architecture

### Generator (Inference Model) — Wan2.1-T2V-1.3B (Causal)

| Parameter | Value |
|---|---|
| Model | `CausalWanModel` (causal variant of Wan2.1) |
| Total parameters | **~1.3B** |
| Hidden dimension | 1536 |
| FFN dimension | 8960 |
| Attention heads | 12 |
| Head dimension | 128 (= 1536 / 12) |
| Transformer layers | 30 |
| QK normalization | Enabled |
| Cross-attention normalization | Enabled |
| Precision | **bfloat16** |

### Teacher Model (Training Only) — Wan2.1-T2V-14B

| Parameter | Value |
|---|---|
| Hidden dimension | 5120 |
| FFN dimension | 13824 |
| Attention heads | 40 |
| Transformer layers | 40 |

The teacher model is **not loaded during inference** — only the 1.3B generator is used.

### Text Encoder — UMT5-XXL

| Parameter | Value |
|---|---|
| Model | `umt5_xxl` (google/umt5-xxl) |
| Precision | bfloat16 |
| Max text length | 512 tokens |
| Text embedding dimension | 4096 |

---

## Checkpoint & Weights

| Parameter | Value |
|---|---|
| Checkpoint file | `checkpoints/rolling_forcing_dmd.pt` |
| Weights used | **EMA** (Exponential Moving Average), via `--use_ema` |
| EMA weight (training) | 0.99 |
| EMA start step (training) | 200 |
| Non-EMA key in checkpoint | `generator` |
| EMA key in checkpoint | `generator_ema` |
| Training init checkpoint | `checkpoints/ode_init.pt` |
| Training method | DMD (Distribution Matching Distillation) |

The `--use_ema` flag loads the `generator_ema` state dict from the checkpoint. EMA weights are generally smoother and produce higher-quality results. The script uses EMA by default.

---

## Diffusion & Sampling Parameters

### Noise Schedule (Flow Matching)

| Parameter | Value | Source |
|---|---|---|
| Scheduler | `FlowMatchScheduler` | `utils/scheduler.py` |
| Num train timesteps | 1000 | config |
| Timestep shift | **5.0** | `model_kwargs.timestep_shift` |
| Sigma max | 1.0 | scheduler default |
| Sigma min | 0.003 / 1.002 ≈ 0.00299 | scheduler default |
| Extra one step | True | `wan_wrapper.py` |

The flow matching sigma schedule is:

```
sigmas = linspace(sigma_min, sigma_max, 1001)[:-1]   # 1000 values
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # shift=5.0
timesteps = sigmas * 1000
```

### Denoising Steps

| Parameter | Value |
|---|---|
| Nominal denoising steps | `[1000, 800, 600, 400, 200]` (**5 steps**) |
| Warp denoising steps | **True** |

With `warp_denoising_step=true`, the nominal timesteps are remapped through the flow-match sigma schedule:

```python
warped_steps = scheduler.timesteps[1000 - nominal_steps]
```

This maps the evenly-spaced nominal steps to the actual sigma values used during denoising.

### Guidance

| Parameter | Value |
|---|---|
| Guidance scale (CFG) | **3.0** |
| Negative prompt | Chinese quality-degradation descriptors (see config) |

Negative prompt (from config):
> 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走

---

## Rolling Forcing Specific Parameters

### Frame Blocking

| Parameter | Value |
|---|---|
| Frames per block (`num_frame_per_block`) | **3** |
| Independent first frame | False |
| Uniform timestep per block (`same_step_across_blocks`) | True |

Frames are grouped into blocks of 3 latent frames for joint denoising. With `--num_output_frames 120`, this yields **40 blocks**.

### Rolling Window

| Parameter | Value |
|---|---|
| Rolling window length | **5 blocks** (= number of denoising steps) |
| Total windows | `num_blocks + window_length - 1` = 40 + 5 - 1 = **44 windows** |
| Context noise | 0 (clean context, no noise on previously denoised blocks) |

The rolling window slides one block at a time: each window covers up to 5 consecutive blocks. Blocks within the window are assigned decreasing noise levels from the denoising step list.

### KV Cache & Attention Sink

| Parameter | Value |
|---|---|
| Tokens per frame | 1560 |
| Transformer blocks with cache | 30 |
| Cross-attention cache size | 512 tokens (text length) |
| Attention sink | Keeps first block in KV cache during rolling |
| Local attention size | -1 (global attention, no local windowing) |

---

## Random Seed

| Parameter | Value |
|---|---|
| Default seed | **0** |
| Seed per GPU (distributed) | `seed + local_rank` |
| Num samples per prompt | 1 |

For reproducibility, each GPU gets `seed + local_rank`. With 4 GPUs, seeds are 0, 1, 2, 3. Since `DistributedSampler(shuffle=False)` assigns prompts round-robin, each prompt's seed is deterministic given the GPU count.

---

## Multi-GPU Notes

- Uses `torchrun` with NCCL backend for distributed inference
- GPU selection is controlled externally by `CUDA_VISIBLE_DEVICES`
- `MASTER_PORT` is controlled externally and passed to `torchrun --master_port` (default: `29501`)
- `--num_gpus` must equal the number of visible GPUs or the script will fail fast
- `DistributedSampler(shuffle=False, drop_last=True)` distributes prompts evenly
- **Important**: if `num_prompts % num_gpus != 0`, the last few prompts will be **skipped** (due to `drop_last=True`). For 32 prompts, use 1, 2, 4, or 8 GPUs
- Each GPU loads the full model independently (no model parallelism)
- VRAM per GPU: ~8-10 GB for 1.3B model in bfloat16

---

## Summary Table for Quick Reference

| Item | Value |
|---|---|
| Resolution | 480 x 832 |
| FPS | 16 |
| Latent frames | 120 |
| Pixel frames | 477 |
| Duration | ~29.8 s |
| Model | Wan2.1-T2V-1.3B (Causal) |
| Weights | EMA (weight=0.99) |
| Precision | bfloat16 |
| Denoising steps | 5 (flow matching, warped) |
| Guidance scale | 3.0 |
| Frames per block | 3 |
| Seed | 0 |
| Scheduler | FlowMatchScheduler (shift=5.0) |
