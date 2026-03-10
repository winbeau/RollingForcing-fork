# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rolling Forcing is a research implementation for real-time, streaming long video generation from text prompts. It uses a rolling-window denoising strategy with attention sinks on top of Wan2.1 base models. Built upon the Self-Forcing codebase.

Paper: https://arxiv.org/abs/2509.25161

## Setup

```bash
conda create -n rolling_forcing python=3.10 -y
conda activate rolling_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Common Commands

### Inference
```bash
python inference.py \
    --config_path configs/rolling_forcing_dmd.yaml \
    --output_folder videos/rolling_forcing_dmd \
    --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
    --data_path prompts/example_prompts.txt \
    --num_output_frames 126 \
    --use_ema
```

### Training (8 GPUs, single machine)
```bash
torchrun --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint 127.0.0.1:29500 \
  train.py \
  -- \
  --config_path configs/rolling_forcing_dmd.yaml \
  --logdir logs/rolling_forcing_dmd
```

### Gradio Demo
```bash
python app.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt
```

## Architecture

### Entry Points
- `train.py` — Training entry point. Loads config, selects trainer type, launches distributed training.
- `inference.py` — Batch inference. Loads pipeline, processes prompts, saves MP4 videos.
- `app.py` — Gradio web demo for interactive single-GPU inference.

### Core Layers

**`model/`** — Model definitions, each subclass of `BaseModel` (in `base.py`):
- `dmd.py` (DMD — Distribution Matching Distillation) — primary training method
- `diffusion.py`, `gan.py`, `sid.py`, `causvid.py`, `ode_regression.py` — alternative training variants
- `BaseModel` initializes four components: generator (trainable causal model), real score model (frozen teacher), fake score model (trainable discriminator), text encoder, and VAE.

**`pipeline/`** — Inference and training pipelines:
- `rolling_forcing_inference.py` — `CausalInferencePipeline`: real-time generation with KV cache and attention sink
- `rolling_forcing_training.py` — `RollingForcingTrainingPipeline`: rolling window denoising for training
- `causal_diffusion_inference.py`, `bidirectional_*.py` — alternative inference strategies

**`trainer/`** — Training loops:
- `distillation.py` — `ScoreDistillationTrainer` (DMD): the main trainer, alternates generator and discriminator updates
- `diffusion.py`, `gan.py`, `ode.py` — alternative trainer implementations
- Trainer type is selected by `trainer` field in config YAML

**`wan/`** — Modified Wan2.1 base model (mostly upstream code):
- `modules/model.py` — `WanModel`: DiT (Diffusion Transformer) backbone
- `modules/causal_model.py` — `CausalWanModel`: causal variant with autoregressive masking
- `modules/vae.py`, `modules/t5.py`, `modules/attention.py` — VAE, T5 text encoder, flash attention

**`utils/`** — Shared utilities:
- `wan_wrapper.py` — Wrapper classes (`WanTextEncoder`, `WanVAEWrapper`, `WanDiffusionWrapper`) that adapt Wan models for this framework
- `dataset.py` — Dataset classes (text prompts, image-text pairs, LMDB)
- `distributed.py` — FSDP setup and sharding strategies
- `loss.py` — Prediction types (X0Pred, VPred, NoisePred, FlowPred)
- `scheduler.py` — Diffusion noise scheduler

### Configuration System
- OmegaConf-based. `configs/default_config.yaml` provides defaults, merged with specific configs like `configs/rolling_forcing_dmd.yaml`.
- Key parameters: `trainer` (selects trainer class), `distribution_loss` (loss variant), `num_frame_per_block` (frames denoised together), `denoising_step_list` (noise schedule).

### Key Concepts
- **Rolling Window**: Overlapping denoising passes across frame blocks to reduce error accumulation
- **Attention Sink**: Persistent attention tokens that anchor global context across long sequences
- **KV Cache**: Cached key/value pairs for efficient autoregressive generation
- **Frame Blocking**: `num_frame_per_block` groups frames for joint denoising (default: 3)
- **DMD Training**: Generator produces samples, real/fake score models provide distillation signal; `dfake_gen_update_ratio` controls discriminator-to-generator update frequency

### Model Checkpoints
- Wan2.1 base models downloaded from HuggingFace to `wan_models/`
- Rolling Forcing checkpoints in `checkpoints/`
- Training uses ODE-initialized checkpoint (`checkpoints/ode_init.pt`) and 14B teacher model
- Inference uses 1.3B model with DMD-distilled checkpoint
