# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'headkv_attention',
    'headkv_window_attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def headkv_attention(
    q,
    k,
    v,
    kv_cache,
    current_start=None,
    grid_sizes=None,
    freqs=None,
    start_frame=0,
    prompt_v=None,
    cache_update_mode="default",
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=True,
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    HeadKV attention using FlashAttention varlen interface.

    Args:
        q, k, v: [B, L, H, D]
        kv_cache: HeadKVCache
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, h, d = q.shape
    out_dtype = q.dtype
    drop_head_mask = None
    raw_drop_mask = getattr(kv_cache, "drop_head_mask", None)
    if raw_drop_mask is not None:
        drop_head_mask = torch.as_tensor(raw_drop_mask, dtype=torch.bool, device=q.device)
        if drop_head_mask.numel() != h:
            drop_head_mask = None
    frame_seqlen = None
    if grid_sizes is not None:
        frame_tokens = (grid_sizes[:, 1] * grid_sizes[:, 2]).to(torch.long)
        if torch.any(frame_tokens <= 0):
            raise ValueError(f"Invalid frame token sizes: {frame_tokens.tolist()}")
        if torch.unique(frame_tokens).numel() != 1:
            raise ValueError(f"Mixed frame token sizes in batch are not supported: {frame_tokens.tolist()}")
        frame_seqlen = int(frame_tokens[0].item())

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    def _build_region_mask(frame_ids: torch.Tensor, sync_t: int, region: str) -> torch.Tensor:
        region = str(region).strip().lower()
        if region in {"", "none", "off"}:
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        def _token_mask(token: str) -> torch.Tensor:
            token = token.strip().lower()
            if token == "sink1":
                return frame_ids == 0
            if token.startswith("recent"):
                n_str = token[len("recent"):]
                n = int(n_str) if n_str.isdigit() else 0
                if n <= 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                low = max(0, sync_t - (n - 1))
                return (frame_ids >= low) & (frame_ids <= sync_t)
            if token.startswith("lag"):
                lag_str = token[len("lag"):]
                lag = int(lag_str) if lag_str.isdigit() else -1
                if lag < 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                target = max(0, sync_t - lag)
                return frame_ids == target
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        out = torch.zeros_like(frame_ids, dtype=torch.bool)
        for tok in region.split("+"):
            if tok.strip():
                out |= _token_mask(tok)
        return out

    def _apply_soft_ablate_to_k_flat(
        k_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        k_frame_ids_flat: torch.Tensor | None,
        chunk_start_token: int,
    ) -> torch.Tensor:
        if k_frame_ids_flat is None or frame_seqlen is None or frame_seqlen <= 0:
            return k_flat_chunk

        raw_mask = getattr(kv_cache, "soft_ablate_head_mask", None)
        if raw_mask is None:
            return k_flat_chunk
        soft_head_mask = torch.as_tensor(raw_mask, dtype=torch.bool, device=k_flat_chunk.device)
        if soft_head_mask.numel() != h or not torch.any(soft_head_mask):
            return k_flat_chunk

        region = str(getattr(kv_cache, "soft_ablate_region", "none"))
        if region.strip().lower() in {"", "none", "off"}:
            return k_flat_chunk

        scale = float(getattr(kv_cache, "soft_ablate_scale", 1.0))
        if scale >= 0.9999:
            return k_flat_chunk
        if scale < 0.0:
            scale = 0.0

        sync_t = int(chunk_start_token // frame_seqlen)
        for b_idx in range(b):
            for h_idx in range(h):
                if not bool(soft_head_mask[h_idx].item()):
                    continue
                seq_idx = b_idx * h + h_idx
                ks = int(cu_seqlens_k_chunk[seq_idx].item())
                ke = int(cu_seqlens_k_chunk[seq_idx + 1].item())
                if ke <= ks:
                    continue
                local_ids = k_frame_ids_flat[ks:ke].to(dtype=torch.long)
                select = _build_region_mask(local_ids, sync_t=sync_t, region=region)
                if not torch.any(select):
                    continue
                local_k = k_flat_chunk[ks:ke]
                local_k[select] = local_k[select] * scale
                k_flat_chunk[ks:ke] = local_k
        return k_flat_chunk

    kv_cache.update(
        k,
        v,
        current_start=current_start,
        grid_sizes=grid_sizes,
        freqs=freqs,
        start_frame=start_frame,
        prompt_v=prompt_v,
        cache_update_mode=cache_update_mode,
    )
    if fa_version is not None and fa_version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    def run_varlen(
        q_chunk: torch.Tensor,
        k_flat_chunk: torch.Tensor,
        v_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        max_seqlen_k_chunk: int,
        cu_seqlens_q_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lq_chunk = q_chunk.shape[1]
        q_flat_chunk = q_chunk.transpose(1, 2).reshape(b * h * lq_chunk, d)
        q_flat_chunk = half(q_flat_chunk).unsqueeze(1)
        k_flat_chunk = half(k_flat_chunk).unsqueeze(1)
        v_flat_chunk = half(v_flat_chunk).unsqueeze(1)

        if q_scale is not None:
            q_flat_chunk = q_flat_chunk * q_scale

        q_flat_chunk = q_flat_chunk.to(v_flat_chunk.dtype)
        k_flat_chunk = k_flat_chunk.to(v_flat_chunk.dtype)

        if cu_seqlens_q_override is not None:
            cu_seqlens_q_chunk = cu_seqlens_q_override
        else:
            cu_seqlens_q_chunk = torch.arange(
                0, (b * h + 1) * lq_chunk, step=lq_chunk, dtype=torch.int32, device=q.device
            )

        if (fa_version is None or fa_version == 3) and FLASH_ATTN_3_AVAILABLE:
            out_chunk = flash_attn_interface.flash_attn_varlen_func(
                q=q_flat_chunk,
                k=k_flat_chunk,
                v=v_flat_chunk,
                cu_seqlens_q=cu_seqlens_q_chunk,
                cu_seqlens_k=cu_seqlens_k_chunk,
                max_seqlen_q=lq_chunk,
                max_seqlen_k=max_seqlen_k_chunk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic
            )[0]
        else:
            assert FLASH_ATTN_2_AVAILABLE
            out_chunk = flash_attn.flash_attn_varlen_func(
                q=q_flat_chunk,
                k=k_flat_chunk,
                v=v_flat_chunk,
                cu_seqlens_q=cu_seqlens_q_chunk,
                cu_seqlens_k=cu_seqlens_k_chunk,
                max_seqlen_q=lq_chunk,
                max_seqlen_k=max_seqlen_k_chunk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                deterministic=deterministic
            )

        out = out_chunk.squeeze(1).reshape(b, h, lq_chunk, d).transpose(1, 2)
        if drop_head_mask is not None and torch.any(drop_head_mask):
            out[:, :, drop_head_mask, :] = 0
        return out

    use_decoupled = (
        getattr(kv_cache, "post_prune_rope", False)
        and getattr(kv_cache, "sink_grid_decoupling", False)
        and hasattr(kv_cache, "get_decoupled_flat_kv")
    )
    if use_decoupled:
        if freqs is None:
            raise ValueError("freqs is required when sink_grid_decoupling=True")
        if grid_sizes is None:
            raise ValueError("grid_sizes is required when sink_grid_decoupling=True")
        if frame_seqlen is None:
            raise ValueError("frame_seqlen is required when sink_grid_decoupling=True")
        if lq % frame_seqlen != 0:
            raise ValueError(f"q length {lq} must be divisible by frame_seqlen {frame_seqlen}.")

        out_buf = torch.empty(b, lq, h, d, device=q.device, dtype=out_dtype)
        base_start = int(current_start or 0)
        cu_seqlens_q_fixed = torch.arange(
            0, (b * h + 1) * frame_seqlen, step=frame_seqlen,
            dtype=torch.int32, device=q.device,
        )
        for offset in range(0, lq, frame_seqlen):
            q_chunk = q[:, offset:offset + frame_seqlen]
            if hasattr(kv_cache, "get_decoupled_flat_kv_and_frames"):
                k_flat, v_flat, cu_seqlens_k, max_seqlen_k, k_frame_ids_flat = kv_cache.get_decoupled_flat_kv_and_frames(
                    current_start=base_start + offset,
                    grid_sizes=grid_sizes,
                    freqs=freqs,
                )
            else:
                k_flat, v_flat, cu_seqlens_k, max_seqlen_k = kv_cache.get_decoupled_flat_kv(
                    current_start=base_start + offset,
                    grid_sizes=grid_sizes,
                    freqs=freqs,
                )
                k_frame_ids_flat = None
            k_flat = _apply_soft_ablate_to_k_flat(
                k_flat_chunk=k_flat,
                cu_seqlens_k_chunk=cu_seqlens_k,
                k_frame_ids_flat=k_frame_ids_flat,
                chunk_start_token=base_start + offset,
            )
            out_buf[:, offset:offset + frame_seqlen] = run_varlen(q_chunk, k_flat, v_flat, cu_seqlens_k, max_seqlen_k, cu_seqlens_q_override=cu_seqlens_q_fixed)
        return out_buf

    k_frame_ids_flat = None
    if getattr(kv_cache, "post_prune_rope", False):
        if hasattr(kv_cache, "get_flat_kv_and_pos"):
            k_flat, v_flat, cu_seqlens_k, max_seqlen_k, pos_ids = kv_cache.get_flat_kv_and_pos()
            if freqs is None:
                raise ValueError("freqs is required when post_prune_rope=True")
            if hasattr(kv_cache, "apply_rope_to_flat_k"):
                k_flat = kv_cache.apply_rope_to_flat_k(k_flat, pos_ids, freqs=freqs)
                k_frame_ids_flat = pos_ids[:, 0].to(dtype=torch.long)
            else:
                raise ValueError("kv_cache must provide apply_rope_to_flat_k for post-prune RoPE.")
        else:
            raise ValueError("kv_cache must provide get_flat_kv_and_pos or get_decoupled_flat_kv for post-prune RoPE.")
    else:
        k_flat, v_flat, cu_seqlens_k, max_seqlen_k = kv_cache.get_flat_kv()
        if frame_seqlen is not None and frame_seqlen > 0 and hasattr(kv_cache, "global_end_index"):
            k_frame_ids_flat = torch.empty((k_flat.shape[0],), dtype=torch.long, device=q.device)
            for b_idx in range(b):
                global_end = int(kv_cache.global_end_index[b_idx])
                for h_idx in range(h):
                    seq_idx = b_idx * h + h_idx
                    ks = int(cu_seqlens_k[seq_idx].item())
                    ke = int(cu_seqlens_k[seq_idx + 1].item())
                    if ke <= ks:
                        continue
                    seq_len = ke - ks
                    global_start = max(0, global_end - seq_len)
                    token_idx = torch.arange(global_start, global_end, device=q.device, dtype=torch.long)
                    k_frame_ids_flat[ks:ke] = token_idx // frame_seqlen

    k_flat = _apply_soft_ablate_to_k_flat(
        k_flat_chunk=k_flat,
        cu_seqlens_k_chunk=cu_seqlens_k,
        k_frame_ids_flat=k_frame_ids_flat,
        chunk_start_token=int(current_start or 0),
    )

    out = run_varlen(q, k_flat, v_flat, cu_seqlens_k, max_seqlen_k)
    return out.type(out_dtype)


def headkv_window_attention(
    q,
    k,
    v,
    kv_cache,
    current_start=None,
    grid_sizes=None,
    freqs=None,
    start_frame=0,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    HeadKV-aware window attention for rolling forcing denoising pass.

    Collects per-head ragged historical KV from cache (read-only, no update),
    combines with uniform window KV, and runs flash_attn_varlen_func with
    causal=False for bidirectional attention within the window.

    Args:
        q: [B, Lq, H, D] — full window query (already roped)
        k: [B, Lk, H, D] — full window key (already roped)
        v: [B, Lk, H, D] — full window value
        kv_cache: HeadKVCache — read-only, NOT updated
        current_start: int — token position of window start
        grid_sizes: [B, 3] — (F, H, W) for RoPE
        freqs: RoPE frequencies
        start_frame: int — frame index of window start
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, h, d = q.shape
    _, lk, _, _ = k.shape
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Step 1: Collect per-head ragged historical KV from cache
    post_prune_rope = getattr(kv_cache, "post_prune_rope", False)
    use_decoupled = (
        post_prune_rope
        and getattr(kv_cache, "sink_grid_decoupling", False)
        and hasattr(kv_cache, "get_decoupled_flat_kv")
    )

    if use_decoupled:
        # Decoupled mode: dynamic RoPE mapping relative to window start
        hist_k_flat, hist_v_flat, hist_cu_seqlens, hist_max_seqlen = kv_cache.get_decoupled_flat_kv(
            current_start=current_start,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
    elif post_prune_rope:
        # Post-prune mode: get unroped KV + positions, apply standard RoPE
        if hasattr(kv_cache, "get_flat_kv_and_pos"):
            hist_k_flat, hist_v_flat, hist_cu_seqlens, hist_max_seqlen, pos_ids = kv_cache.get_flat_kv_and_pos()
            if hasattr(kv_cache, "apply_rope_to_flat_k") and hist_k_flat.numel() > 0:
                hist_k_flat = kv_cache.apply_rope_to_flat_k(hist_k_flat, pos_ids, freqs=freqs)
        else:
            hist_k_flat, hist_v_flat, hist_cu_seqlens, hist_max_seqlen = kv_cache.get_flat_kv()
    else:
        # Pre-prune mode: KV already has RoPE baked in
        hist_k_flat, hist_v_flat, hist_cu_seqlens, hist_max_seqlen = kv_cache.get_flat_kv()

    # Step 2: Flatten window KV to per-head format
    # k: [B, Lk, H, D] → [B, H, Lk, D] → [B*H, Lk, D]
    window_k = k.transpose(1, 2).reshape(b * h, lk, d)
    window_v = v.transpose(1, 2).reshape(b * h, lk, d)

    # Step 3: Concatenate historical (per-head ragged) + window (uniform) KV
    combined_parts_k = []
    combined_parts_v = []
    combined_lens = []

    for i in range(b * h):
        hist_start = int(hist_cu_seqlens[i].item())
        hist_end = int(hist_cu_seqlens[i + 1].item())
        hist_len = hist_end - hist_start

        if hist_len > 0:
            combined_parts_k.append(hist_k_flat[hist_start:hist_end])
            combined_parts_v.append(hist_v_flat[hist_start:hist_end])
        combined_parts_k.append(window_k[i])
        combined_parts_v.append(window_v[i])
        combined_lens.append(hist_len + lk)

    combined_k_flat = half(torch.cat(combined_parts_k, dim=0)).unsqueeze(1)
    combined_v_flat = half(torch.cat(combined_parts_v, dim=0)).unsqueeze(1)

    max_seqlen_k = max(combined_lens) if combined_lens else 0
    cu_seqlens_k = torch.zeros(b * h + 1, dtype=torch.int32, device=q.device)
    for i, l in enumerate(combined_lens):
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + l

    # Step 4: Flatten query
    # q: [B, Lq, H, D] → [B, H, Lq, D] → [B*H*Lq, D] → [B*H*Lq, 1, D]
    q_flat = q.transpose(1, 2).reshape(b * h * lq, d)
    q_flat = half(q_flat).unsqueeze(1)

    if q_scale is not None:
        q_flat = q_flat * q_scale

    q_flat = q_flat.to(combined_v_flat.dtype)
    combined_k_flat = combined_k_flat.to(combined_v_flat.dtype)

    cu_seqlens_q = torch.arange(
        0, (b * h + 1) * lq, step=lq, dtype=torch.int32, device=q.device
    )

    # Step 5: Run varlen attention with causal=False (bidirectional within window)
    if fa_version is not None and fa_version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    if (fa_version is None or fa_version == 3) and FLASH_ATTN_3_AVAILABLE:
        out = flash_attn_interface.flash_attn_varlen_func(
            q=q_flat,
            k=combined_k_flat,
            v=combined_v_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=False,
            deterministic=deterministic,
        )[0]
    else:
        assert FLASH_ATTN_2_AVAILABLE
        out = flash_attn.flash_attn_varlen_func(
            q=q_flat,
            k=combined_k_flat,
            v=combined_v_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
            deterministic=deterministic,
        )

    out = out.squeeze(1).reshape(b, h, lq, d).transpose(1, 2)

    # Apply drop head mask
    raw_drop_mask = getattr(kv_cache, "drop_head_mask", None)
    if raw_drop_mask is not None:
        drop_head_mask = torch.as_tensor(raw_drop_mask, dtype=torch.bool, device=q.device)
        if drop_head_mask.numel() == h and torch.any(drop_head_mask):
            out[:, :, drop_head_mask, :] = 0

    return out.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
