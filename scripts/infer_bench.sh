#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Batch inference script for Rolling Forcing
#
# Usage:
#   bash scripts/infer_bench.sh --num_gpus 4 --num_output_frames 126
#
# Output:
#   OUTPUT_DIR/prompts.csv         — index,prompt mapping
#   OUTPUT_DIR/video_000.mp4 ...   — generated videos
# =============================================================================

# Defaults
CONFIG="configs/rolling_forcing_dmd.yaml"
CHECKPOINT="checkpoints/rolling_forcing_dmd.pt"
PROMPTS="prompts/MovieGenVideoBench_num32.txt"
OUTPUT_DIR="videos/MovieGenVideoBench_num32"
NUM_GPUS=1
NUM_OUTPUT_FRAMES=126
MASTER_PORT="${MASTER_PORT:-29501}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)            CONFIG="$2";            shift 2 ;;
        --checkpoint)        CHECKPOINT="$2";        shift 2 ;;
        --prompts)           PROMPTS="$2";           shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --num_gpus)          NUM_GPUS="$2";          shift 2 ;;
        --num_output_frames) NUM_OUTPUT_FRAMES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    die "--num_gpus must be a positive integer, got: $NUM_GPUS"
fi

if ! [[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || [[ "$MASTER_PORT" -gt 65535 ]]; then
    die "MASTER_PORT must be an integer in [1, 65535], got: $MASTER_PORT"
fi

# Validate prompt file exists
if [[ ! -f "$PROMPTS" ]]; then
    die "Prompt file not found: $PROMPTS"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Require that the runtime-visible GPU count matches the requested process count.
# GPU selection is intentionally delegated to the caller via CUDA_VISIBLE_DEVICES.
VISIBLE_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
if ! [[ "$VISIBLE_GPUS" =~ ^[0-9]+$ ]]; then
    die "Failed to detect visible GPU count (got: $VISIBLE_GPUS)"
fi

if [[ "$VISIBLE_GPUS" -ne "$NUM_GPUS" ]]; then
    die "Visible GPU count ($VISIBLE_GPUS) does not match --num_gpus ($NUM_GPUS). Set CUDA_VISIBLE_DEVICES externally so the visible devices match the requested process count."
fi

# ---------------------------------------------------------------------------
# 1. Generate prompts.csv
# ---------------------------------------------------------------------------
echo "index,prompt" > "$OUTPUT_DIR/prompts.csv"
idx=0
while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -n "$line" ]]; then
        escaped="${line//\"/\"\"}"
        echo "${idx},\"${escaped}\"" >> "$OUTPUT_DIR/prompts.csv"
        idx=$((idx + 1))
    fi
done < "$PROMPTS"
TOTAL=$idx
echo "Generated prompts.csv with $TOTAL prompts"

# Warn if num_gpus doesn't evenly divide prompt count (DistributedSampler drop_last=True)
if [[ $NUM_GPUS -gt 1 ]] && [[ $((TOTAL % NUM_GPUS)) -ne 0 ]]; then
    echo "WARNING: $TOTAL prompts is not divisible by $NUM_GPUS GPUs."
    echo "         DistributedSampler(drop_last=True) will skip the last $((TOTAL % NUM_GPUS)) prompt(s)."
fi

# ---------------------------------------------------------------------------
# 2. Launch inference in the background
# ---------------------------------------------------------------------------
if [[ $NUM_GPUS -gt 1 ]]; then
    CMD=(torchrun --master_port="$MASTER_PORT" --nproc_per_node="$NUM_GPUS" inference.py)
else
    CMD=(python inference.py)
fi

echo "Starting inference with $NUM_GPUS GPU(s) ..."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "MASTER_PORT=$MASTER_PORT"
printf 'Launch command:'
printf ' %q' "${CMD[@]}"
printf ' %q' \
    --config_path "$CONFIG" \
    --checkpoint_path "$CHECKPOINT" \
    --data_path "$PROMPTS" \
    --output_folder "$OUTPUT_DIR" \
    --num_output_frames "$NUM_OUTPUT_FRAMES" \
    --use_ema \
    --save_with_index
printf '\n'
"${CMD[@]}" \
    --config_path "$CONFIG" \
    --checkpoint_path "$CHECKPOINT" \
    --data_path "$PROMPTS" \
    --output_folder "$OUTPUT_DIR" \
    --num_output_frames "$NUM_OUTPUT_FRAMES" \
    --use_ema \
    --save_with_index &
INFER_PID=$!

# ---------------------------------------------------------------------------
# 3. Monitor and rename videos as they appear
# ---------------------------------------------------------------------------
renamed=0
echo "Monitoring for generated videos ..."

rename_ready_files() {
    for f in "$OUTPUT_DIR"/*-0_ema.mp4; do
        [[ -e "$f" ]] || return 0
        base=$(basename "$f")
        idx_num=${base%%-*}
        new_name=$(printf "video_%03d.mp4" "$idx_num")
        mv "$f" "$OUTPUT_DIR/$new_name"
        renamed=$((renamed + 1))
        echo "  Renamed: $base -> $new_name  ($renamed/$TOTAL)"
    done
}

while kill -0 "$INFER_PID" 2>/dev/null; do
    rename_ready_files
    sleep 2
done

# Collect exit code
wait "$INFER_PID" || true
INFER_EXIT=$?

# Final sweep
rename_ready_files

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Inference Complete ==="
echo "Exit code:      $INFER_EXIT"
echo "Videos renamed: $renamed / $TOTAL"
echo "Output dir:     $OUTPUT_DIR"

remaining=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*-0_ema.mp4" | wc -l)
if [[ "$remaining" -gt 0 ]]; then
    echo "WARNING: $remaining file(s) were not renamed"
fi

exit "$INFER_EXIT"
