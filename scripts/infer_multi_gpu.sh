#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Multi-GPU parallel inference with fair prompt distribution (no drop_last)
#
# Launches N independent python processes (one per GPU) instead of torchrun,
# so every prompt is processed even when the count isn't divisible by GPU count.
#
# Usage:
#   bash scripts/infer_multi_gpu.sh --gpus 0,1,2,3 --prompts prompts/128.txt
#   bash scripts/infer_multi_gpu.sh --gpus 0,1,2   --prompts prompts/128.txt
#
# Output:
#   OUTPUT_DIR/prompts.csv         — index,prompt mapping (0-based)
#   OUTPUT_DIR/video_000.mp4 ...   — generated videos (3-digit 0-padded)
# =============================================================================

# Defaults
CONFIG="configs/rolling_forcing_dmd.yaml"
CHECKPOINT="checkpoints/rolling_forcing_dmd.pt"
PROMPTS="prompts/MovieGenVideoBench_num32.txt"
OUTPUT_DIR="videos/output"
GPUS="0,1,2,3"
NUM_OUTPUT_FRAMES=120
NUM_FRAMES_ARG=""
NUM_OUTPUT_FRAMES_ARG=""
USE_HEADKV=false
HEADKV_CONFIG=""

die() {
    echo "ERROR: $*" >&2
    exit 1
}

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    die "Neither python nor python3 was found in PATH"
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)            CONFIG="$2";              shift 2 ;;
        --checkpoint)        CHECKPOINT="$2";           shift 2 ;;
        --prompts)           PROMPTS="$2";              shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";           shift 2 ;;
        --gpus)              GPUS="$2";                 shift 2 ;;
        --num-frames)        NUM_FRAMES_ARG="$2";       shift 2 ;;
        --num_output_frames) NUM_OUTPUT_FRAMES_ARG="$2"; shift 2 ;;
        --use_headkv)        USE_HEADKV=true;           shift 1 ;;
        --headkv_config)     HEADKV_CONFIG="$2";        shift 2 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# Resolve num_output_frames from --num-frames / --num_output_frames
if [[ -n "$NUM_FRAMES_ARG" ]] && [[ -n "$NUM_OUTPUT_FRAMES_ARG" ]] && [[ "$NUM_FRAMES_ARG" != "$NUM_OUTPUT_FRAMES_ARG" ]]; then
    die "--num-frames ($NUM_FRAMES_ARG) conflicts with --num_output_frames ($NUM_OUTPUT_FRAMES_ARG)"
fi
if [[ -n "$NUM_FRAMES_ARG" ]]; then
    NUM_OUTPUT_FRAMES="$NUM_FRAMES_ARG"
elif [[ -n "$NUM_OUTPUT_FRAMES_ARG" ]]; then
    NUM_OUTPUT_FRAMES="$NUM_OUTPUT_FRAMES_ARG"
fi

if ! [[ "$NUM_OUTPUT_FRAMES" =~ ^[1-9][0-9]*$ ]]; then
    die "--num-frames/--num_output_frames must be a positive integer, got: $NUM_OUTPUT_FRAMES"
fi

# Validate prompt file
if [[ ! -f "$PROMPTS" ]]; then
    die "Prompt file not found: $PROMPTS"
fi

# Validate num_frame_per_block divisibility
NUM_FRAME_PER_BLOCK=$(awk -F: '/^num_frame_per_block:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$CONFIG")
NUM_FRAME_PER_BLOCK=${NUM_FRAME_PER_BLOCK:-1}
if ! [[ "$NUM_FRAME_PER_BLOCK" =~ ^[1-9][0-9]*$ ]]; then
    die "Failed to resolve num_frame_per_block from config: $CONFIG"
fi
if [[ $((NUM_OUTPUT_FRAMES % NUM_FRAME_PER_BLOCK)) -ne 0 ]]; then
    die "--num-frames must be divisible by num_frame_per_block ($NUM_FRAME_PER_BLOCK), got: $NUM_OUTPUT_FRAMES"
fi

# ---------------------------------------------------------------------------
# Parse GPU list
# ---------------------------------------------------------------------------
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NUM_GPUS=${#GPU_IDS[@]}

if [[ $NUM_GPUS -eq 0 ]]; then
    die "--gpus must specify at least one GPU ID"
fi

echo "GPUs: ${GPU_IDS[*]} (${NUM_GPUS} total)"

# ---------------------------------------------------------------------------
# Read prompts into array
# ---------------------------------------------------------------------------
PROMPT_LINES=()
while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -n "$line" ]]; then
        PROMPT_LINES+=("$line")
    fi
done < "$PROMPTS"
TOTAL=${#PROMPT_LINES[@]}

if [[ $TOTAL -eq 0 ]]; then
    die "No prompts found in: $PROMPTS"
fi

echo "Total prompts: $TOTAL"

# ---------------------------------------------------------------------------
# 1. Generate prompts.csv
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

echo "index,prompt" > "$OUTPUT_DIR/prompts.csv"
for ((i = 0; i < TOTAL; i++)); do
    escaped="${PROMPT_LINES[$i]//\"/\"\"}"
    echo "${i},\"${escaped}\"" >> "$OUTPUT_DIR/prompts.csv"
done
echo "Generated prompts.csv with $TOTAL prompts"

# ---------------------------------------------------------------------------
# 2. Fair prompt distribution
# ---------------------------------------------------------------------------
BASE=$((TOTAL / NUM_GPUS))
REMAINDER=$((TOTAL % NUM_GPUS))

# Compute per-GPU count and offset
COUNTS=()
OFFSETS=()
offset=0
for ((g = 0; g < NUM_GPUS; g++)); do
    if [[ $g -lt $REMAINDER ]]; then
        count=$((BASE + 1))
    else
        count=$BASE
    fi
    COUNTS+=("$count")
    OFFSETS+=("$offset")
    offset=$((offset + count))
done

echo "Distribution: ${COUNTS[*]} (offsets: ${OFFSETS[*]})"

# ---------------------------------------------------------------------------
# 3. Create temp prompt files and sub-directories per GPU
# ---------------------------------------------------------------------------
TMPDIR_BASE=$(mktemp -d "${OUTPUT_DIR}/.tmp_multi_gpu_XXXXXX")
trap 'rm -rf "$TMPDIR_BASE"' EXIT

for ((g = 0; g < NUM_GPUS; g++)); do
    gpu_offset=${OFFSETS[$g]}
    gpu_count=${COUNTS[$g]}

    # Write per-GPU prompt file
    prompt_file="${TMPDIR_BASE}/prompts_gpu${g}.txt"
    for ((j = 0; j < gpu_count; j++)); do
        echo "${PROMPT_LINES[$((gpu_offset + j))]}" >> "$prompt_file"
    done

    # Create per-GPU output sub-directory
    mkdir -p "${TMPDIR_BASE}/out_gpu${g}"
done

# ---------------------------------------------------------------------------
# 4. Build common inference arguments
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --config_path "$CONFIG"
    --checkpoint_path "$CHECKPOINT"
    --num_output_frames "$NUM_OUTPUT_FRAMES"
    --use_ema
    --save_with_index
)

if [[ "$USE_HEADKV" == true ]]; then
    COMMON_ARGS+=(--use_headkv)
fi
if [[ -n "$HEADKV_CONFIG" ]]; then
    COMMON_ARGS+=(--headkv_config "$HEADKV_CONFIG")
fi

# ---------------------------------------------------------------------------
# 5. Launch one process per GPU
# ---------------------------------------------------------------------------
PIDS=()
echo ""
echo "Starting inference with $NUM_GPUS GPU(s) ..."

for ((g = 0; g < NUM_GPUS; g++)); do
    gpu_id=${GPU_IDS[$g]}
    prompt_file="${TMPDIR_BASE}/prompts_gpu${g}.txt"
    out_dir="${TMPDIR_BASE}/out_gpu${g}"

    echo "  GPU $gpu_id: ${COUNTS[$g]} prompts (offset ${OFFSETS[$g]})"

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" inference.py \
        "${COMMON_ARGS[@]}" \
        --data_path "$prompt_file" \
        --output_folder "$out_dir" \
        &
    PIDS+=($!)
done

# ---------------------------------------------------------------------------
# 6. Wait for all processes
# ---------------------------------------------------------------------------
echo ""
echo "Waiting for ${NUM_GPUS} inference processes ..."

ALL_OK=true
for ((g = 0; g < NUM_GPUS; g++)); do
    pid=${PIDS[$g]}
    gpu_id=${GPU_IDS[$g]}
    if wait "$pid"; then
        echo "  GPU $gpu_id (PID $pid): done"
    else
        exit_code=$?
        echo "  GPU $gpu_id (PID $pid): FAILED (exit $exit_code)"
        ALL_OK=false
    fi
done

if [[ "$ALL_OK" != true ]]; then
    die "One or more GPU processes failed"
fi

# ---------------------------------------------------------------------------
# 7. Rename and consolidate videos
# ---------------------------------------------------------------------------
echo ""
echo "Consolidating videos ..."

renamed=0
for ((g = 0; g < NUM_GPUS; g++)); do
    gpu_offset=${OFFSETS[$g]}
    gpu_count=${COUNTS[$g]}
    out_dir="${TMPDIR_BASE}/out_gpu${g}"

    for ((j = 0; j < gpu_count; j++)); do
        global_idx=$((gpu_offset + j))
        src="${out_dir}/${j}-0_ema.mp4"
        dst=$(printf "%s/video_%03d.mp4" "$OUTPUT_DIR" "$global_idx")

        if [[ -f "$src" ]]; then
            mv "$src" "$dst"
            renamed=$((renamed + 1))
        else
            echo "  WARNING: missing ${src}"
        fi
    done
done

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Inference Complete ==="
echo "Videos renamed: $renamed / $TOTAL"
echo "Output dir:     $OUTPUT_DIR"
echo "Prompts CSV:    $OUTPUT_DIR/prompts.csv"

if [[ $renamed -ne $TOTAL ]]; then
    echo "WARNING: expected $TOTAL videos, got $renamed"
    exit 1
fi
