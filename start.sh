#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RECIPE_DIR/model-profiles.sh"
source "$RECIPE_DIR/platform-config.sh"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
if [[ "$PLATFORM_KIND" == spark ]]; then
  for argument in "$@"; do
    case "$argument" in
      --gpu-memory-utilization|--gpu-memory-utilization=*)
        echo "Use GPU_MEMORY_UTILIZATION (capped at 0.7 on Spark)" >&2; exit 2 ;;
    esac
  done
  awk -v value="$GPU_MEMORY_UTILIZATION" 'BEGIN {exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value <= 0.7)}' || {
    echo "Spark GPU_MEMORY_UTILIZATION must be greater than zero and at most 0.7" >&2; exit 2;
  }
fi
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
RUNTIME_CACHE="${RUNTIME_CACHE:-$HOME/.cache/qwen38-rtx/${QUANT:-exl3}}"
mkdir -p "$RUNTIME_CACHE"
MODEL_PATH="/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION"
if [[ ! -f "$HF_CACHE/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/config.json" ]]; then
  echo "Missing checkpoint. Run QUANT=${QUANT:-exl3} $RECIPE_DIR/download.sh" >&2
  exit 1
fi
MTP_TOKENS="${MTP_TOKENS:-$( [[ ${QUANT:-exl3} == nvfp4 ]] && echo 2 || echo "$DEFAULT_EXL3_MTP_TOKENS" )}"
PLE_MMAP="${PLE_MMAP:-${VLLM_PLE_MMAP:-1}}"
[[ "$PLE_MMAP" == 0 || "$PLE_MMAP" == 1 ]] || { echo "PLE_MMAP must be 0 or 1" >&2; exit 2; }
EXTRA_ARGS=()
[[ "${ENFORCE_EAGER:-0}" == 1 ]] && EXTRA_ARGS+=(--enforce-eager)
if [[ "${MTP_TOKENS:-3}" != 0 ]]; then
  EXTRA_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS:-3}}")
fi
docker run -d --name "${CONTAINER_NAME:-qwen38-${QUANT:-exl3}}" \
  --gpus "device=${GPU:-0}" --ipc=host --network=host \
  -e OMP_NUM_THREADS="${CPU_THREADS:-8}" \
  -e CUDA_CACHE_PATH=/root/.cache/cuda \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e QWEN38_B12X_VOCAB="${B12X_VOCAB:-$DEFAULT_B12X_VOCAB}" \
  -e QWEN38_B12X_NVFP4="${B12X_NVFP4:-0}" \
  -e VLLM_PLE_MMAP="$PLE_MMAP" \
  -e VLLM_PLE_MMAP_WORKERS="${PLE_MMAP_WORKERS:-32}" \
  -e VLLM_PLE_MMAP_CHUNK="${PLE_MMAP_CHUNK:-2048}" \
  -e VLLM_PLE_MMAP_PREWARM="${PLE_MMAP_PREWARM:-0}" \
  -e VLLM_PLE_MMAP_READAHEAD="${PLE_MMAP_READAHEAD:-2048}" \
  -e VLLM_PLE_MMAP_PINNED="${PLE_MMAP_PINNED:-0}" \
  -e VLLM_PLE_MMAP_SERIAL="${PLE_MMAP_SERIAL:-128}" \
  -e VLLM_PLE_MMAP_PAGE_PREFETCH="${VLLM_PLE_MMAP_PAGE_PREFETCH:-0}" \
  -e VLLM_PLE_MMAP_RANDOM="${VLLM_PLE_MMAP_RANDOM:-0}" \
  -e VLLM_PLE_MMAP_DECODE_STEPS="${VLLM_PLE_MMAP_DECODE_STEPS:-0}" \
  -e VLLM_PLE_CPU_OFFLOAD="$((1 - PLE_MMAP))" -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800 \
  -v "$RUNTIME_CACHE:/root/.cache" \
  -v "$HF_CACHE:/root/.cache/huggingface:ro" \
  "${IMAGE:-$DEFAULT_IMAGE}" "$MODEL_PATH" \
  --served-model-name "qwen38-${QUANT:-exl3}" --port "${PORT:-8001}" \
  --tensor-parallel-size 1 --distributed-executor-backend mp \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  "${EXTRA_ARGS[@]}" "$@"
