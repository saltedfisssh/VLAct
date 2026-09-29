#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODE="${1:-full}"
shift || true
SOURCE="${SOURCE:-/mnt/xiaoyu_ssd_intern/RoboDojo/data/RoboDojo_lerobot_v30_video}"
BASE_VLM="${BASE_VLM:-/mnt/xiaoyu_ssd_intern/pretrained_checkpoints/Qwen3-VL-4B-Instruct-Action}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/mnt/xiaoyu_ssd_intern/pretrained_checkpoints/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt}"
RUN_ROOT="${RUN_ROOT:-$PWD/results/RoboDojo}"
if [[ "$MODE" == smoke ]]; then
  NPROC="${NPROC:-1}"
  CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/data_smoke}"
  RUN_ID="${RUN_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
  BATCH_SIZE="${BATCH_SIZE:-1}"; GRAD_ACCUM="${GRAD_ACCUM:-2}"; MAX_STEPS="${MAX_STEPS:-2}"
  prepare_args=(--max-tasks 2 --episodes-per-task 2 --validation-fraction .5)
  overrides=(trainer.optimizer_offload=true trainer.num_warmup_steps=0 trainer.save_interval="$MAX_STEPS" trainer.eval_interval="$MAX_STEPS" trainer.eval_batches=1 trainer.logging_frequency=1 datasets.vla_data.num_workers=0 trainer.save_training_state=false)
else
  NPROC="${NPROC:-8}"
  CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/data_full}"
  RUN_ID="${RUN_ID:-vlact_robodojo_oft_$(date -u +%Y%m%dT%H%M%SZ)}"
  BATCH_SIZE="${BATCH_SIZE:-8}"; GRAD_ACCUM="${GRAD_ACCUM:-2}"; MAX_STEPS="${MAX_STEPS:-30000}"
  prepare_args=(--validation-fraction .1)
  overrides=(trainer.save_training_state=true)
fi
prepare=(uv run --no-sync python -m examples.RoboDojo.data --source "$SOURCE" --output "$CACHE_DIR" "${prepare_args[@]}")
train=(uv run --no-sync python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" --module examples.RoboDojo.train
  --config examples/RoboDojo/train.yaml "framework.qwenvl.base_vlm=$BASE_VLM" "trainer.pretrained_checkpoint=$PRETRAINED_CKPT"
  "datasets.vla_data.cache_dir=$CACHE_DIR" "datasets.vla_data.source_override=$SOURCE" "run_root_dir=$RUN_ROOT" "run_id=$RUN_ID"
  "datasets.vla_data.per_device_batch_size=$BATCH_SIZE" "trainer.gradient_accumulation_steps=$GRAD_ACCUM" "trainer.max_train_steps=$MAX_STEPS"
  "${overrides[@]}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  if [[ "${PREPARE:-1}" == 1 ]]; then printf '%q ' "${prepare[@]}"; printf '\n'; fi
  printf '%q ' "${train[@]}"; printf '\n'
  exit 0
fi
[[ -x .venv/bin/python && -f "$BASE_VLM/config.json" && -f "$PRETRAINED_CKPT" ]] || { echo 'Existing .venv / local model / pretrained checkpoint missing' >&2; exit 1; }
uv run --no-sync python -c 'import sys, torch; n=int(sys.argv[1]); assert torch.cuda.device_count() >= n, f"Requested {n} GPUs, found {torch.cuda.device_count()}"' "$NPROC"
# Set PREPARE=0 when using an already prepared (e.g. reviewed subtask) cache.
if [[ "${PREPARE:-1}" == 1 ]]; then "${prepare[@]}"; fi
mkdir -p "$RUN_ROOT/logs"
"${train[@]}" 2>&1 | tee -a "$RUN_ROOT/logs/$RUN_ID.log"
