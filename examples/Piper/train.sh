#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mode=${1:-smoke}
data_mix=${DATA_MIX:-${2:-piper_fold}}
rung=${3:-s2_pair}
case "$mode" in smoke|train|prepare) ;; *) echo 'mode: smoke|train|prepare' >&2; exit 2;; esac
case "$rung" in s0|s2|s2_pair) ;; *) echo 'rung: s0|s2|s2_pair' >&2; exit 2;; esac
# Backward-compatible aliases; selection and weights always come from the registry.
case "$data_mix" in piper) data_mix=piper_fold;; piper_x) data_mix=piper_x_native;; esac
python=${PYTHON:-$PWD/.venv/bin/python}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
source_root=${PIPER_SOURCE_ROOT:-/mnt/xiaoyu_teleop_data/piper/latest}
pretrained=${PRETRAINED_CKPT:-$PWD/playground/Pretrained_models/VLAct-Qwen3VL4B-Pretrained/checkpoints/steps_100000_pytorch_model.pt}
prep_extra=(); train_extra=()
if [[ "$mode" == smoke ]]; then
  cache_root=${PIPER_CACHE_ROOT:-$PWD/playground/Piper/registered_smoke}
  prep_extra+=(--smoke); train_extra+=(--smoke)
  if [[ "${SAVE_OPTIMIZER:-0}" == 0 ]]; then train_extra+=(--weights-only); fi
  stats=${STATS_SAMPLES:-64}; gpus=1; steps=${STEPS:-2}; batch=${BATCH_SIZE:-1}; accumulation=${ACCUMULATION:-1}
else
  cache_root=${PIPER_CACHE_ROOT:-$PWD/playground/Piper/registered}
  stats=${STATS_SAMPLES:-8192}; gpus=${GPUS:-8}; steps=${STEPS:-60000}; batch=${BATCH_SIZE:-2}; accumulation=${ACCUMULATION:-8}
fi
# Use the requested release in smoke as well once its download is available.
if [[ "$mode" == train || -f "$pretrained" || -n "${PRETRAINED_CKPT:-}" ]]; then
  train_extra+=(--pretrained "$pretrained")
fi
if [[ "${BALANCE_DATASET_WEIGHTS:-false}" == true ]]; then prep_extra+=(--balance-dataset-weights); fi
if [[ -n "${BASE_VLM:-}" ]]; then train_extra+=(--vlm "$BASE_VLM"); fi
if [[ -n "${ACTION_EXPERT:-}" ]]; then train_extra+=(--expert "$ACTION_EXPERT"); fi
if [[ -n "${RESUME:-}" ]]; then train_extra+=(--resume "$RESUME"); fi
run_name=${RUN_NAME:-${data_mix}_${rung}_${mode}_$(date -u +%Y%m%d_%H%M%S)}
output=${OUTPUT_ROOT:-$PWD/results/Piper}/$run_name
prep=("$python" -m starVLA.dataloader.piper_datasets --data-root-dir "$source_root"
  --cache-root "$cache_root" --data-mix "$data_mix" --stats-samples "$stats"
  --workers "${PREPARE_WORKERS:-4}" --alignment "${PIPER_ALIGNMENT:-nominal}" "${prep_extra[@]}")
train=("$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$gpus"
  -m starVLA.piper.train --cache "$cache_root/mixtures/$data_mix" --data-mix "$data_mix" --rung "$rung"
  --output "$output" --steps "$steps" --batch-size "$batch" --accumulation "$accumulation"
  --workers "${NUM_WORKERS:-2}" --save-every "${SAVE_EVERY:-10000}"
  --log-every "${LOG_EVERY:-1}" "${train_extra[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${prep[@]}"; printf '\n'
  if [[ "$mode" != prepare ]]; then printf '%q ' "${train[@]}"; printf '\n'; fi
  exit 0
fi
if [[ "$mode" == train && ! -f "$pretrained" ]]; then
  echo "VLAct_Qwen3_Pretrain download is not complete: $pretrained" >&2; exit 2
fi
"${prep[@]}"
if [[ "$mode" == prepare ]]; then exit 0; fi
mkdir -p "${LOG_ROOT:-$PWD/results/Piper/logs}"
"${train[@]}" 2>&1 | tee "${LOG_ROOT:-$PWD/results/Piper/logs}/$run_name.log"
