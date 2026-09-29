#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export SOURCE="${SOURCE:-$PWD/../lerobot/outputs/robodojo/review_35_qwen3vl30b}"
export RUN_ROOT="${RUN_ROOT:-$PWD/results/RoboDojo}"
export CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/data_subtask_smoke}"
export RUN_ID="${RUN_ID:-subtask_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
export MAX_STEPS="${MAX_STEPS:-4}"
prepare=(uv run --no-sync python -m examples.RoboDojo.data
  --source "$SOURCE" --output "$CACHE_DIR" --language-mode subtask
  --allow-unreviewed-smoke --split-unit task --episodes-per-task 1 --validation-fraction .1)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${prepare[@]}"; printf '\n'
else
  "${prepare[@]}"
fi
export PREPARE=0
exec bash examples/RoboDojo/launch.sh smoke \
  datasets.vla_data.expected_language_mode=subtask trainer.audit_batches=10 "$@"
