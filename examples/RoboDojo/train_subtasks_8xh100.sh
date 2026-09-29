#!/usr/bin/env bash
# Uses a prepared subtask cache; its manifest records annotation review status.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export RUN_ROOT="${RUN_ROOT:-$PWD/results/RoboDojo}"
export CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/data_reviewed_subtasks}"
if [[ -z "${SOURCE:-}" ]]; then
  SOURCE="$(uv run --no-sync python -c 'import json,sys; print(json.load(open(sys.argv[1]))["spec"]["source"])' "$CACHE_DIR/manifest.json")"
fi
export SOURCE
export RUN_ID="${RUN_ID:-vlact_robodojo_subtask_$(date -u +%Y%m%dT%H%M%SZ)}"
export PREPARE=0
exec bash examples/RoboDojo/train_8xh100.sh \
  datasets.vla_data.expected_language_mode=subtask trainer.audit_batches=2 "$@"
