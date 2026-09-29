#!/usr/bin/env bash
set -euo pipefail
export NPROC=8
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch.sh" full "$@"
