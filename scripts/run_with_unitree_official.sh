#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-/home/ljc/isaaclab/bin/python}"
export PYTHONPATH="$project_dir/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$project_dir"
exec bash "$project_dir/scripts/run_with_rsl5.sh" "$@"
