#!/usr/bin/env bash
# Multi-height evaluation only; never starts or changes training.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python
matrix_checkpoint="${1:-}"
if [[ -z "$matrix_checkpoint" ]]; then
  matrix_checkpoint="$("$ISAACLAB_PYTHON" scripts/latest_stairs_checkpoint.py)"
fi
exec bash scripts/run_with_rsl5.sh scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-Stairs-Long-G1-Play-v0 \
  --checkpoint "$matrix_checkpoint" --stair_matrix \
  --stair_test_speed "${STAIR_TEST_SPEED:-0.8}" \
  --viz viser --viser_port 8082 --device cuda:0 --seed 42 \
  --follow_env 22 --real-time
