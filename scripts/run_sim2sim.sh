#!/usr/bin/env bash
# Isolated MuJoCo entry: does not start Isaac Sim, ROS, DDS, or the old controller.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${ISAACLAB_PYTHON:-/home/ljc/isaaclab/bin/python}"
checkpoint="$project_dir/logs/rsl_rl/g1_amp_stairs_long/2026-09-16_20-26-52_reverse_height_v11/model_50100.pt"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
exec "$python_bin" "$project_dir/scripts/sim2sim/run_mujoco.py" --checkpoint "$checkpoint" "$@"
