#!/usr/bin/env bash
# Frozen 8999 BeamDojo policy; native MuJoCo, no Isaac Sim startup.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${MUJOCO_PYTHON:-/home/ljc/isaaclab/bin/python}" "$project_dir/scripts/sim2sim/beamdojo_mujoco.py" "$@"
