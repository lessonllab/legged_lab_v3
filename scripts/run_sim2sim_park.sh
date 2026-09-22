#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${ISAACLAB_PYTHON:-/home/ljc/isaaclab/bin/python}"
checkpoint="$("$python_bin" "$project_dir/scripts/latest_stairs_checkpoint.py")"
exec "$project_dir/scripts/run_sim2sim.sh" --checkpoint "$checkpoint" --scene "$project_dir/scripts/sim2sim/scenes/terrain_park.xml" --spawn 0 -10 .8 --camera-distance 16 --real-time "$@"
