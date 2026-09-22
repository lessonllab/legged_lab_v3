#!/usr/bin/env bash
# Continue the completed 41598 checkpoint with independent adaptive stair height.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python
training_console="logs/adaptive_height_10000_$(date +%Y%m%d_%H%M%S).log"
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Stairs-Long-G1-v0 \
  --viz none --device cuda:0 --num_envs 2048 \
  --max_iterations 10000 --seed 42 \
  --run_name long33_adaptive_height_from41598 \
  --resume \
  --load_run /home/ljc/legged_lab_v3/logs/rsl_rl/g1_amp_stairs_long/2026-09-15_23-47-35_long33_precontact_from31599 \
  --checkpoint model_41598.pt agent.device=cuda:0 2>&1 | tee "$training_console"
