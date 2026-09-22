#!/usr/bin/env bash
# Resume the last saved checkpoint after the reboot; finish the original budget.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python
training_console="logs/adaptive_height_resume42400_$(date +%Y%m%d_%H%M%S).log"
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Stairs-Long-G1-v0 \
  --viz none --device cuda:0 --num_envs 2048 \
  --max_iterations 9198 --seed 42 \
  --run_name long33_adaptive_height_resume42400 \
  --resume \
  --load_run /home/ljc/legged_lab_v3/logs/rsl_rl/g1_amp_stairs_long/2026-09-16_10-37-40_long33_adaptive_height_from41598 \
  --checkpoint model_42400.pt agent.device=cuda:0 2>&1 | tee "$training_console"
