#!/usr/bin/env bash
# V12 branches from the independently evaluated, more balanced v11 model.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python
if pgrep -f '^/home/ljc/isaaclab/bin/python scripts/rsl_rl/train.py .*LeggedLab-Isaac-AMP-(Stairs-Long|Foothold)-G1-v0' >/dev/null; then
  echo '已有正式训练运行，请先结束原训练，避免重复占用 GPU。' >&2
  exit 1
fi
resume_checkpoint="${1:-$("$ISAACLAB_PYTHON" scripts/latest_stairs_checkpoint.py --foothold)}"
if [[ ! -f "$resume_checkpoint" ]]; then
  echo "Checkpoint not found: $resume_checkpoint" >&2
  exit 1
fi
training_console="logs/stairs_foothold_v12_$(date +%Y%m%d_%H%M%S).log"
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Foothold-G1-v0 --viz none --device cuda:0 \
  --num_envs "${NUM_ENVS:-2048}" --max_iterations "${TRAIN_ITERATIONS:-10000}" --seed 42 \
  --run_name foothold_v12 --resume --load_run "$(dirname "$resume_checkpoint")" \
  --checkpoint "$(basename "$resume_checkpoint")" agent.device=cuda:0 2>&1 | tee "$training_console"
