#!/usr/bin/env bash
# Continue adaptive stair height plus flat reverse/turn practice and rough-terrain progression. Stop the old trainer first.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python
if pgrep -f '^/home/ljc/isaaclab/bin/python scripts/rsl_rl/train.py .*LeggedLab-Isaac-AMP-Stairs-Long-G1-v0' >/dev/null; then
  echo '已有长楼梯训练运行，请先在原训练终端按 Ctrl+C，避免同时启动两份正式训练。' >&2
  exit 1
fi
resume_checkpoint="${1:-$("$ISAACLAB_PYTHON" scripts/latest_stairs_checkpoint.py)}"
resume_iteration="$(basename "$resume_checkpoint" .pt)"
resume_iteration="${resume_iteration#model_}"
remaining_iterations="${TRAIN_ITERATIONS:-$((51598-resume_iteration))}"
if (( remaining_iterations <= 0 )); then
  echo '原定结束回合已达到；如需延长，请明确设置 TRAIN_ITERATIONS。' >&2
  exit 1
fi
training_console="logs/stairs_reverse_height_$(date +%Y%m%d_%H%M%S).log"
echo "Resume: $resume_checkpoint; additional iterations: $remaining_iterations"
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
 --task LeggedLab-Isaac-AMP-Stairs-Long-G1-v0 --viz none --device cuda:0 \
 --num_envs 2048 --max_iterations "$remaining_iterations" --seed 42 \
 --run_name reverse_height_v11 --resume --load_run "$(dirname "$resume_checkpoint")" \
 --checkpoint "$(basename "$resume_checkpoint")" agent.device=cuda:0 2>&1 | tee "$training_console"
