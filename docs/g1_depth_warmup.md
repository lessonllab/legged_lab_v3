# G1 视觉策略：平地直行预训练

此配置用于修复此前“能站住、不能持续前进”的问题。**本阶段训练平地直行，不是崎岖地形。**
保留深度相机、8 帧历史和 CNN，后续可迁移到相同输入结构的崎岖地形任务。
平地阶段本身不证明策略已经会利用视觉识别障碍；教师蒸馏尚未加入。

训练任务：`LeggedLab-Isaac-AMP-Depth-Warmup-G1-v0`。
播放任务：`LeggedLab-Isaac-AMP-Depth-Warmup-G1-Play-v0`。

## 已落实的修改

- 固定指令为 vx=0.5 m/s、vy=0、wz=0，参考状态重置不再覆盖指令。
- PPO 仍采样动作，但标准差固定为 0.5（`learn_std=False`），熵系数为 0。
- 添加不前进惩罚，权重 -0.5；只在前进指令 >0.3 m/s 时生效，重置后前 0.5 秒不处罚。
  实际 vx<0.15 m/s 时开始处罚，倒退增加处罚。它不是摔倒终止项。
- 扭矩惩罚从 -2e-6 调为 -1.5e-7，脚离地奖励从 0.75 调为 0.25。
  保留前进/转向奖励 1/2、脚滑惩罚、关节限制和摔倒终止。
- 关闭质量/摩擦随机化、外力和推扰；训练保留本体观测噪声，播放关闭。
- AMP 保存 10 轮容量，历史足够时每次采样约一半来自最新一轮、一半来自更早轮次。
  缓冲未满时只采有效样本；每轮训练样本数量和判别器更新间隔保持原设置。
- 新增 `AMP/task_reward_per_step`、`AMP/style_reward_per_step`、`AMP/mixed_reward_per_step`、
  `AMP/style_zero_fraction` 和 `AMP/online_disc_score` 日志。

标准差、奖励和回放改动集中用于新任务。旧任务保持原奖励与一轮缓冲容量，通用算法新增奖励日志和缓冲有效性修复。
来源与试验依据见 [研究方案](analysis/g1_depth_20260910/research_modification_plan.md)。这些调整需要通过正式训练验证，不能保证某个轮数必然收敛。

## 开始新训练

```bash
source /home/ljc/isaaclab/bin/activate
cd /home/ljc/legged_lab_v3
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-Warmup-G1-v0 --viz none \
  --num_envs 512 --max_iterations 10000 --run_name straight_v2
```

日志写入 `logs/rsl_rl/g1_amp_depth_warmup/`，每 500 轮保存一次模型。
这条指令从头训练。不要添加旧 `depth_baseline` 的 resume 参数；它会带回原先已增大的 std 和优化器状态。
启动脚本使用项目独立的 RSL-RL 5.4.1，并处理 OpenBLAS 启动崩溃设置。

10000 轮是本阶段训练预算，不是通过验收的保证。建议第 1000 轮后开始检查确定性前进能力，
不要仅凭 `Mean reward` 上升或存活时间增加继续等待。

## 固定直行验收

替换为新任务实际生成的 checkpoint 绝对路径：

```bash
bash scripts/run_with_rsl5.sh scripts/tools/verify_amp_depth_progress.py \
  --task LeggedLab-Isaac-AMP-Depth-Warmup-G1-v0 --viz none \
  --checkpoint /absolute/path/to/model_1000.pt \
  --num_envs 64 --seed 42 --output /tmp/warmup_eval_seed42.json
```

再使用 seed 43 和不同输出文件复测，合计 128 个首回合。评估固定为 0.5 m/s、零转向、确定性动作，
终点在自动重置前记录；同时报告所有回合与存活回合，避免短命的参考初速度抬高平均速度。

建议验收：存活率与 4 米净位移达标率都超过 80%，平均 XY 速度误差 <0.15 m/s，
并观察是否存在滑步、单脚静止或动作幅度过大。平地验收通过后才增加崎岖地形和转向。
本阶段没有地形课程，因此看不到地形等级增长是正常现象。

播放：

```bash
bash scripts/run_with_rsl5.sh scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-Depth-Warmup-G1-Play-v0 --viz kit --num_envs 4 \
  --checkpoint /absolute/path/to/model_1000.pt
```

## 恢复同一个 warmup 实验

```bash
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-Warmup-G1-v0 --viz none --num_envs 512 \
  --resume --load_run YOUR_WARMUP_RUN_DIRECTORY --checkpoint model_1000.pt \
  --max_iterations 2000 --run_name straight_v2_resume
```

`load_run` 为 `g1_amp_depth_warmup` 下的运行目录名；`checkpoint` 在此传文件名。
RSL-RL 的 `max_iterations` 在恢复时表示本次追加训练轮数。
网络、优化器和 AMP 判别器恢复；历史回放不序列化到每个 checkpoint，恢复后从新采样逐步填充，
因此不保证逐位复现中断前的随机训练轨迹。此设计避免每个模型额外存储大批可重新采集的数据。

## 验证记录（2026-09-10）

- 6 项新测试覆盖回放跨轮/绕回、部分重置、冷启动、惩罚门控、旧配置隔离、真实 PPO 更新和恢复。
- 3 项深度策略测试、3 项平地速度指令测试通过。
- 用户环境实际启动 512 个机器人，默认 24 步采样与默认 PPO 更新设置，完成 3 轮并生成 checkpoint。
- 实际观测维度为 policy=495、critic=787、depth=8×36×64，与原视觉任务一致；三轮 mean_std 均为 0.5。
- 从该 checkpoint 在 4 个真实仿真环境恢复训练，通过；恢复后 mean_std 仍为 0.5，回放从有效新样本重新填充。
- 固定直行评估脚本已支持选择 warmup 平地任务，并用短测模型验证了重置前终点记录。
  该仅训练 3 轮的模型在 4 个测试回合中均提前失败；这不是正式训练或行走验收结果。

上述短测只验证实现与训练流程。正式直行能力需要后续训练和独立验收。
