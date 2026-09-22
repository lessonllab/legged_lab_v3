# Target v5 修复与验证

2026-09-12。训练任务 `LeggedLab-Isaac-AMP-Depth-Target-G1-v1`，播放任务
`LeggedLab-Isaac-AMP-Depth-Target-G1-Play-v1`；实验目录 `g1_amp_depth_target_v5`。
这是现有 PPOAMP 工程的 Instinct 配套适配，不是完整复现 Instinct 的 WasabiPPO/MoE。

## 训练行为

- 站立关节惩罚同时要求平移命令 <0.15 m/s、角速度命令绝对值 <0.15 rad/s。
  原地左右转向不会误触发。公共奖励函数的原有平移默认阈值 0.06 保留，
  v5 显式设为 0.15；公共函数的转向判定修复也适用于旧任务。
- 线/角速度跟踪权重为 2/2，增加目标方向代价 `heading_error`，权重 -1。
  实现与本机 Instinct 相同，使用目标控制器的转向指令绝对值作为方向误差代理。
  不用于固定速度命令任务。保留现有站立分关节权重、足底支撑/边缘约束及 AMP 权重。
- 升级使用真实移动目标期间的跟踪均值：XY>0.6、yaw>0.5；有效机会至少 2 秒，
  单个目标最大净推进至少 1 米或有效到达，并且正常超时、没有失败终止。
  目标初始距离须大于停止半径 0.4 米加 0.2 米；有效到达还须至少接近 0.2 米。
  出生在目标附近、站立和到达后的等待不贡献升级。命令重采样重新建立距离基准，
  回合重置清除统计、跳过重置瞬间，不能把传送或反复往返的路程当成净推进。
  有移动机会的失败回合降级；机会足够且 XY 或 yaw 跟踪 <0.3 也降级，纯站立回合保持。
  这是针对本机复现漏洞的修正；本机 Instinct 任务本身的 yaw 升级门槛也是 0。
- 训练从默认站姿附近开始：XY/yaw ±0.1，根速度各分量 ±0.2，关节位置 ±0.15 rad、
  关节速度为零。PLAY 使用默认关节位置。参考动作仍用于完整的 10 帧 AMP 示范和镜像采样。
- 动作标准差从 0.5 起步并允许学习，使用 RSL-RL 的对数参数形式，范围 [0.05,1]；
  熵系数为 Instinct 的 0.006。初值和范围是本工程的适配，Instinct 初值为 1.0。
  未改判别器优化器、归一化器或奖励混合公式；应继续观察风格奖励和实际步态。
- 深度参与 actor 和 critic，输入保持 8×18×32；无理想地形高度输入。
  六类崎岖地形保留，从等级 0 开始；目标范围和相机参数保持原配置。

## 验证范围

- `test_amp*.py` 共 37 项测试通过，覆盖转向奖励、目标/课程反例、部分重置、视觉、
  AMP 参考状态和实际 PPO 更新。新测试确认标准差参数参与梯度更新，且可保存/恢复。
  日志：[unit.log](unit.log)。
- CUDA PhysX 六环境实测通过：视觉与 AMP 观察尺寸和数值正常；三次站姿/部分重置的
  最大关节扰动为 0.14955 rad；其他环境的目标和统计不受部分重置影响。
  参考序列仍随时间变化，50 步物理运行无非有限值。详见 [physical.json](physical.json)。
- 使用最终指令相同的 1024 个环境完成 3 轮 PPOAMP 短训练并正常退出，模型成功保存。
  最终 checkpoint 的所有张量均有限，深度编码器权重确实更新；动作标准差更新到约
  0.49979–0.50145，确认不再固定为 0.5。详见 [train_smoke.json](train_smoke.json)
  和 [train_smoke.log](train_smoke.log)。短测独立保存在 `g1_amp_depth_target_v5_smoke`，
  正式训练尚未启动。
- 真实仿真发现首次 reset 时环境的 `reset_terminated` 别名尚未建立，已修正为读取
  终止管理器缓冲区，重新实测通过。PLAY 同时移除了重复站姿重置事件。

这些检查验证了实现、数值和训练通路，不能证明新策略已经学会走路或保证收敛。

## 启动

```bash
cd /home/ljc/legged_lab_v3
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh \
  scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-Depth-Target-G1-v1 \
  --viz none --device cuda:0 --num_envs 1024 \
  --max_iterations 30000 --seed 42 --run_name target_v5_scratch \
  agent.device=cuda:0 agent.resume=false
```

不要用 v4 的 checkpoint 直接恢复此任务：探索参数从标量 std 改为 log std，
任务奖励和初始状态也不同。新训练保存到独立目录；旧日志和模型保留。

应固定种子、地形难度分别检查前进、左右转向和目标追踪；有效到达、跌倒及风格一起判断。
地形等级不再作为单独的效果结论。训练预算 30000 轮，每 500 轮保存；达到预算不等于收敛。
