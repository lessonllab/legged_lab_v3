# G1 视觉崎岖任务：风格 v3 验证

日期：2026-09-11。正式训练尚未启动，用户将稍后从头开始。

## 修改依据和范围

v2 判别器每帧只有角速度与关节状态，共 61 维、4 帧；最近训练中风格奖励接近零，
同时用户观察到左右步态不自然。这不足以证明唯一原因，但判别信息与本机 InstinctLab
的 10 帧身体状态观测存在明显差异。

新增任务 `LeggedLab-Isaac-AMP-Depth-Style-G1-v0`，实验目录 `g1_amp_depth_style_v3`：

- 10 帧，每帧 67 维：身体坐标重力方向、相对默认姿态的关节角、0.05 倍关节速度、
  身体坐标中的根节点线速度和角速度；总输入 670 维，时间跨度 0.18 秒。
- 机器人与参考使用同一编码，线速度都定义在根 link 原点。
  参考速度由根节点位置差分得到，不应直接与机器人质心速度混用。
- 每回合以 50% 概率镜像整段参考，按关节名字交换左右，roll/yaw 轴改变符号；
  极向量和轴向量分别使用正确镜像符号，保持帧顺序。
- v2 actor/critic/相机/地形/任务奖励及 PPO/AMP 优化器参数保持原值。
  镜像参考不等同于策略对称损失，不直接镜像深度图，也不强迫崎岖地形上的左右落脚相同。
- 保留 v2 任务和旧 checkpoint；v3 从头训练，不完整恢复旧判别器或短测模型。

这是有明确验证边界的风格输入修正，不是 InstinctLab 的完整 WasabiPPO/MoE 复现，
也不能保证已经消除跛行。其长期效果要通过新训练与旧模型在相同命令、地形条件下比较。

## 验证结果

1. `unit.log`：21 项 AMP 相关回归测试通过，包含新增编码、真实反射等价、
   镜像两次还原、历史顺序、部分重置以及 v2 配置保持一致测试。
2. `physical_state.json`：CUDA PhysX 4 环境，实际写入 10 帧参考姿态和速度，
   然后读取机器人状态。每帧误差均小于 6.4e-6，实际观测与参考均为 `(4,10,67)`。
   G1 实际关节顺序与 retarget 配置一致；镜像和部分环境重置检查通过。
3. 写入参考速度时显式使用 `v_com = v_link + omega × offset_world`。
   本机 link 速度写接口经 forward/read 后未保持请求值，诊断脚本因此使用 COM 接口加显式转换；
   没有修改全局 Isaac Lab。该检查用于验证读出的 AMP 状态，不代表动力学行走测试。
4. `train_final_smoke.log`：最终代码在 GPU 上以 512 环境从头完成 3 轮 PPO+AMP 更新，
   判别器输入 670 维；`model_2.pt` 保存成功，进程正常退出。
   保存配置 `resume: false`，检查点所有张量有限，见 `train_result.json`。
5. 先前 128 环境 3 轮短测日志为 `train_smoke.log`，发生在根 link 速度修正之前；
   最终验收以 512 环境短测为准。短测只验证训练链路，不证明收敛、视觉依赖或步态改善。

## 从头训练

```bash
cd /home/ljc/legged_lab_v3
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh \
  scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-Style-G1-v0 \
  --viz none --device cuda:0 --num_envs 512 \
  --max_iterations 80000 --seed 42 --run_name style_v3_scratch \
  agent.resume=false
```

这是带前向深度相机与地形课程的崎岖训练，初始低难度地形不等于纯平地任务。
训练后使用对应的 `LeggedLab-Isaac-AMP-Depth-Style-G1-Play-v0` 和新 checkpoint 重放。
