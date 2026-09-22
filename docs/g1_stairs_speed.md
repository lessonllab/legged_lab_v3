# 从 26500 续训：平地 3 m/s、楼梯 1.5 m/s

更新：2026-09-15。新增任务 `LeggedLab-Isaac-AMP-Stairs-G1-v1`；原 `v0` 仍用于历史播放与比较。
这些数值是课程最终的指令速度上限，尚未证明策略能够达到；上下楼分别推进。阶高仍为 8～23 cm。

## 其他项目的做法与本次取舍

- [Extreme Parkour 官方代码](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot.py)：`_reward_feet_edge` 用足部位置查询地形边缘掩码，与过滤后的接触状态相交；代码在地形等级大于 3 时施加此项。[默认配置](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py) 的边缘宽度阈值为 5 cm。它是四足项目，权重与接触模型不能原样搬到 G1。
- [FastStair 论文](https://arxiv.org/html/2601.10365v1)：使用 DCM 落脚规划器引导安全踏面落点，随后调整速度与落脚奖励，分别训练高低速专家，再融合。论文报告 Oli 人形机器人上楼最高指令速度 1.65 m/s，不能视作本项目 G1 的能力证明。
- [legged_gym 官方代码](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)：`update_command_curriculum` 在速度跟踪奖励超过最大值的 80% 后扩展指令范围。可借鉴的是“跟踪达标再升速”，本项目还保留通过、停稳和失败检查。

本次实现是基于现有策略的渐进续训，没有复现 FastStair 的 DCM 规划器、专家网络或 LoRA。继续使用完整参考动作和 AMP=0.5，保持网络输入、动作、相机和地形几何兼容。

## 新版本改变了什么

### 速度课程

- 平地各档采样区间：0.45～0.6、0.65～1.0、0.9～1.5、1.2～2.0、1.8～2.5、2.3～3.0 m/s。前四档沿用旧含义。
- 楼梯各档：0.35～0.55、0.55～0.75、0.75～1.0、1.0～1.25、1.25～1.5 m/s。前两档沿用旧含义。
- 上下楼各自累计至少 50 个当前阶段回合，成功率大于 80% 才提高一档速度；当前高度五档完成后提高阶高并回到最低速度档。低于 50% 降一档速度；最低速度仍不通过则降低高度。
- 保留 20% 的前一高度低速练习。平地仍采用独立的连续回合证据。不会因为改了速度上限就立刻把所有环境推到最高速。
- 楼梯升档还要求：至少 0.5 秒的有效巡航片段、平均速度跟踪分数大于 0.6、平均实际前向速度至少达到平均指令的 80%。修订后的中间档允许实际速度达到相邻下一档上限；上界取“当前指令的 120%”和“下一档上限”较大者。最高速度档仍严格要求实际/指令处于 80%～120%，不能借此直接升高台阶。巡航只统计前向指令达到采样速度上限 80% 的片段；终点减速、初始化过渡和手动目标不作为巡航证据。低速精确跟踪单独记录，不把允许进入下一档称为已经掌握低速。
- 指令仍在接近终点时减速。平地目标间距约 6 m，适合训练加速与停转，不能等价于长距离持续 3 m/s 跑步；后续应另做长直道验证。

### 踩边奖励

- `feet_support`：权重 -0.25 → -0.5，支撑采样比例目标 85% → 90%。
- `feet_edge`：权重 -0.1 → -0.2。
- 新任务将前 80 ms 完全不计分改为接触后 40 ms 内线性增加到完整惩罚。弱接触与卸载仍按载荷减弱；腾空不罚。原任务默认行为不变。
- 前向平面速度跟踪权重 2 → 3，其他奖励保持原值。

这些是待验证的训练参数。边缘检测仍为脚底射线近似，不是落脚可行性规划，也没有加入“踩边立即终止”。需要结合足部接触回放与通过率确认是否改善，尤其注意避免惩罚增大后出现停滞。

### 模型恢复

从旧 v4 检查点加载网络、AMP、优化器及原高度/速度档，清空旧成功统计与平地升档证据，防止旧标准直接触发新标准升档。26500 实际保存的上下楼阶段均为 8 cm、最低速度档。

新模型保存课程版本 v5，包括速度表与配置标识。恢复时核对速度表；v5 需要 `Stairs-v1` / `Stairs-Play-v1`，不能交给旧 v0 播放任务。原检查点不修改。3 轮短跑结果仅用于验证，不作为正式续训起点。

升档判定修订标记为 `promotion_version=2`。恢复此前 v5 模型时也会清空旧升档证据，保留阶段；同版本恢复则继续保留证据窗口。分项日志 `ascent/diagnostic_*`、`descent/diagnostic_*` 与 `diagnostic_attempts` 使用同一当前阶段有效回合分母，失败原因可以重叠。

## 升档诊断与长梯实验

对 27200 冻结模型、seed 42、0.45 m/s 指令的短梯诊断中，8 cm 下楼 3 个完整回合均通过并停稳，但都被旧的超速条件拒绝；平均实际速度约 0.645 m/s、平均跟踪分数约 0.639。排除起步前 0.5 秒后仍全部被拒绝。样本很小，仅用于定位条件冲突，不是泛化能力评测。

因此中间档允许进入相邻的、更符合已有实际速度的速度区间，继续保留完成、停稳、终局几何、失败终止与跟踪检查。新增“排除起步前 0.5 秒”的影子统计，暂不以它替换现有统计。

原始诊断：`logs/evaluations/stairs_speed_gate_27200.json`；分项汇总：`logs/evaluations/stairs_speed_gate_27200_summary.json`。

33 级长梯另见 [长梯实验说明](g1_stairs_long.md)。

## 正式续训命令

从原 26500 追加 3000 轮，写入新实验目录。先分段检查速度跟踪、踩边和通过率，再决定继续增加轮数。

```bash
cd /home/ljc/legged_lab_v3
export ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Stairs-G1-v1 \
  --viz none --device cuda:0 --num_envs 2048 \
  --max_iterations 3000 --seed 42 \
  --run_name from26500_flat3_stairs1p5 \
  --resume \
  --load_run /home/ljc/legged_lab_v3/logs/rsl_rl/g1_amp_stairs_specialist/2026-09-15_00-05-33_stairs_window_style05 \
  --checkpoint model_26500.pt agent.device=cuda:0
```

输出：`logs/rsl_rl/g1_amp_stairs_speed/<时间>_from26500_flat3_stairs1p5/`。
以后从该新目录继续时，将 `--load_run` 换成其绝对路径，并将 `--checkpoint` 换成要恢复的模型文件名。

## 高速固定场景评测

评测器已增加任务选择和速度列表。下面的 `$SPEED_CHECKPOINT` 需设置为训练产生的具体模型绝对路径，不能使用字面占位路径。

```bash
bash scripts/run_with_rsl5.sh scripts/tools/compare_stair_checkpoints.py \
  --task LeggedLab-Isaac-AMP-Stairs-G1-Play-v1 \
  --viz none --device cuda:0 --eval-envs 192 --steps 2001 --seed 42 \
  --stair-speeds .45 .75 1.0 1.25 1.5 \
  --flat-speeds .8 1.5 2.0 2.5 3.0 \
  --output "logs/evaluations/stairs_speed_$(date +%Y%m%d_%H%M%S).json" \
  --checkpoints "$SPEED_CHECKPOINT"
```

该评测沿用 8/10/14 cm 固定高度。需同时看实际速度、指令速度、失败与完整通过；只看课程等级或速度上限不能证明已经达到目标。正式评价应重复种子，并保留低速案例防止能力退化。

## 本次验证

- 相关自动检查：32 项通过，1 项 GPU 映射检查在受限 CPU 测试环境跳过。
- GPU 实际续训：64 个环境，从原 26500 加载，完成 3 轮更新并保存 v5 模型，进程正常退出。验证目录为 `logs/rsl_rl/g1_amp_stairs_speed/2026-09-15_15-10-15_speed_v1_smoke/`。
- GPU 独立恢复评测：加载上述 `model_26502.pt`，64 个环境运行 1101 步，完成回合重置；输出包含平地 3.0 m/s 和 8/10/14 cm 上下楼 1.5 m/s 案例，统计值均为有限数，正常退出。
- 短跑证明了迁移、更新、保存、恢复和高速指令评测链路可运行，不证明踩边已减少或目标速度已学会。正式续训仍使用原 26500。
