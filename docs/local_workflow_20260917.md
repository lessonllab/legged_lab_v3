# ljc · G1 视觉运动控制

维护者：**ljc**

面向 Unitree G1 EDU 29 自由度机器人的深度视觉行走研究项目。基于 Isaac Lab 和 PPO/AMP，使用前向深度历史与本体状态学习平地行走、楼梯和崎岖地形运动，并通过 Viser 在浏览器中交互查看策略与参考动作。

本地项目目录：`/home/ljc/legged_lab_v3`。Python 包名 `legged_lab` 和现有任务 ID 保持稳定，便于加载历史模型。

## 当前进展（2026-09-17）

- 当前主线为 **Foothold v12**：深度视觉 PPO/AMP，AMP 风格奖励系数 **0.25**，保留 30 段参考动作。
- 楼梯固定 **17 级**，高度档位为 **8、10、12、14、16、18、20、25、30 cm**；上下楼独立自适应晋降级，不要求到出口后站稳。
- 高度课程速度为 0.55～0.75 m/s，同时保留 0.2～0.5 m/s 慢速、高速 8 cm 楼梯、较低高度复习及更高一级挑战。平地目标上限 3 m/s，楼梯指令上限 1.5 m/s；这些不代表已实现的速度。
- 环境分配：平地 25%、上楼 25%、下楼 20%、块状 15%、起伏 10%、坡面 5%。奖励包含 Instinct 风格踩边检测、脚尖撞立面、落脚支撑、持续停滞、转弯蹭地与抬脚、地形净空等项。
- **训练已手动停止**。最近一次日志分析到 62958 回合，最后保存模型为 **62900**。课程阶段为上楼 25 cm、下楼 30 cm；低速高台阶和自然转弯仍是短板，不能将课程晋级视为独立测试通过。
- Viser 支持单机器人、手动地形/阶高/速度调节、点击地面设目标、放大的策略深度输入和视觉范围显示。

当前续训模型：
`logs/rsl_rl/g1_amp_stairs_long/2026-09-17_20-54-41_foothold_v12/model_62900.pt`。

参考：[v12 奖励与课程](analysis/foothold_v12_20260917.md)、[30 cm 上限与迁移](analysis/height30_20260917.md)、[54000/55000/58999 固定条件对比](analysis/checkpoint_comparison_20260917.md)。最新 62900 尚未完成独立固定条件评估，不能直接认定优于全部旧模型。

## 运行环境

当前机器已验证的环境：

- Linux、NVIDIA GPU；Python 3.12。
- Isaac Sim 6.0 / Isaac Lab 3.0；本机 Isaac Lab 位于 `/home/ljc/isaaclab6/IsaacLab`。
- Python 解释器：`/home/ljc/isaaclab/bin/python`。
- RSL-RL 5.4.1：独立放在 `.runtime/rsl_rl_5_4_1/`，由包装脚本设置导入路径。
- 当前浏览器播放使用 **PhysX + Viser**，不需要切换到 Newton。

每个新终端先设置：

```bash
cd /home/ljc/legged_lab_v3
export ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python
```

已配置好的本机可直接使用后面的命令。迁移到新环境时，在安装并配置 Isaac Lab 后安装本项目和独立依赖：

```bash
"$ISAACLAB_PYTHON" -m pip install -e source/legged_lab
"$ISAACLAB_PYTHON" -m pip install --no-deps \
  --target .runtime/rsl_rl_5_4_1 'rsl-rl-lib==5.4.1'
```

独立安装只提供 RSL-RL 包本身，其余依赖沿用已配置的 Isaac Lab 环境。机器人资源和动作数据采用 Git LFS；新克隆需要正确下载 LFS 内容。不要把 `.runtime/` 当作无用文件删除，它包含当前运行所需的依赖。

## 主线任务

- `LeggedLab-Isaac-AMP-Foothold-G1-v0`：当前 v12 训练入口。
- `LeggedLab-Isaac-AMP-Foothold-G1-Play-v0`：对应 PLAY 配置。
- `scripts/play_stairs_control.sh` 使用兼容策略输入的 `LeggedLab-Isaac-AMP-Stairs-Long-G1-Play-v0`，只加载模型权重并固定测试地形。

旧版 Scratch、Stairs、Depth 等配置仍供继承、历史模型和对照实验使用。

## 播放最新模型

若 8082 已有播放器，先在其终端按 `Ctrl+C`，再执行：

```bash
cd /home/ljc/legged_lab_v3
bash scripts/play_stairs_control.sh
```

脚本自动选择最新正式检查点，排除 smoke 运行。也可明确指定本次保留的 62900：

```bash
STAIR_TEST_SPEED=0.65 bash scripts/play_stairs_control.sh \
  /home/ljc/legged_lab_v3/logs/rsl_rl/g1_amp_stairs_long/2026-09-17_20-54-41_foothold_v12/model_62900.pt
```

打开 [Viser](http://127.0.0.1:8082/)。默认单机器人、10 cm 上楼；右侧可选最高 30 cm、上下楼/块状/起伏/平地等地形，点击“应用地形并重置机器人”。楼梯固定 17 级，点击地面可设置目标。默认速度上限 0.8 m/s，上述指定命令为 0.65 m/s。

播放不会自动跟随新保存的权重更新，查看新模型需要重启播放器。

## MuJoCo sim2sim

已接入 `hiking-in-the-wild-sim2sim` 的 G1 机器人和地形，可直接加载本项目视觉策略：

```bash
bash scripts/run_sim2sim.sh
```

默认使用 50100 轮检查点和对齐训练的 12 cm、17 级楼梯，速度上限 0.65 m/s。左键选点后开始仿真，右侧显示原始深度和策略输入；空格停止。固定路线已通过 30 秒登阶并站稳验证；原 hiking 短楼梯和中途停走重启仍未保证稳定。更换模型、视角操作和验证结果见 [sim2sim 使用说明](sim2sim_mujoco.md)。

## 训练与恢复

### 从头训练混合地形主线

```bash
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Scratch-G1-v1 \
  --viz none --device cuda:0 --num_envs 2048 \
  --max_iterations 30000 --seed 42 \
  --run_name ljc_visual_scratch agent.device=cuda:0
```

### 继续当前 v12 训练

```bash
cd /home/ljc/legged_lab_v3
TRAIN_ITERATIONS=2100 bash scripts/resume_stairs_foothold.sh \
  /home/ljc/legged_lab_v3/logs/rsl_rl/g1_amp_stairs_long/2026-09-17_20-54-41_foothold_v12/model_62900.pt
```

从已保存的 62900 追加 2100 轮，目标为原定 65000（运行器最终检查点通常为 64999）。停止前尚未保存的更新不会恢复。省略模型参数会自动选最新正式 v12；省略 `TRAIN_ITERATIONS` 默认追加 10000 轮。默认 2048 个环境，可通过 `NUM_ENVS` 修改；脚本拒绝重复启动同类训练。

当前课程保留已学高度和速度阶段；每方向至少 100 次尝试及一个回合时间，窗口通过率达到 80% 升一级、低于 50% 降一级，最高 30 cm。20 cm 旧模型支持迁移，保留策略/AMP/优化器并重新统计晋级证据。

新训练写入新目录，源检查点保留。跨任务加载需匹配检查点迁移规则。

### 查看训练记录

```bash
"$ISAACLAB_PYTHON" -m tensorboard.main \
  --logdir logs/rsl_rl/g1_amp_stairs_long \
  --host 127.0.0.1 --port 6006
```

打开 [TensorBoard](http://127.0.0.1:6006/)。重点结合查看上下楼 `height_cm`、`speed_phase`、`window_pass_rate`，以及 slow/crossings、slow/successes、块状地形失败比例、转弯漂移和速度跟踪。慢速穿越率与速度合格率需分别查看。

## 重放参考动作

重放参考动画不会加载策略，不代表机器人已经学会该动作。以下命令读取训练运行保存的动作列表与权重：

```bash
bash scripts/run_with_rsl5.sh scripts/tools/replay_amp_reference.py \
  --run_dir logs/rsl_rl/g1_amp_stairs_specialist/2026-09-15_00-05-33_stairs_window_style05 \
  --port 8081
```

打开 [参考动作播放器](http://127.0.0.1:8081/)。添加 `--list_motions` 可列出片段名称；添加 `--motion 片段名称` 可指定一个片段。

## 固定场景评测

当前 17 级评测使用 `compare_stairs_checkpoints.py`，固定种子、初始条件及第一回合，比较旧 55000 基线与最新模型：

```bash
bash scripts/run_with_rsl5.sh scripts/tools/compare_stairs_checkpoints.py \
  --viz none --device cuda:0 --num_envs 32 --seed 42 \
  --cases slowup20 slowdown20 up20 down20 boxes rough reverse turn_left turn_right \
  --output docs/analysis/comparison_55000_62900.json \
  --checkpoints \
  logs/rsl_rl/g1_amp_stairs_long/2026-09-16_20-26-52_reverse_height_v11/model_55000.pt \
  logs/rsl_rl/g1_amp_stairs_long/2026-09-17_20-54-41_foothold_v12/model_62900.pt
```

这是待运行的评测命令，不是已有结果。上述楼梯案例固定为 20 cm（慢速 0.3 m/s，普通默认 0.65 m/s），不能验证 25/30 cm；更高台阶可先用交互播放检查。单种子结果不能替代多种子验证，转弯自然程度也不能仅用漂移判断。

历史短楼梯评测工具 `compare_stair_checkpoints.py` 和报告仍保留。

## 检查点保留与清理

2026-09-17 已删除四个密集保存的长楼梯训练目录中的 **186 个冗余中间 `.pt`**，释放约 **3.42 GiB**。保留每个被清理运行的首个、最近三个、每千回合节点，以及代码/报告引用的检查点；最新 62900、当前播放 61700、旧基线 55000、sim2sim 默认 50100 均保留。

本次未删除训练日志、TensorBoard、运行参数、导出策略、参考动作及其他 `.pt` 数据。完整删除和保留清单见 [清理清单](analysis/checkpoint_cleanup_20260917.json)。已删除的中间权重不能再用于逐点复测。

## 项目目录

```text
source/legged_lab/legged_lab/
  tasks/locomotion/amp/     任务、奖励、课程和视觉配置
  rsl_rl/amp/               AMP 与检查点恢复逻辑
  data/                    参考动作与机器人资源
scripts/
  rsl_rl/                  训练、播放和 Viser
  tools/                   参考重放、检查和模型评测
  experiments/             历史批量实验工具
  run_with_rsl5.sh          当前环境的统一启动入口
docs/                      使用文档、研究记录与评测证据
logs/                      本机训练输出和模型，默认不提交
.runtime/                  本机依赖与可再生输出，默认不提交
```

文档索引见 [docs/README.md](README.md)。无用缓存可以重新生成；模型、参考动作、原始 TensorBoard 事件和评测证据应单独确认后再清理。

## 来源与许可

`ljc` 为当前衍生版本的维护者。上游项目来源、引用与第三方许可说明集中在 [NOTICE.md](../NOTICE.md)，原许可证保留在 [LICENCE](../LICENCE)。源码原有的版权与单独许可声明继续保留。

