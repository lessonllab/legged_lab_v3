# Legged Lab: G1 Visual Locomotion

![Isaac Sim](https://img.shields.io/badge/Isaac_Sim-6.0-silver)
![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-3.0-silver)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![RSL-RL](https://img.shields.io/badge/RSL--RL-5.4.1-blue)
![Platform](https://img.shields.io/badge/Platform-Linux-orange)

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [News & Updates](#news-updates)
- [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Acknowledgement](#acknowledgement)

<a id="overview"></a>
## Overview

面向 **Unitree G1 EDU 29 自由度机器人**的深度视觉运动控制研究项目，由 **ljc** 基于 [Legged Lab](https://github.com/zitongbai/legged_lab) 扩展。项目独立于 Isaac Lab 开发，使用 PPO 和 AMP 动作先验学习平地、楼梯与崎岖地形运动。

**Key Features:**

- 深度视觉与本体状态融合：8 帧深度历史，支持延迟和机器人自身遮挡。
- PPO/AMP：本地 AMP 实现基于 RSL-RL 5.4.1，当前主线保留 30 段参考动作，风格奖励系数 0.25。
- Foothold v12：落脚支撑、踩边与碰撞约束、低速跨阶和转向训练。
- 混合地形与自适应课程：固定 17 级楼梯，上下楼独立晋降级，阶高配置覆盖 8–30 cm。
- PhysX + Viser：浏览器播放、点击目标、调整地形和观察策略深度输入。
- MuJoCo sim2sim：机器人参数、观测与地形对齐，以及固定路线验证。
- 保留上游 DeepMimic、动画重放和速度控制任务，供研究及历史模型使用。

<a id="demo"></a>
## Demo

准备好本地检查点后启动交互演示：

```bash
bash scripts/play_stairs_control.sh /absolute/path/to/model.pt
```

打开 [Viser](http://127.0.0.1:8082/) 查看机器人、地形和深度输入。参考动作重放和 MuJoCo 演示见下文。

**验证范围：**50100 检查点已记录在 MuJoCo 中完成 12 cm、17 级固定路线的 30 秒测试；停走重启和任意场景尚未保证稳定。最新 62900 尚未完成独立固定条件评估，课程晋级不等于通过率验证。详见 [sim2sim 说明](docs/sim2sim_mujoco.md)与[检查点对比](docs/analysis/checkpoint_comparison_20260917.md)。仓库不附带训练检查点。

<a id="news-updates"></a>
## News & Updates

- **2026/09/22**：按上游 README 结构整理安装、使用、路线图与发布范围说明。
- **2026/09/17**：Foothold v12 加入低速跨阶、转向接触和块状地形训练，阶高上限扩展至 30 cm。训练手动停止，最后保存检查点为 62900。
- **2026/09/16**：完成 MuJoCo 机器人与视觉输入对齐，记录固定 12 cm 楼梯路线验证。
- **2026/09**：扩展深度历史、混合地形课程、脚底支撑检测及 Viser 交互播放。

详细证据见 [文档索引](docs/README.md)。本机路径、模型保留记录和完整操作说明保留在 [2026-09-17 工作流](docs/local_workflow_20260917.md)。

<a id="installation"></a>
## Installation

### Prerequisites

- Linux、NVIDIA GPU（Isaac Lab 训练）。
- 已安装并配置的 Isaac Sim 6.0 / Isaac Lab 3.0、Python 3.12 环境。
- Git LFS，用于机器人 USD 与参考动作资源。
- RSL-RL 5.4.1，按下文隔离安装。

这些版本来自本项目本机运行环境。Isaac Lab 安装见其[官方文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)。旧 Docker 工具已归档；本版本使用本地环境，不沿用上游旧版 RSL-RL AMP 分支的安装命令。

### Setup Steps

1. 将本仓库克隆到 Isaac Lab 目录之外，进入仓库根目录。

   ```bash
   git clone https://github.com/lessonllab/legged_lab_v3.git
   cd legged_lab_v3
   ```

   当前仓库为私有仓库，需要仓库访问权限。

2. 下载 Git LFS 资源：

   ```bash
   git lfs install
   git lfs pull
   ```

3. 激活已经配置好的 Isaac Lab Python 环境，然后安装：

   ```bash
   export ISAACLAB_PYTHON="$(command -v python)"
   "$ISAACLAB_PYTHON" -m pip install -e source/legged_lab
   "$ISAACLAB_PYTHON" -m pip install --no-deps \
     --target .runtime/rsl_rl_5_4_1 'rsl-rl-lib==5.4.1'
   ```

   隔离库其余依赖沿用 Isaac Lab 环境；这不是空 Python 环境的一键安装命令。后续通过 `scripts/run_with_rsl5.sh` 设置导入路径。运行所需的 `.runtime/` 不应删除。

4. 可选：准备 MuJoCo 外部仓库，版本和场景补丁见[发布范围说明](docs/repository_release.md)。

<a id="usage"></a>
## Usage

### 1. Prepare Motion Data

仓库使用 Git LFS 提供 `source/legged_lab/legged_lab/data/MotionData` 中的参考动作，以及 `data/Robots` 中的机器人资源。

增加动作时，可先用 [GMR](https://github.com/YanjieZe/GMR) 重定向，再转换数据：

```bash
"$ISAACLAB_PYTHON" scripts/tools/retarget/dataset_retarget.py \
  --robot g1 --input_dir temp/gmr_data/ --output_dir temp/lab_data/ \
  --config_file scripts/tools/retarget/config/g1_29dof.yaml --loop clamp
```

将转换结果放入 `data/MotionData` 并更新任务动作配置。格式见 [gmr_to_lab.py](scripts/tools/retarget/gmr_to_lab.py)。

### 2. Training & Play

<details open>
<summary>Train: 从头训练混合地形策略</summary>

```bash
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Scratch-G1-v1 \
  --viz none --device cuda:0 --num_envs 2048 \
  --max_iterations 30000 --seed 42 \
  --run_name visual_scratch agent.device=cuda:0
```

</details>

<details>
<summary>Resume: 继续 Foothold v12 训练</summary>

需要兼容检查点，跨任务加载受课程迁移规则约束。`--max_iterations` 表示本次追加轮数。

```bash
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Foothold-G1-v0 \
  --viz none --device cuda:0 --num_envs 2048 \
  --max_iterations 2100 --seed 42 --run_name foothold_v12 \
  --resume --load_run /absolute/path/to/run --checkpoint model_62900.pt \
  agent.device=cuda:0
```

本机快捷脚本 `scripts/resume_stairs_foothold.sh` 包含本机解释器路径，迁移时优先使用上述入口。

</details>

<details>
<summary>Play: Viser 交互播放</summary>

```bash
STAIR_TEST_SPEED=0.65 bash scripts/play_stairs_control.sh \
  /absolute/path/to/model.pt
```

默认单机器人，使用兼容输入的 Stairs-Long PLAY 任务。打开 [Viser](http://127.0.0.1:8082/)。迁移机器前检查脚本的解释器默认路径；播放不会自动切换到新保存的模型。

</details>

<details>
<summary>Reference: 重放 AMP 参考动作</summary>

```bash
bash scripts/run_with_rsl5.sh scripts/tools/replay_amp_reference.py \
  --run_dir /absolute/path/to/run --port 8081
```

运行目录需要包含动作配置。参考重放不加载策略，不代表机器人已经学会这些动作。

</details>

### 3. Evaluation & Sim2sim

固定条件比较：

```bash
bash scripts/run_with_rsl5.sh scripts/tools/compare_stairs_checkpoints.py \
  --viz none --device cuda:0 --num_envs 32 --seed 42 \
  --cases slowup20 slowdown20 up20 down20 boxes rough reverse turn_left turn_right \
  --output logs/checkpoint_comparison.json \
  --checkpoints /absolute/path/to/baseline.pt /absolute/path/to/candidate.pt
```

这些楼梯案例为 20 cm，不能验证 25/30 cm。应使用多种子，并分别统计穿越、失败与超时。

MuJoCo 需要外部资源、依赖，以及检查点旁的 `params/env.yaml` 和 `params/agent.yaml`：

```bash
bash scripts/run_sim2sim.sh --checkpoint /absolute/path/to/model.pt
```

依赖、资源导出和验证边界见 [MuJoCo 文档](docs/sim2sim_mujoco.md)。

### Repository Contents

```text
source/legged_lab/       Python 包、任务、算法、测试、机器人与动作资源
scripts/                训练、播放、重定向、评测与 sim2sim 工具
docs/                   使用说明、实验记录与评测证据
.vscode/                共享开发配置
```

沿用上游的源码和资源提交范围，增加本项目测试与研究文档。训练输出 `logs/`、本机依赖 `.runtime/`、缓存、私有环境配置及独立第三方仓库不上传。发布文件总量限制为 **1,000,000,000 字节**，按 LFS 资源实际大小核算。对照说明见 [repository_release.md](docs/repository_release.md)。

<a id="roadmap"></a>
## Roadmap

- [x] 深度历史输入与 PPO/AMP 训练链路。
- [x] 混合地形与独立上下楼课程。
- [x] Viser 交互播放及视觉输入诊断。
- [x] MuJoCo 适配和固定路线验证。
- [ ] 完成最新 v12 检查点的固定条件、多种子对比。
- [ ] 提升低速高台阶、自然转向和停走重启稳定性。
- [ ] 对新增奖励和课程进行消融评估。
- [ ] 扩展跨地形泛化与真机验证。

<a id="acknowledgement"></a>
## Acknowledgement

感谢 [Legged Lab](https://github.com/zitongbai/legged_lab)、[Isaac Lab](https://github.com/isaac-sim/IsaacLab)、[RSL-RL](https://github.com/leggedrobotics/rsl_rl)、[AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)、[GMR](https://github.com/YanjieZe/GMR)、[MimicKit](https://github.com/xbpeng/MimicKit)、[InstinctLab](https://github.com/project-instinct/InstinctLab) 和 [hiking-in-the-wild-sim2sim](https://github.com/jie0110/hiking-in-the-wild-sim2sim)。

原始代码许可证见 [LICENCE](LICENCE)，上游引用及第三方许可见 [NOTICE.md](NOTICE.md)。部分适配文件标注 CC BY-NC 4.0；机器人、动作数据及其他第三方内容沿用各自许可。
