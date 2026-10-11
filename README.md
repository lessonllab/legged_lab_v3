# Legged Lab: G1 Visual Locomotion

![Isaac Sim](https://img.shields.io/badge/Isaac_Sim-6.0-silver)
![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-3.0-silver)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![RSL-RL](https://img.shields.io/badge/RSL--RL-5.4.1-blue)
![Platform](https://img.shields.io/badge/Platform-Linux-orange)

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Current Training Status](#current-training-status)
- [News & Updates](#news-updates)
- [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Acknowledgement](#acknowledgement)

<a id="overview"></a>
## Overview

面向 **Unitree G1 EDU 29 自由度机器人**的运动控制研究项目，由 **ljc** 基于 [Legged Lab](https://github.com/zitongbai/legged_lab) 扩展。当前演示覆盖局部高程图梅花桩策略，以及历史深度视觉 PPO/AMP 楼梯策略；两条分支分别维护训练和验证结果。

**Key Features:**

- 梅花桩任务：中央平台出发、左右交替落脚、外环到达停车，支持正向／斜向路线和地形扰动。
- 全身控制与地形感知：G1 29 关节策略，结合本体状态与局部高程输入。
- MuJoCo sim2sim：冻结检查点推理、训练参数适配、原生碰撞动力学与固定案例评测。
- 浏览器交互播放：选择难度、出生方向和扰动，显示高程采样及射线。
- 深度视觉历史分支：8 帧深度历史、PPO/AMP 与混合楼梯课程，保留 12 cm 和 20 cm 上台阶录像。
- [在线演示播放器](https://lessonllab.github.io/legged_lab_v3/)：直接播放、拖动进度和全屏查看；附视频来源与轨迹。

<a id="demo"></a>
## Demo

### MuJoCo 实际策略演示

[**点击在线播放全部视频**](https://lessonllab.github.io/legged_lab_v3/)：打开页面后直接点击播放，无需下载。

**延长梅花桩：8999 轮模型，12 × 12 m 地图，2 m 宽外环，L10 正向、间距扰动 v2。**中央平台保持 1.4 × 1.4 m，正向桩路延长约 1 m。进入终点平台 1 m 后发零速指令，演示最终停在平台内约 1.25 m；绿色与红色点显示策略实际使用的理想高程采样。

[![G1 MuJoCo 延长梅花桩演示](assets/demo/g1_mujoco_stones12_8999.png)](https://lessonllab.github.io/legged_lab_v3/#stones)

[在线播放梅花桩](https://lessonllab.github.io/legged_lab_v3/#stones) · [原 8 m 地图固定案例评测](artifacts/sim2sim_mujoco_8999/results.json) · [运行与复现说明](docs/sim2sim_mujoco.md)

**20 cm 上台阶：62900 轮视觉模型，17 级、踏面 30 cm，总升高 3.4 m。**保留完整 30 秒和到顶停车；与梅花桩模型属于不同训练分支。

[![G1 MuJoCo 20 cm 上台阶演示](assets/demo/g1_mujoco_stairs20_62900.jpg)](https://lessonllab.github.io/legged_lab_v3/#stairs20)

[在线播放 20 cm 台阶](https://lessonllab.github.io/legged_lab_v3/#stairs20) · [历史 12 cm 台阶视频](https://lessonllab.github.io/legged_lab_v3/#stairs12) · [视频来源与校验值](assets/demo/manifest.json)

运行梅花桩播放：`bash scripts/run_beamdojo_mujoco.sh --port 8091`。默认使用延长地图；页面可选难度 0–10、正向／斜向地图、出生方向与扰动，并切换高程采样和射线显示。修改后点击“应用设置并重新开始”。原 8 m 地图可加 `--map-size 8 --ring-width 1 --stop-depth .2`。需要自行准备检查点及其归档 `params/`，模型文件不随仓库发布。

视频使用冻结策略控制原生 MuJoCo 动力学。单条演示不能替代统计评测；原 8 m 地图固定测试为 15/16 成功，该结果不适用于延长地图。高程点为理想地形查询，不是原始 MID-360 点云。


准备好本地检查点后启动交互演示：

```bash
bash scripts/play_stairs_control.sh /absolute/path/to/model.pt
```

打开 [Viser](http://127.0.0.1:8082/) 查看机器人、地形和深度输入。参考动作重放和 MuJoCo 演示见下文。

**验证范围：**50100 检查点已记录在 MuJoCo 中完成 12 cm、17 级固定路线的 30 秒测试；停走重启和任意场景尚未保证稳定。62900 已完成 20 cm、17 级固定路线的 30 秒测试，任意场景尚未验证，课程晋级不等于通过率验证。详见 [sim2sim 说明](docs/sim2sim_mujoco.md)与[检查点对比](docs/analysis/checkpoint_comparison_20260917.md)。仓库不附带训练检查点。

<a id="current-training-status"></a>
## Current Training Status

**2026/10/11：当前梅花桩演示使用冻结的 8999 轮模型，已接入原生 MuJoCo sim2sim。**正向／斜向地图对应不同桩路布局，机器人出生时面向所选路线；模型输入为本体状态、理想局部高程图和航向跟踪特征。历史 AMP 深度视觉楼梯策略是独立分支，两者的检查点和观测不能混用。

- **原 8 m 地图：**八个出生方向 × L0/L10 共 16 个固定案例，15 个完成到达与站稳；L10/180°未完成停车。该测试不是多种子泛化成功率。[评测记录](artifacts/sim2sim_mujoco_8999/results.json)
- **延长 12 m 地图：**2 m 宽外环，正向桩路延长约 1 m。进入终点平台 1 m 后锁定零速；L10 正向、扰动 v2、seed42 演示在 8.32 s 完成，最终停在平台内约 1.25 m。[演示记录](assets/demo/g1_mujoco_stones12_8999.json)
- **高程可视化：**显示策略实际接收的 15×15 采样点，绿色为支撑面、红色为间隙，可切换射线。这是理想高程输入，尚未验证真实 MID-360 数据接入。
- **20 cm 楼梯：**独立视觉模型 62900 完成 17 级、踏面 30 cm、总升高 3.4 m 的 30 秒 MuJoCo 演示，并在顶端停车。[轨迹记录](assets/demo/g1_mujoco_stairs20_62900.json)

[在线播放](https://lessonllab.github.io/legged_lab_v3/) · [MuJoCo 运行与验证说明](docs/sim2sim_mujoco.md)。本次地图、停车和显示更新未重新训练模型；下一步是延长地图的八方向、多种子评估，以及不规则桩面与感知误差测试。

<a id="news-updates"></a>
## News & Updates

- **2026/10/11**：发布冻结 8999 模型的原生 MuJoCo 梅花桩 sim2sim；原 8 m 地图固定测试通过 15/16，保留失败案例及验证范围。
- **2026/10/11**：播放地图扩展为 12×12 m，外环加宽到 2 m；终点改为进入平台 1 m 后发零速，新增实际高程采样与射线显示。
- **2026/10/11**：录制 62900 视觉模型的 20 cm、17 级台阶完整 30 秒演示；上线[直接播放的视频页面](https://lessonllab.github.io/legged_lab_v3/)，保留历史 12 cm 视频及来源记录。
- **2026/10**：梅花桩任务扩展到中央起点、左右交替桩路与外环终点，支持正向／斜向地图、路线朝向出生及大小／间距扰动；交互播放可选择难度与路线。
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

   当前仓库为公开仓库，无需登录即可查看和克隆。

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

- [x] 深度历史与 PPO/AMP 楼梯训练链路。
- [x] 正向／斜向梅花桩地图、路线朝向出生与交互难度选择。
- [x] 冻结 8999 模型的 MuJoCo 适配及原地图固定案例评测。
- [x] 延长桩路、终点平台内停车和高程输入可视化。
- [x] 在线视频播放器，以及 12 cm／20 cm 台阶录像。
- [ ] 完成延长地图的八方向、多种子评估。
- [ ] 评估不规则桩面、间距和高差的泛化能力。
- [ ] 接入实物雷达或深度相机高程图，测试噪声、延迟与遮挡。
- [ ] 完成奖励、课程与方法消融，以及真机验证。

<a id="acknowledgement"></a>
## Acknowledgement

感谢 [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)、[BeamDojo](https://arxiv.org/abs/2502.10363) 及其 [humanoid-gym 适配实现](https://github.com/ezoicoder/humanoid-gym)、[Legged Lab](https://github.com/zitongbai/legged_lab)、[Isaac Lab](https://github.com/isaac-sim/IsaacLab)、[RSL-RL](https://github.com/leggedrobotics/rsl_rl)、[AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)、[GMR](https://github.com/YanjieZe/GMR)、[MimicKit](https://github.com/xbpeng/MimicKit)、[InstinctLab](https://github.com/project-instinct/InstinctLab) 和 [hiking-in-the-wild-sim2sim](https://github.com/jie0110/hiking-in-the-wild-sim2sim)。

原始代码许可证见 [LICENCE](LICENCE)，上游引用及第三方许可见 [NOTICE.md](NOTICE.md)。部分适配文件标注 CC BY-NC 4.0；机器人、动作数据及其他第三方内容沿用各自许可。
