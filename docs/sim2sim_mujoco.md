# G1 视觉策略 MuJoCo sim2sim

入口直接加载本项目 RSL-RL 5.4.1 的检查点，使用 hiking 工程的 MuJoCo 机器人结构，并以训练 USD 校准机器人参数、碰撞和视觉网格。CPU 即可推理与计算深度，无需启动 Isaac Sim、ROS 或 DDS。

## 启动和选点

```bash
bash scripts/run_sim2sim.sh
```

默认检查点为 `g1_amp_stairs_long/2026-09-16_20-26-52_reverse_height_v11/model_50100.pt`。默认场景为训练同款中央低、四周向上的楼梯：中央主测试区域每级 **12 cm**，踏面 **30 cm**，共 **17 级**，总升高 **2.04 m**，目标速度上限 **0.65 m/s**。

窗口打开后保持初始姿态并暂停仿真，第一次选定目标才开始运行，与训练重置后的起步状态一致。右侧仍显示初始深度，可旋转和缩放视角。

- 左键单击地面或台阶顶面：选定目标；绿色标记显示位置。机器人、墙面、深度面板不可选。
- 左键拖动旋转，右键拖动平移，滚轮缩放，F 切换跟随。
- 空格将目标设在当前位置，Esc 退出；W/S、A/D 修改目标的世界 X/Y 坐标，每次 1 m。
- 深度面板上图为 64×36 原始光轴深度，白框是裁剪范围；下图为策略实际接收的 32×18 最新深度帧。黄色近、蓝色远，范围 0～2.5 m。
- G1 头部可见，但按训练配置不参与深度射线遮挡。

直接运行已验证路线，无需手动找终点：

```bash
bash scripts/run_sim2sim.sh --target 6.7 0
```

无窗口复现（默认目标也是 `(6.7, 0)`）：

```bash
bash scripts/run_sim2sim.sh --headless --duration 30 \
  --report .runtime/sim2sim/validation.json
```

默认解释器为 `/home/ljc/isaaclab/bin/python`，可通过 `ISAACLAB_PYTHON` 覆盖。`--checkpoint` 可指定其他模型，其旁边必须保留 `params/env.yaml`、`params/agent.yaml`。`--speed` 可修改速度；当前高度课程训练范围为 0.55～0.75 m/s，不能假定越慢越稳定。`--depth-delay 1` 可切换为一帧深度延迟，默认 0。

GUI 默认运行直到关闭；`--duration` 限制实际仿真时长，不计第一次点击前等待时间。无窗口默认 60 秒，加 `--real-time` 可按实时速度运行。严重倾倒时退出码为 2，未倾倒为 0；退出码 0 不表示到达，必须同时检查 `target_reached`。

## 本次对齐和验证（2026-09-16）

固定 50100 检查点和初始姿态，对比 Isaac 与 MuJoCo：

- 训练地形直接调用本机 Isaac Lab 的原始几何函数导出，保留中央平台、每级尺寸和环形布局。训练场地沿 X 相接：为复现出口视野，保留相邻的 10 cm、14 cm 课程地块；**出生点所在的主地块为 12 cm**，验证目标位于该地块出口。相邻地块的通过能力未测试。
- 机器人质量、质心、惯量、关节安装、碰撞体和视觉三角网格均来自训练 USD。相机相对位姿、视场、光轴深度、身体遮挡过滤与训练一致。
- 29 关节顺序、默认角、刚度、阻尼、惯量补偿、限力和动作尺度按训练配置加载。物理 200 Hz、策略 50 Hz，MuJoCo 使用原生限力 PD 和 implicitfast 积分。
- 本体输入 495 维：角速度、去偏航后的旋转矩阵第一/第三列、速度指令、绝对关节角、关节速度、前次动作；每项 5 帧，旧帧在前。
- 深度裁剪、3×3 高斯模糊、归一化和 37 帧历史与训练一致，取偏移 35、30、25、20、15、10、5、0，输入 `8×18×32`。
- 将 Isaac 的 12 个静止/行走/登阶/到达姿态放入 MuJoCo 比较：归一化深度单像素最大误差小于 **3×10⁻⁶**，当前帧本体观测最大误差小于 **3×10⁻⁷**。历史顺序另有独立测试。这不表示两个物理引擎动力学完全相同。
- Isaac：同一模型、12 cm 楼梯、0.65 m/s、固定初始状态，30 秒完成并站稳，无重置。参考测试固定摩擦系数 1，关闭质量随机化和外力推扰。
- MuJoCo：30 秒走完整段 17 级并站稳，终点位置约 `(6.648, -0.016, 2.822)`，目标距离 **0.055 m**，无摔倒。
- 实际 GUI 等待目标再启动的 30 秒测试得到相同结果，等待时仿真时间保持为 0；记录为 `aligned_12cm_gui.json`。8 项自动化测试通过，包括四个方向逐级检查 17 个 12 cm 台阶的实际射线高度。

结果见 `sim2sim_validation/aligned_12cm_50100.json`、`aligned_12cm_isaac.json`、`aligned_12cm_observations.json`。这些是固定路线验证，不是所有目标和速度的成功率评估。

**仍有限制：**先以零指令站立 5 秒再突然起步的测试发生失稳，缓慢增加速度也未解决。因此首次目标选择前采用暂停，而不是让策略先运行站立阶段。中途停走、重新起步、任意方向选点及不同检查点的稳定性尚未保证。暂停仅用于开始前，不修改动作或重置行走中的观测历史。

## 原 hiking 场景

原有带墙短楼梯仍保留在原路径，可显式选择：

```bash
bash scripts/run_sim2sim.sh \
  --scene hiking-in-the-wild-sim2sim/unitree_mujoco/unitree_robots/g1/scene_29dof_terrain_with_camera.xml \
  --spawn 0 2 .8 --target 3 2 --speed .55
```

它是 12 cm 高、20 cm 深、上下各 5 级的短楼梯，与训练场景不同；此前测试在第一阶前停滞，不能因为训练同款场景通过就视为这个场景也通过。旧平地/短楼梯结果保存在 `sim2sim_validation`，产生于本次视觉网格对齐之前。

`--control velocity --command VX VY WZ` 保留用于诊断。正常播放采用目标闭环，根据偏航误差修正方向，到目标阈值内停止；固定机身速度指令不能替代训练中的目标控制器。

## 重新导出与测试

```bash
/home/ljc/isaaclab/bin/python scripts/sim2sim/export_robot_profile.py \
  source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd \
  scripts/sim2sim/g1_training_profile.json
/home/ljc/isaaclab/bin/python scripts/sim2sim/export_training_visuals.py \
  scripts/sim2sim/g1_training_profile.json
/home/ljc/isaaclab/bin/python scripts/sim2sim/export_training_terrain.py \
  --config logs/rsl_rl/g1_amp_stairs_long/2026-09-16_20-26-52_reverse_height_v11/params/env.yaml
/home/ljc/isaaclab/bin/python -m pytest -q scripts/sim2sim/test_adapter.py
```

地形导出默认读取 `/home/ljc/isaaclab6/IsaacLab` 的几何函数，可用 `--isaaclab` 改路径。导出额外需要 pxr、scipy、trimesh，运行不需要 Isaac Lab。运行依赖 MuJoCo、torch、numpy、PyYAML、tensordict 和项目 `.runtime/rsl_rl_5_4_1`。模型网格沿用项目 Unitree 训练资源许可；目标控制器保留本项目 InstinctLab 来源及 CC BY-NC 4.0 说明，见 `NOTICE.md`。

## BeamDojo square-course model_8999 (2026-10-11)

The new independent adapter is `scripts/sim2sim/beamdojo_mujoco.py`.
Start the browser play with `scripts/run_beamdojo_mujoco.sh --port 8091`.
The default checkpoint is the completed `square_mixed_generalize_5000_4096/model_8999.pt`.
This is native MuJoCo dynamics with frozen policy weights, not reference-motion playback.
The UI selects difficulty 0–10, cardinal/diagonal maps, variants 0–3, route-facing spawn,
and speed 0.5–1 m/s. Click “应用设置并重新开始” to start again after arrival/failure.

Interface: exact checkpoint joint order, ELU 708→512→256→128→29 network,
identity-interior mean tails and execution bounds from checkpoint metadata,
term-major five-frame proprioceptive history, 225 yaw-aligned elevation cells,
three gyro-integrated course-tracking features, 0.25 joint action scale,
checkpoint PD gains/effort limits/armature, 200 Hz physics / 50 Hz policy.
The USD-calibrated robot profile's asset hash matches the checkpoint contract.
The height input is an ideal terrain query; this does not test a real lidar or camera.
Course tracking uses ideal body gyro and projected gravity, with no absolute-heading policy input.

Terrain is rebuilt from the checkpoint's archived generator and identical 2 cm raster.
The 8 m square map retains the 1.4 m central platform and 1 m outer ring, with gaps at -1 m.
All 160,000 cell-center downward rays match the collision raster, with zero mismatches.
Robot masses, inertias, joint frames and collision shapes use the existing USD-calibrated profile.
MuJoCo contact dynamics remain different from PhysX; they are not claimed identical.

Initial fixed-seed transfer matrix: 16 cases (eight headings × L0/L10), seed42,
variant2, 1 m/s cruise command. 15 reached the outer ring and stood for one second;
L10/180° crossed the map boundary without qualifying parking. This small deterministic
set is not a statistical estimate of broad transfer success. Native recording and
results live in `artifacts/sim2sim_mujoco_8999/`.
Reproduce checks with `/home/ljc/isaaclab/bin/python scripts/sim2sim/validate_beamdojo.py`.
Reproduce video with `MUJOCO_GL=egl /home/ljc/isaaclab/bin/python scripts/sim2sim/record_beamdojo.py`.

## Published demo recordings

The README now links two videos under `assets/demo/`, managed by Git LFS:

- `g1_mujoco_stones_8999.mp4`: new native MuJoCo L10 cardinal/variant2 rollout,
  frozen8999, 1 m/s cruise, with simulation time and terminal status shown.
- `g1_mujoco_stairs_50100.mp4`: byte-identical copy of the 2026-09-22 historical
  `03_stairs_up.mp4`, model50100, 17 × 12 cm stairs, complete30 seconds at original speed.

The clips use different models and observation interfaces. Neither is a reference
motion replay. `assets/demo/manifest.json` records provenance and SHA256 checksums;
the historical ascent's matching trajectory is also published. Source weights and
machine-local training logs remain excluded, as stated in the repository release policy.
