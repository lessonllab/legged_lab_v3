# G1 深度视觉 AMP：使用与验证

2026-09-10：原 rough 视觉基线出现原地停留，已新增 [平地直行预训练配置](g1_depth_warmup.md)。
当前建议从该 warmup 任务开始；以下保留原视觉崎岖任务的说明。

新增任务 `LeggedLab-Isaac-AMP-Depth-G1-v0` 和 `LeggedLab-Isaac-AMP-Depth-G1-Play-v0`。
这是可训练的单阶段视觉 PPO+AMP 基线，采用不对称 actor/critic；没有实现教师蒸馏或残差门控。
网络接入成功不等于已训练出可部署的行走策略。

## 输入与动作

- actor：`policy`（495 维本体与指令历史）和 `depth`（8×36×64）。不读取理想高度扫描、仿真线速度或地形编号。
- critic：保留原 rough 配置的 787 维特权观测，包含高度扫描。
- discriminator：保持动作状态历史，无图像和地形信息。
- 视觉网络：RSL-RL `CNNModel`，卷积通道 16/32/32，MLP 256/128，输出 29 维关节动作。
- 相机：静态地形 RayCasterCamera，名义头部安装位置，约 48°向下俯视；64×36，25 Hz。控制仍为 50 Hz。
- 时序：ObservationManager 保存 8 个控制时刻的图像（160 ms，期间允许重复帧），reset 由同一管理器清空，不是 8 个独立相机帧。
- 深度：输入必须是以米为单位的光轴深度，0.1～2.5 m 映射到 [0,1]；无效/过近值为 -1，有限远距离截为 1。
- 指令：前进 0.2～1.0 m/s，横移 0，yaw-rate ±0.5 rad/s；保留 standing command 分支。关闭 heading controller 和推扰。

对称增强仅在新视觉任务中关闭，避免图像与动作错误配对。原 rough/flat 策略输入和网络保持原样。
初始地形等级设为 0，默认并行数 256；现有 rough 地形集合与奖励保留，可通过课程增加难度。

## 当前机器的运行环境

使用用户启动说明中的环境：

```bash
conda deactivate
source /home/ljc/isaaclab/bin/activate
cd /home/ljc/legged_lab_v3
```

该环境的默认 `rsl_rl` 指向 AME_Locomotion 的 3.0.1 分支，缺少当前项目所需的
`rsl_rl.models` / `rsl_rl.extensions`。项目使用 `.runtime/rsl_rl_5_4_1/` 中的独立
RSL-RL 5.4.1，由 `scripts/run_with_rsl5.sh` 为当前进程设置导入路径，不覆盖用户环境或其他项目。
如在新机器上没有此目录，先安装该纯 Python 包（其余依赖由 Isaac Lab 环境提供）：

```bash
python -m pip install --no-deps --target .runtime/rsl_rl_5_4_1 rsl-rl-lib==5.4.1
```

启动脚本使用当前已激活的 Python，也可显式设置 `ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python`。
已兼容本机 Isaac Lab 的 velocity task 新命名空间及 sim_launcher 入口。

请通过上面的启动脚本运行训练和播放。脚本在导入 NumPy/SciPy 前设置
`OPENBLAS_NUM_THREADS=1`，处理本机 Kit 启动时在 `fork → blas_thread_shutdown_`
调用链出现的崩溃（退出码 139）；OMP/MKL 线程数未指定时默认设为 1。
这些设置只影响本次启动的进程，不修改环境激活文件，也不限制 GPU 上的并行机器人数量。
线程设置依据见 [OpenBLAS 使用说明](https://github.com/OpenMathLib/OpenBLAS/blob/develop/USAGE.md)。

修复后已在同一 Python 环境以 **512 个机器人、默认地形网格、默认 PPO 参数**完成
2 轮训练并正常退出（退出码 0），运行名为 `depth_512_startup_fix`。
该检查只缩短了训练轮数，没有缩减环境数、地形网格或 PPO 每轮采样量。
验证日志保存在 `logs/rsl_rl/g1_amp_depth/2026-09-09_21-28-32_depth_512_startup_fix/startup_validation.log`。

## 先检查相机

```bash
bash scripts/run_with_rsl5.sh scripts/tools/inspect_amp_depth.py --viz none
```

默认 4 个环境、8 个零动作步骤，仅用于相机与 reset 检查，不是策略评估。
输出到 `.runtime/g1_depth_check/`：

- `depth_metres.npy`：环境 0 初始相机的原始深度。
- `depth.png`：固定量程灰度图；黑色无效，白色为 2.5 m 或更远。
- `summary.json`：形状、有效像素比例、深度范围和 reset 结果。

## 训练

先小规模试跑：

```bash
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-G1-v0 --viz none \
  --num_envs 4 --max_iterations 2 --run_name depth_smoke \
  env.scene.terrain.terrain_generator.num_rows=2 \
  env.scene.terrain.terrain_generator.num_cols=6 \
  agent.num_steps_per_env=8 \
  agent.algorithm.num_learning_epochs=1 agent.algorithm.num_mini_batches=1 \
  agent.algorithm.amp_cfg.disc_update_interval=1
```

正常训练起点（耗时随设备和任务而变，未在本轮启动长训练）：

```bash
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-G1-v0 --viz none \
  --num_envs 256 --max_iterations 20000 --run_name depth_baseline
```

这是从头训练的独立任务，不能直接 resume 原 flat/rough 的 MLP checkpoint。
日志写入 `logs/rsl_rl/g1_amp_depth/`。若恢复同一视觉任务，按已有 resume 参数指定对应运行。

## 播放与导出

```bash
bash scripts/run_with_rsl5.sh scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-Depth-G1-Play-v0 --viz kit --num_envs 4 \
  --checkpoint /absolute/path/to/visual_checkpoint.pt
```

PLAY 保持相机、本体观测及网络输入不变，关闭质量/摩擦随机化、观测噪声及地形课程。
原 play 流程通过 RSL-RL 导出包含 CNN 的 TorchScript/ONNX。
ONNX 输入为 `obs`（本体历史）和 `depth`（已处理的图像历史），输出为 `actions`。
导出模型不包含相机采集、米制转换或历史队列；部署端需复现这些步骤。

## 验证范围与下一步

CPU 测试覆盖：无效深度、数值范围、真实 ObservationManager 的图像形状与部分环境 reset、
actor 特权信息隔离、CNN 梯度、TorchScript 及 ONNX wrapper 输出一致性。
测试命令：

```bash
bash scripts/run_with_rsl5.sh source/legged_lab/test/test_amp_depth.py
```

2026-09-09：在用户指定的 `/home/ljc/isaaclab/bin/python` 环境中，上述 3 项测试和
平地速度指令的 3 项回归测试全部通过；真实仿真 4 环境、每轮 8 步 rollout、2 轮训练更新
完成并生成 checkpoint，试跑时将判别器更新间隔设为 1，以覆盖 PPO 和 AMP 判别器更新。
相机独立检查通过，包含图像有效性和真实环境 reset 后的历史清空。
这是连通性验证，日志中的短程 success_rate 不能当作行走通过率。

本机试跑产物位于 `logs/rsl_rl/g1_amp_depth/2026-09-09_21-16-00_depth_user_env_smoke/`：
`model_1.pt` 已成功导出为 `exported/policy.pt` 和 `exported/policy.onnx`。
TorchScript 重新加载后输出与原网络一致，ONNX 文件通过结构检查；本环境没有 ONNX Runtime，
因此尚未验证 ONNX Runtime 的数值一致性。试跑模型仅用于检查流程，不能用于实机行走。

实机之前仍需：

1. 核对当前 G1 USD 与实机相机外参、流内参及视野。现在使用的是名义近似，不是实机标定。
2. 补充身体遮挡：当前只射线查询静态地形，没有动态身体遮挡。
3. 根据实际相机流加入延迟、掉帧和深度噪声；目前仅模拟 25 Hz 帧保持和无效值处理。
4. 对缓坡、低台阶、凹凸地面分别训练和验收，并以冻结/打乱深度图做消融。
5. 若采用教师蒸馏，另加学生 rollout 的教师标注与 checkpoint 适配；当前代码不执行蒸馏。
6. 校验关节顺序、动作尺度、PD 与部署端一致，再进行 sim2sim 和低速实机评估。
