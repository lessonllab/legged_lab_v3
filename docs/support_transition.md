# 支撑转换实验：保留正常步态的梅花桩动作修正

2026-10-20 发布，2026-10-22 将梅花桩视频展示更新为仅16700；正式实验完成于 2026-10-16。原 14700 模型保留为主基线，16700 为偏航更小的实验候选。本轮没有楼梯训练、真实传感器或实机测试。

## 设计与结果

第一阶段 14700→15700 的新增全身残差伴随偏航和摆臂协调退步。第二阶段从该 15700 检查点完整恢复参数与 Adam，再训练 **1000 次 PPO 更新、4096 环境**：

- 冻结原 14700 演员及其归一化。附加修正只作用于十二个腿部、三个腰部关节；原网络仍动态控制全部 29 个关节，包括手臂。
- 对下一次触地的离散位置分布、剩余时间和触地后的身体／载荷状态做事件监督。预测器与 PPO 控制梯度分开，避免预测特征同时被两个目标改变。
- 保留残差幅度正则 0.005，新增批内平均残差正则 0.02。没有固定手臂姿态、摆臂幅度或迈步相位。
- 原运动、落脚、航向、停车奖励与 11700 教师软保持项保留；50% 平地、50% 桩路，保留正向／斜向地图、泛化变体和自动升降级。

每次配对包含 64 个已有地图和 128 个新种子生成的同族场景，共 **192 个桩路案例**。14700、15700 和 16700 均完成到达与停车：

- 14700→16700：整段平均绝对偏航 **2.172°→1.699°**；身体晃动 **1.587°→1.678°**，约增加 6%。
- 15700→16700：偏航 **2.604°→1.686°**；左臂／右脚协调相关性 **0.430→0.631**，右侧没有清晰改善。
- 同足步幅约 **49 cm**，交替触地比例为 1，过桩腾空帧比例为 0；相邻左右触地间隔由约 0.343 s 变为 0.355 s。

两组配对各自运行，16700 重复测试不能称为 384 张独立地图。只有一个训练种子；三个修改没有逐项消融。基线通过率已经达到本批测试上限，不能宣称通过率提升。臂腿相关性与角动量分解是轨迹指标，不是摆臂帮助稳定的因果证明。

[16700 梅花桩演示](https://lessonllab.github.io/legged_lab_v3/support_transition_stage2/) · [完整诊断与取舍](analysis/support_transition_stage2_20261016.md) · [数据、视频及校验值](../artifacts/support_transition_stage2/README.md)

## 环境与模型准备

训练和原生评估使用 **Isaac Sim 6.0.1、Isaac Lab 3、Python 3.12、RSL-RL 5.4.1**。沿用根 README 的 Isaac Lab 环境与项目安装步骤。隔离安装 RSL-RL 后运行：

```bash
export ISAACLAB_PYTHON=/absolute/path/to/isaaclab/python
"$ISAACLAB_PYTHON" -m pip install --no-deps --target .runtime/rsl_rl_5_4_1 rsl-rl-lib==5.4.1
export PYTHONPATH="$PWD/scripts:$PWD/scripts/tools${PYTHONPATH:+:$PYTHONPATH}"
```

每个检查点必须和原训练目录的 `contract.json`、`params/` 一起使用；加载器会核对完整观测／控制合同及归档演员源码哈希，不能只复制一个 `.pt` 后忽略其依赖。

- 原始 14700 SHA256：`51045fad72a8d49c4d20624fa95655867a3c8ae7208c4abc6e86197f7d94d8ef`。
- 第一阶段 15700 SHA256：`5547f0c07ee835bfa679b34f543497e6bd334b3227f91cf6e2a518c84eedae93`。
- 修正后 16700 SHA256：`b0ee6a9bf3c501d0e231f20a1dab308b10e1010eedc55a891c0b3248bde1b9a0`。

模型权重、优化器状态和完整逐帧轨迹体积较大，不随本次代码发布；公开证据中保留其来源路径和哈希。下列命令需要自行准备匹配模型，并保持合同中引用的基础／教师检查点可访问；如迁移目录，须保留原引用目录结构。历史机器路径不是新的安装要求。

## 播放与成对评估

```bash
bash scripts/run_with_rsl5.sh scripts/play_beamdojo_square_mixed.py \
  --checkpoint /absolute/path/to/model_16700.pt \
  --device cuda:0 --visualizer none --viser_port 8095 \
  --initial_family cardinal --route_degrees 0 --terrain_level 10 \
  --variant 7 --speed 0.75 --interactive_commands \
  --follow_view front10 --start_auto_course
```

浏览器打开 `http://127.0.0.1:8095/`。页面提供地形、出生方向与指令调节。第一次启动前确认同端口没有旧播放进程；不要并行启动多个原生 Isaac 场景。

```bash
bash scripts/run_with_rsl5.sh scripts/tools/compare_square_policy_effects.py \
  --reference /absolute/path/to/model_14700.pt \
  --candidate /absolute/path/to/model_16700.pt \
  --environment /absolute/path/to/model_14700.pt \
  --labels 14700 corrected16700 --variants 0 7 \
  --arm-audit --output .runtime/support_transition_saved \
  --device cpu --visualizer none

bash scripts/run_with_rsl5.sh scripts/tools/audit_square_arm_coordination.py \
  --input .runtime/support_transition_saved
bash scripts/run_with_rsl5.sh scripts/tools/audit_support_transition_stage1.py \
  --input .runtime/support_transition_saved
```

新种子组将 `--variants 0 7` 改为 `--variants 3 4 6 7`，增加 `--geometry-seed 20261016` 并使用独立输出目录。报告数据中分别标注了开发地图、新种子同族地图和退步样本；公开录像仅展示16700。

## 复现本次 1000 轮修正

```bash
bash scripts/run_with_rsl5.sh scripts/train_beamdojo_support_transition_v2.py \
  --checkpoint /absolute/path/to/model_15700.pt \
  --teacher /absolute/path/to/model_11700.pt \
  --num_envs 4096 --updates 1000 --terrain_level 8 \
  --phase generalize --flat_fraction 0.5 \
  --run_name support_transition15700_corrective1000 \
  --device cuda:0 --headless
```

这是特定 15700→16700 实验的恢复入口，会检查起点版本、基础网络哈希及归档合同。它不是任意模型通用训练入口。课程初始难度设置为 8，之后按原课程自动升降；1000 为本段新增更新数。

## 测试与适用边界

无需私有检查点的 CPU 测试：

```bash
bash scripts/run_with_rsl5.sh -m pytest -q scripts/tools/test_support_transition_portable.py
```

原 11 项检查点恢复测试保留在 `test_support_transition_stage1.py` 与 `test_support_transition_stage2.py`；通过 `SUPPORT_TRANSITION_11700`、`SUPPORT_TRANSITION_14700`、`SUPPORT_TRANSITION_15700` 环境变量指定本地模型和对应归档，缺失时明确跳过。

当前残差门控面向本体纯前进 0.3–1.0 m/s、前方观察到间隙的状态；站立、后退、转向时通常使用冻结基础动作。门控关闭时相同输入的动作函数保持一致，但不保证整个闭环轨迹逐位相同，临近地图边界也可能触发门控。桩顶等高，仍未验证高低混合桩和连续楼梯；新视频为 Isaac 轨迹重放，MuJoCo 只渲染姿态。
