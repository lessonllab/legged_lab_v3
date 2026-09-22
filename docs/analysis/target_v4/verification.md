# G1 视觉目标点任务 v4

日期：2026-09-11。已完成修改与短测，正式长训练未启动。

## 实现

新增 `LeggedLab-Isaac-AMP-Depth-Target-G1-v0` 与对应 `-Play-v0`，继承 v3 的
10 帧 AMP、镜像参考、深度相机、actor/critic、崎岖地形几何、任务奖励和落脚约束。
实验目录 `g1_amp_depth_target_v4`，旧 Style v3 仍是速度指令任务，可作对照。

目标指令依据本机 InstinctLab 的 `PoseVelocityCommand` 和 parkour 配置适配：

- 每块地形 50 个候选目标；四个检查半径 0.05/0.10/0.15/0.20 米，允许高度差 0.05 米。
- 台阶在局部 X=3.7、Y=0 的末端设目标，与 Instinct 的台阶目标设置一致。
  其他现有地形在局部 XY 各 ±3.7 米范围内采样；相较默认整块采样，留出边界距离。
- 8–12 秒定时重新采样目标、速度上限和 5% 站立指令。
  到达后等到下次定时采样，不立即跳到新目标。
- 起伏路面上限为 0.45–1.0 m/s，台阶、箱体、两类坡面为 0.45–0.8 m/s。
  实际前向指令是身体朝向坐标中的目标前向距离乘 2，限幅到 [0, 采样上限]；横移为 0。
  转向指令为目标朝向误差乘 2，限幅 ±1 rad/s。进入水平 0.4 米范围后全指令归零。
- 课程照 Instinct 改成跟踪指数得分：XY>0.6 且 yaw>0 时升级，XY<0.3 时降级。
  每步指数得分除以完整回合步数后累加；提前摔倒不能靠很短时间内的好跟踪直接升级。
  到点站立仍纳入跟踪得分，这是原方法的行为，并不是越障通过率。
- 保证地形列按比例有序生成，训练/PLAY 都按同一列分配算法解析地形名，
  PLAY 关闭每回合课程更新但保留有序地形布局。
- 重置/课程迁移时依据新地形行列选择世界坐标目标，不重复叠加环境原点，
  并立即更新新回合首个观测中的速度指令。
- 终端显示指令参数，并记录跟踪分数、目标距离、完成目标数、零指令比例。
  完成目标数包含新分配目标已在到达范围内的情况，不应单独用作障碍通过率。
- Viser 对所选机器人显示目标圈、到达半径、目标距离；速度箭头与深度预览保留。

这是现有地形上的指令与课程适配，未复制 Instinct 的专用站立地形、全部障碍参数、
全部奖励、WasabiPPO/MoE 或实机导航系统。目标来源于仿真地形，目标本身不是脚掌落点，
也不提供绕障规划保证。

## 测试

- `unit.log`：25 项 AMP 相关测试通过。新测试覆盖方向旋转、身后目标、前向限速、
  到点停止、站立、世界平移不变性、按地形限速、目标/速度部分重置、课程阈值、
  整回合分数归一化，以及旧 v3 配置不变。
- `train_smoke.log`：512 CUDA 环境，完整 10×20 地形布局及目标采样，
  从头训练 3 轮 PPO+AMP，保存 `model_2.pt`，正常退出。
  `train_result.json`：resume=false，检查点 129 个张量均为有限值。
- `physical.json`：6 CUDA 环境覆盖六种地形，10 行难度、每块 50 个候选点，
  共 3000 点全部位于当前块范围内；逐点对实际导入场景的地形做射线校验。
  候选点存储高度与中心地表高度最大差 0.049607 米：Isaac Lab 存储圆周采样点高度，
  因此允许配置中的 5 厘米差值；控制使用 XY 距离。
- 实际环境中强制 env 0 升级、env 1 降级，验证新目标属于新地形块，其他环境不受影响。
  移动机器人到目标/离开目标，分别验证停止和恢复追踪；随后 40 个动力学步的指令始终有限且在范围内。
- Viser 8082 临时预览确认六类地形标签、红色目标圈、距离文本、速度箭头和深度图。
  预览使用诊断状态，不是训练完成的策略；短测不证明步态已经改善或视觉已被有效利用。

## 复现

从项目根目录运行，使用 `ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh` 前缀：

- 单元检查：`-m unittest discover -s source/legged_lab/test -p 'test_amp_*.py'`
- 实景检查：`scripts/tools/inspect_amp_target.py --viz none --device cuda:0`
- 正式训练命令：见 [当前训练入口](../../g1_depth_instinct.md)。使用 Target v4，从头训练。

参考代码：
[Instinct 目标指令](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/mdp/commands/pose_velocity_command.py)、
[任务参数](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py)、
[课程](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/mdp/curriculums.py)。
适配文件已标注来源及 CC BY-NC 4.0；运行不依赖 InstinctLab 包。
