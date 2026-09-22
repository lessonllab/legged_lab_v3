# G1 EDU 29 自由度：原厂深度相机崎岖地形行走路线

核对日期：2026-09-09。硬件由用户确认为 G1 EDU 29 自由度、相机保持原厂位置。
本文保留最初路线调研。后续已加入单阶段视觉 PPO+AMP 基线并完成短程仿真连通性验证，
详见 [当前实现与训练说明](g1_depth_training.md)。教师蒸馏仍属于后续方案，尚无收敛行走结果。

## 结论与路线选择

以尽快实现功能为目标，优先复现 **Hiking in the Wild 的 InstinctLab parkour 任务**。
它已经有深度图、本体观测、AMP、地形课程和 G1 部署链路。普通 Shadowing 任务不等于这个视觉任务。
官方任务入口是 `Instinct-Parkour-Target-Amp-G1-v0`，见
[parkour 使用文档](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/README.md)。

如果以复用本仓库已训练的 rough 策略为目标，则推荐 **高度图教师 → 深度图学生蒸馏**。
这是对当前项目的工程建议，不是 Hiking 论文的方法复现。Hiking 明确采用单阶段深度视觉强化学习；
高度图教师蒸馏可以参考
[Now You See That](https://hellod035.github.io/Now_You_See_That/)。

平地训练可用于控制和奖励排错，但不是开始视觉崎岖地形训练的必经阶段。
不要把已关闭高度扫描的平地 actor 当成崎岖地形教师。
残差冲突门控创新留在基础视觉行走稳定之后进行。

## 已核验的公开实现

- [Hiking in the Wild](https://project-instinct.github.io/hiking-in-the-wild/)：单阶段深度图和本体状态到关节动作，采用地形边缘及足部体积点约束、可行目标采样；项目页链接了代码与模型数据。
- [InstinctLab](https://github.com/project-instinct/InstinctLab)：当前说明使用 Isaac Sim 5.1.0、Isaac Lab 指定提交 `f73c331738` 和 instinct_rl。与本仓库的 Isaac Lab 3.x / RSL-RL 栈不同，应使用独立环境；本文没有安装或替换依赖。
- [instinct_onboard](https://github.com/project-instinct/instinct_onboard)：说明已测试 G1 29DoF、Jetson Orin NX、Ubuntu 22.04、ROS2 Humble。可参考 ONNX 部署和观测流水线，不可直接混用本项目 checkpoint。
- [Now You See That 仓库](https://github.com/Hellod035/Now_You_See_That)：检查时根目录可见 README 和 assets，没有确认可运行的训练代码。适合方法参考，不应当作已可直接运行的训练框架。
- [Perceptive Humanoid Parkour](https://php-parkour.github.io/)：专家策略通过 DAgger 与 RL 合成深度视觉学生，G1 实验包含多技能越障；项目页仍标注 Code Coming Soon。普通崎岖行走不必先复现完整跑酷技能体系。

## 原厂相机：必须保持实机视野

相机保持 G1 原厂头部安装位置；不要套用向下俯视、可见机器人四周地面的理想高度扫描器模型。
原厂资料入口：[Unitree G1 手册](https://marketing.unitree.com/article/en/G1/User_Manual.html)。
实际设备型号、启用的深度流内参及深度单位，在连接机器时读取确认。

Instinct 的 [parkour 相机配置](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py)
标注 G1 head camera nominal pose：相对于其 `torso_link`，平移约为 `(0.0488, 0.0100, 0.4378) m`；
原始图像配置 `64×36`，水平/垂直视野约 `89.51°/58.29°`，仿真更新周期 `0.02 s`。
配置还包含裁剪、模糊、深度归一化及多帧历史。它们是该实现的参数，不是所有 G1 相机流的固定规格。

不能直接照搬其四元数：旧 Isaac Lab 常用 `wxyz`，本机新 camera 配置声明 `xyzw`。
还必须核对 torso 原点、固定关节合并和 world/ROS optical 轴约定。
相机应跟随机身完整姿态变化；需要通过画面验证俯仰、横滚、转身后的可见地形。
标称相机姿态不能替代对当前 USD 和实机图像的核验。

## 在 legged_lab_v3 中的推荐实现

### A. 先验收崎岖地形教师

复用 `G1AmpRoughEnvCfg` 的地形、接触、动作和高度扫描。固定一份 teacher checkpoint 与其 env/agent 配置。
先分开评估缓坡、低台阶、凹凸地面，记录通过率、跌倒率、速度误差、滑移和越界率。
教师在目标地形上可靠之后再蒸馏；仅看总 reward 不能判断是否已学会通过地形。

低速范围可以先试 `0.3～0.8 m/s`，属于建议起点而非已验证参数。暂不混入大间隙、高台和高速跑。
地图和训练速度应配合原厂相机向前的可见范围，避免把侧移、后退中不可见的落脚区域当成必然可观测。

### B. 深度传感器和观测

新增独立的 rough-depth 任务，保留教师所需的高度扫描，但不把它交给学生。
第一版静态地面可用 `RayCasterCameraCfg`，输出 `distance_to_image_plane`。
[Isaac Lab 官方实现](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/sensors/ray_caster/ray_caster_camera.html)
区分相机平面深度与到光心的欧氏距离，且该实现只支持静态 mesh；只投射地形会漏掉身体遮挡。
需要身体遮挡时使用支持动态网格的相机实现，或检查渲染相机路径。

观测分成三个接口：

- `teacher_policy`：严格复现 checkpoint 的本体历史、指令和高度扫描，顺序与归一化不变。
- `student_proprio`：实机可获得的 IMU、关节状态、指令、上一步动作。
- `student_depth`：按相机时间戳排列的深度图历史及必要的有效性信息。

处理流水线：米制深度 → 无效值处理 → 裁剪/缩放 → 范围归一化 → 历史缓存。
训练和部署必须相同；无效深度与真实近距离不能不加区分地混为同一数值。
学生不得读取 ground-truth 高度扫描、仿真 root 位置、真实线速度或地形编号。
当前 actor 无线速度输入；如果以后加入速度估计，部署必须包含同一估计器。

### C. 网络与训练

```text
训练：
本体状态 + 指令 + 理想高度扫描 → 冻结教师 → 教师动作标签
本体历史 + 深度图历史 → CNN + 时序模块 → 学生动作 → 仿真环境
                                                  ↓
                              在学生访问的状态上继续查询教师

部署：
原厂深度相机 + IMU/关节状态 + 指令 → 相同学生 → 关节目标
```

采用学生 rollout 的在线行为克隆/DAgger，而不只收集教师成功轨迹后离线拟合。
基础损失为归一化动作空间的 `MSE(student_action, teacher_action)`；首先确认动作复制能闭环行走。
可选辅助任务是预测局部地形或教师 latent，但原厂相机看不到的区域不能当作无噪声监督。
不用强制生成完整高度图也能输出动作。

历史可以使用帧堆叠或 GRU。相机低频异步更新、动作高频执行时，要在仿真复现帧保持、延迟和掉帧；
不能以仿真 `50 Hz` 更新参数推断实机相机也输出 `50 Hz`。动作控制当前是 `50 Hz`。
初始图像分辨率建议从 `64×36` 或 `80×48` 试起，基于显存和闭环结果调整。

第二阶段再逐步加入孔洞、距离相关噪声、模糊、深度比例误差、相机外参扰动及延迟随机化。
加入 AMP/PPO 微调是后续步骤；纯蒸馏阶段不需要重新训练判别器。

### D. 当前代码接入点与缺项

- 环境：在 `tasks/locomotion/amp/config/g1/` 新增深度环境配置和注册，避免改变已有 rough checkpoint 的输入维度。
- 观测：新增深度预处理、缓存及按环境 reset 的实现，使用单独的图像组，不把图像直接展平成 MLP 本体观测。
- 策略：当前 `PPOAMP.construct_algorithm` 可解析自定义 actor 类，但默认仍为 MLP；CNN/时序模型不是加一个 camera 配置就会自动启用。
- 训练：`scripts/rsl_rl/train.py` 有 `DistillationRunner` 分支，但仍需要学生/教师配置、AMP checkpoint 的 actor 加载适配和冻结归一化。
- 对称增强：现有 G1 对称函数只处理本体与高度扫描。视觉分支必须同步镜像图像和时序，或在初版学生训练中关闭这一增强；若相机安装偏离中线，还需评估镜像假设。
- 导出：除 actor 外，需导出编码器、时序状态初始化和预处理参数；逐帧比对训练端与 ONNX 动作输出。
- 部署：关节顺序、默认姿态、动作尺度、控制周期、PD 增益必须匹配。官方 parkour 使用不同资产/执行器配置，含鞋版本也与本项目不同，不能直接加载其网络到本项目机器人。

### E. 验收顺序

1. 单环境相机画面检查：平地、台阶、转身、弯腰；核对朝向、深度数值、遮挡和足前盲区。
2. 小批量仿真：缓存 reset 正确，观测无 NaN/Inf，不串环境，不跨 episode 混帧。
3. 教师基准 → 学生在线蒸馏 → 留出地形闭环评估。失败和超时都计入，不能只统计成功回合。
4. 正常深度、冻结深度、打乱深度的对照，确认地形通过能力确实依赖视觉。
5. 注入延迟/掉帧后再评估，随后做独立仿真器验证和实机相机回放；最后进行低速实机测试。

## 当前状态和前置问题

本轮完成平地直接速度模式、关闭推扰、训练/Play 速度范围对齐，以及参考重置 yaw 限幅。
命令回归测试 3 项通过，修改文件语法检查和 `git diff --check` 通过。

后续视觉接入已修复本机 Isaac Lab 的 velocity task 命名空间和 sim_launcher 入口兼容，
真实仿真已完成短程训练；没有更换本机依赖。验证范围见当前实现说明。

优先级：若现有 InstinctLab 环境可用，先确认运行的是官方 parkour 视觉任务并复现；
若坚持在 legged_lab_v3 中开发，先处理版本兼容，再验收 rough 教师和相机观测，之后才蒸馏。
