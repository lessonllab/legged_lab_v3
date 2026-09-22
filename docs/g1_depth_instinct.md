# G1：参考 InstinctLab 的视觉输入改造

## 当前入口：v6 风格与目标奖励配套修复

2026-09-13：新训练使用 **`Target-G1-v2`**，代码配置版本为 v6，输出到独立目录
`logs/rsl_rl/g1_amp_depth_target_v6/`。前向深度视觉、六类崎岖地形、参考动作镜像、
有效进度课程和足底支撑约束继续启用。

实际合成奖励改为 `r_task + 0.25*f(D)`，其中任务奖励由环境按控制时间步积分，
风格奖励不再额外乘时间步。配套采用双类样本梯度惩罚、显式参数正则及当前 rollout
的判别器更新，修复纯转向时的单脚支撑奖励，并加入全关节站立、骨盆/躯干姿态约束。
详见 [v6 修复与验证记录](analysis/target_v6/verification.md)。

```bash
cd /home/ljc/legged_lab_v3
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh \
  scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-Target-G1-v2 \
  --viz none --device cuda:0 --num_envs 512 \
  --max_iterations 30000 --seed 42 --run_name target_v6_scratch \
  agent.device=cuda:0 agent.resume=false
```

这是从头开始的视觉崎岖地形训练，初始等级 0，每 500 轮保存一次。
不要加载 v5 或短测 checkpoint；新版判别器的归一化与优化器状态不同。
30000 轮是训练预算，步态质量需要固定地形和指令的播放验收。
终端启动摘要以 `[Target v6]` 开头，并打印实际 AMP 配置。

TensorBoard 的 `Train/mean_reward` 仍是任务回报。总奖励及风格作用应同时查看
`AMP/mixed_reward_per_step`、`AMP/weighted_task_reward_per_step`、
`AMP/weighted_style_reward_per_step`、`AMP/raw_style_reward_per_step` 与风格零奖励比例。
播放入口为 `LeggedLab-Isaac-AMP-Depth-Target-G1-Play-v2`。

## 可选：AME 交替梅花桩（加大桩面、缩短间距）

2026-09-13：训练或播放命令加 `--stepping_stones` 可加入第七类地形。
当前使用本地 AME 的 G1 PLAY 实际启用的 `HfAlternateColumnStakesTerrainCfg`，
对应 `velocity_env_cfg_29dof.py:646`，生成函数为 `alternate_column_stakes_terrain`。
此前自行设计的双列、随机大小和间距版本已被替换。按用户后续要求，当前桩面改为 0.4 米、纵向间隙改为 0.1 米。

- 中央 2×2 米方形出生平台，四向通道的桩按左右交替方式排列，周围下沉 2 米。
- 当前配置：桩边长 0.4 米，纵向间隙 0.1 米，交替横向参数保持 0.3 米；
  高度扰动和位置随机抖动均为 0，外围边界配置宽度 0.25 米。
- 使用原函数和 0.05/0.005 米的水平/竖直网格精度，保留其整数网格取整行为。
  上述是配置名义值，实际几何按 AME 原算法离散化，未自行修正间隙或补桩。
- 这是基于 AME PLAY 调整桩面与间距后的固定难度参数，不随地形等级变化。AME 训练中的 stakes1/2/3
  是另外几套参数，不能与这个播放地形混称为完全相同的训练设置。
- 本地任务仍保留目标控制：自动目标在 +X 通道尽头的外围道路，速度上限 0.25–0.55 m/s；
  点击交互仍可指定目标。掉落终止以中央平台为高度基准。没有复制 AME 的策略或奖励。

当前训练不会动态增加地形；以后启动训练时加该参数，输出自动使用独立的 `_stones`
实验目录（v6 为 `g1_amp_depth_target_v6_stones`）。现有六地形训练入口保持原样。

播放时在现有命令加 `--stepping_stones --instinct_play --follow_env 6`，会生成七个机器人，
默认选中“6 · 梅花桩”。当前七列展示布局属于本项目，单块地形保留 AME 生成算法；桩面与间距已按要求调大、调近。

验证：62 项 AMP 回归测试通过；从 AME PLAY 源码抽取配置，检查除了这两项用户指定参数外其余参数保持一致；
两组种子×三个难度下，原函数与移植函数生成的顶点、三角面和出生原点逐元素一致。
实际生成十行地形的 500 个目标全部位于外围道路，独立射线误差小于 8 微米。
中央平台、外围道路、空区及七机器人 PhysX/Viser 播放通过检查。
记录见 [地形与目标检查](analysis/stepping_stones_20260913/terrain.json)。

## 历史入口：v5 视觉目标任务

2026-09-12：v4 已停训。新训练请使用 **`Target-G1-v1`**，对应独立实验目录
`logs/rsl_rl/g1_amp_depth_target_v5/`。保留深度视觉、六类崎岖地形、参考动作及足底支撑约束。
奖励、升级判定、初始化与探索设置的变更见 [v5 修复与验证记录](analysis/target_v5/verification.md)。

```bash
cd /home/ljc/legged_lab_v3
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh \
  scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-Target-G1-v1 \
  --viz none --device cuda:0 --num_envs 1024 \
  --max_iterations 30000 --seed 42 --run_name target_v5_scratch \
  agent.device=cuda:0 agent.resume=false
```

从地形等级 0 起步，满足真实推进/到达、有效跟踪和无失败超时条件才升级。
默认每 500 轮保存模型。30000 是本次训练预算，不是达到步态质量的保证。
终端启动时打印 `[Target v5]` 配置；TensorBoard 新增有效目标到达、净推进距离、
有效移动时长及移动期间的线/角速度跟踪指标，避免用地形平均等级代替通过能力。

播放新模型时用 `LeggedLab-Isaac-AMP-Depth-Target-G1-Play-v1`，支持 `--instinct_play`。
该参数展示 4 行地形，可能超出模型已学会的难度；定量验收应固定难度和随机种子。

## 历史入口：v4 目标点指令（保留用于旧模型排查）

2026-09-11 按用户要求新增 Instinct 式目标点指令。使用下面的 **Target** 任务；
下文 **Style v3** 入口保留作速度指令对照。

```bash
cd /home/ljc/legged_lab_v3
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh \
  scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-Target-G1-v0 \
  --viz none --device cuda:0 --num_envs 512 \
  --max_iterations 80000 --seed 42 --run_name target_v4_scratch \
  agent.resume=false
```

- 每块地形生成 50 个候选目标，8–12 秒重新采样目标与速度上限。
  台阶使用距中心 X=3.7 米、Y=0 的末端目标；其他地形在块内采样平整小片区。
- 起伏路面前进速度上限采样 0.45–1.0 m/s；台阶、箱体、坡面采样 0.45–0.8 m/s。
  这是上限，实际指令可为 0。横移为 0，朝向误差乘 2 后限幅 ±1 rad/s；
  前向位置误差乘 2 后限制在 [0, 采样上限]，目标在身后时先转向。
- 目标水平距离 ≤0.4 米时停下；5% 的指令采样要求站立。到达后等待下一次定时采样。
- 课程使用整回合累计的速度跟踪指数得分，XY>0.6 且 yaw>0 时升级，XY<0.3 时降级。
  得分除以完整回合步数，过早摔倒会拉低得分。到点停止后不会再被原来的位移课程判定为没走够远。
- 终端记录目标距离、完成目标数、零指令比例及课程分数；Viser 给所选机器人显示红色目标圈。
  目标只是行进目的地，不是脚掌落点；目标数和地形等级不能单独证明越障成功。

输出目录为 `logs/rsl_rl/g1_amp_depth_target_v4/`。保留 v3 的深度相机、10 帧 AMP、
参考镜像、崎岖地形几何和落脚约束。此处适配的是目标指令与课程，未复制 Instinct 的全部
跑酷地形、专用站立地形、完整奖励系统或网络结构。当前训练目标来自仿真地形，实机需另行提供目标/速度。

新模型重放使用 `LeggedLab-Isaac-AMP-Depth-Target-G1-Play-v0`，例如：

```bash
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh \
  scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Depth-Target-G1-Play-v0 \
  --viz viser --terrain_showcase --device cpu --real-time \
  --checkpoint /absolute/path/to/new/model_1000.pt agent.device=cpu
```

检查结果见 [目标指令验证记录](analysis/target_v4/verification.md)。

## 2026-09-11：风格 v3，从头训练入口

针对 v2 步态不自然，新增独立崎岖任务 `LeggedLab-Isaac-AMP-Depth-Style-G1-v0`。
风格判别器使用 10 帧（时间跨度 0.18 秒），每帧 67 维：身体坐标中的重力方向、
相对默认关节角、缩放后的关节速度、根节点线速度、角速度。参考动作与机器人使用同一编码。
根节点线速度取 link 原点速度，与参考位置差分的定义一致，避免混入质心偏移。
参考轨迹每回合以 50% 概率左右镜像，交换同名左右关节并修正轴符号，保持时间顺序。
这是参考动作分布增强，不是强制左右脚同时/完全对称，也没有直接翻转相机图像。

按用户要求直接从头训练崎岖地形，保留前向深度视觉、地形课程和落脚边缘约束：

```bash
cd /home/ljc/legged_lab_v3
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh \
  scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-Style-G1-v0 \
  --viz none --device cuda:0 --num_envs 512 \
  --max_iterations 80000 --seed 42 --run_name style_v3_scratch \
  agent.resume=false
```

输出在 `logs/rsl_rl/g1_amp_depth_style_v3/`，不加载旧 checkpoint，也不要加载短测模型。
v3 判别器输入由 244 维变成 670 维，不能完整恢复 v2 训练状态。
旧 v2 任务与模型仍可重放；当前浏览器播放的旧模型不会因改配置自动学会新步态。
新版没有移植 InstinctLab 的完整 WasabiPPO、MoE 和全部奖励；步态是否改善需要正式训练后验证。
测试记录见 [v3 验证报告](analysis/style_v3/verification.md)。

下面保留 v2 的实现说明与旧入口，便于复现比较。

2026-09-11 已修复身体网格与环境对应错误，以及名义光心被封闭头壳遮住的问题。
Instinct 任务使用修正版配置，实验目录加 `_v2`，旧 checkpoint 文件保留。
新版有崎岖地形和可选平地预训练两种任务。

## 改了什么

- 使用 Isaac Lab 原生 `MultiMeshRayCasterCamera`，相机同时检测地形和当前环境中 G1 的
  30 个刚体对应的 31 个视觉目标（另加 1 个地形目标）。每个目标只跟踪一个刚体，
  避免当前 PhysX 将多个身体按身体优先排列、射线相机却按环境优先解释而导致串位。
  只从深度射线中排除没有镜头开孔的 `head_link` 外壳，保留 torso、logo、四肢；
  浏览器外形与物理碰撞均不变。
- 相机仍安装在 torso_link 上的名义头部位置，面向前下方；不是全向视觉。
  使用 Instinct 的完整名义位姿，转换并归一化为本环境要求的 XYZW 四元数。
- 原始图像 64×36、名义视场 89.51°×58.29°，50 Hz 更新。
  裁去顶部 18 行、左右各 16 列，实际输入图像为 18×32，有效视野小于原始视野。
  应用 3×3、sigma=1 的高斯模糊，深度归一化到 [0,1]。
- 每个环境保存 37 帧，抽取偏移 [35,30,25,20,15,10,5,0] 的 8 帧；
  每回合随机增加 0 或 1 帧延迟，即最旧帧距当前约 0.70–0.72 秒。
  同一步重复读取不推进历史；部分环境重置只清除自身历史，用新回合首帧填充。
- actor 和 critic 分别使用独立 CNN，读取同一组延迟图像；两者都不再接收理想高度图。
  critic 仍保留本项目的特权运动状态。实际观测为 policy=495、critic=600、depth=8×18×32。
  高度扫描器仅保留给仿真中的离地高度摔倒检测使用，不作为网络输入。
- 保留之前的修复：动作标准差固定 0.5、熵系数 0、10 轮 AMP 回放及当前/历史混采、
  减轻扭矩惩罚、降低离地奖励、添加不前进惩罚。
- 增加承重脚的支撑不足与台阶边缘代价（权重 -0.25、-0.10）。每只脚使用 27 个
  脚底采样点及外围采样圈，只检测地形；两只脚共 110 条射线。
  接触不足 80 ms 或向上的法向力不超过 20 N 时不施加这两项惩罚。
  这些采样只用于训练奖励，不输入 actor/critic，也不改变深度图。
  细节及台阶验证见 [落脚约束报告](analysis/foot_support_20260911/verification.md)。

参考配置与实现：
[Instinct 相机与图像处理](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py)、
[G1 观测与身体遮挡](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py)。
同时核对了本机 `/home/ljc/InstinctLab` 的对应源码。

这是参考其视觉链路的适配，尚未移植完整的目标点命令、跑酷障碍课程、奖励系统、
本体历史长度和网络结构；不能当作 Instinct 论文的完整复现。
本实现显式将射线最大距离设为 2.5 m，并在模糊前限制深度，避免无命中值污染邻近像素；
正深度小于 0.1 m 时填为最远值。原配置的深度归一化在模糊之后，边界像素可能不同。
0/1 帧延迟使用离散均匀采样。上述是明确的适配选择。
名义外参不是你这台机器的实测标定；当前工作仍是仿真，尚未接入实机深度流。

## 直接开始崎岖地形训练

以下启动的是**崎岖地形任务**，初始地形等级为 0，保留现有地形课程；不是平地任务。
128 个环境是保守起点，可根据速度和显存再增加。

```bash
source /home/ljc/isaaclab/bin/activate
cd /home/ljc/legged_lab_v3
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Depth-Instinct-G1-v0 --viz none \
  --num_envs 128 --max_iterations 10000 --run_name foot_support_v2
```

日志：`logs/rsl_rl/g1_amp_depth_instinct_v2/`。每 500 轮保存一次。
相较之前的 Instinct v1，网络尺寸相同，但深度输入含义已经改变。旧模型可能加载成功，
却不能代表修复后的训练效果；以上命令从头训练，不添加旧 checkpoint 的 resume 参数。
本次只做有限轮数验证，没有启动长期训练。

播放新版崎岖地形 checkpoint：

```bash
bash scripts/run_with_rsl5.sh scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-Depth-Instinct-G1-Play-v0 --viz viser --num_envs 4 --follow_cam --real-time \
  --checkpoint /absolute/path/to/model_1000.pt
```

可选平地预训练任务：`LeggedLab-Isaac-AMP-Depth-Instinct-Warmup-G1-v0`，
播放任务在 `G1` 后加 `-Play`。与新版崎岖任务的网络输入相同，命令固定 0.5 m/s，
关闭地形课程与外力随机化；日志目录为 `logs/rsl_rl/g1_amp_depth_instinct_warmup_v2/`。
选择此任务就是平地，不会自行升级到崎岖地形。
修正版用有限尺寸的纯平网格替代无限平面：当前后端将无限平面三角化至 ±100 万米，
测试中造成约 0.148 m 的跨环境深度误差；有限平地空场景误差小于 3 微米。
配置中的 `generator` 表示生成网格，不表示存在崎岖障碍。

### 当前环境的播放兼容处理

本机 Viser 初始化直接依赖 NewtonManager，而安装的 Newton 1.2.1 缺少适配代码要求的
`ModelFlags`；即使补齐该名称，Viser 仍需要 Newton 模型和状态。此任务使用 PhysX，
现在项目通过 `scripts/rsl_rl/physx_viser.py` 直接连接 Viser：USD 提供地形和身体视觉网格，
PhysX 的实时身体位姿驱动浏览器模型运动，不加载 Newton，也不再自动切换到 Kit。
策略、深度相机、碰撞和物理步长保持原样；浏览器显示读取仿真状态，PLAY 的点击交互
通过单独队列更新目标指令。

默认地址为 `http://localhost:8080`，页面提供暂停、机器人选择、相机跟随和策略深度输入预览。
目标点任务的 Viser PLAY 还支持单击地形设置目标（2026-09-13）：

- 先在“观察机器人”选择机器人，再单击地形；按射线与真实地形网格的最近交点设置世界目标。
  台阶与坡面保留实际高度，红圈标注所选机器人以及手动/随机模式。
- 只有当前选中的机器人可以保留手动目标，其他机器人继续原来的随机目标采样。
  切换机器人时释放原来的手动目标；“当前机器人恢复随机目标”按钮也可主动释放。
- 手动目标不会被 8–12 秒的指令采样覆盖，到达后停下等待新目标。
  摔倒或回合超时重置会释放手动目标，避免把旧目标带到新的出生位置。
- 暂停时也能点击，恢复播放后继续执行。拖动仍用于旋转视角；关闭“点击地形设置目标”
  可恢复纯观察。点击空白且射线未命中地形时保留原目标。

点击回调仅排队，仿真主线程写入目标，策略观察历史按正常物理步更新。
这项交互默认不影响训练，参考动作播放也不启用它。
验证通过：58 项 AMP 回归测试（含地形射线、选中对象隔离、定时采样、到达停止、
切换/释放及部分重置），以及真实 6 机器人 Viser 的点击与切换操作。

使用 `--viser_port 8081` 可更换端口；如从其他设备访问，加 `--viser_host 0.0.0.0`，
并在浏览器输入仿真电脑的 IP 和实际端口。若端口占用，Viser 可能另选端口，以终端地址为准。
深度预览显示所选机器人最新一张策略输入（含裁剪、模糊和延迟），不是 RGB 视频。
策略播放中，所选机器人上方显示绿色目标速度箭头和橙色实际速度箭头，
二者均在世界水平面中显示，长度 0.6 m 对应速度 1 m/s；零速度隐藏对应箭头。
右侧显示前向/侧向目标速度、转向指令及实际平移速度，可关闭速度箭头。
这些标记只存在于 Viser 中，不会成为深度相机的遮挡物或改变策略输入。
几何外形与实时姿态来自实际模型，材质使用统一颜色，暂不复现 USD 的完整纹理效果。
推荐 `--num_envs 4 --real-time`，姿态以每两个控制步一次的频率发送。
Kit 本地窗口仍可通过 `--viz kit` 使用，`--viz none` 和 Newton 物理任务保持原行为。

播放命令加 `--terrain_showcase` 可按地形类型各放置一个机器人（当前共 6 个），
使用同一模型、固定生成难度 0.25。这个播放选项会覆盖 num_envs，给每种地形一个独立列，
不会更改训练配置。右侧选择菜单标注地形名称，选择后近距离跟随；
“全部机器人总览”恢复全景。速度箭头和深度预览对应当前选中的机器人。

浏览器适配验证日志：`/tmp/g1_physx_viser.log`。
已实际启动 4 环境的 `model_10000.pt` 播放，并通过浏览器连接，页面正确提供暂停、跟随、
机器人选择和深度预览；服务端加载 30 个身体视觉部件/机器人并持续发送实时姿态。
另以带旋转和平移的 USD 测例核对网格的身体局部坐标与世界坐标转换。
新增 `--max_steps 100` 可用于有限步数的播放检查；正常观看不传此参数。

## 检查视觉与行走能力

### 重放训练实际使用的参考动作

`replay_amp_reference.py` 从训练保存的 env.yaml 读取动作目录及权重，直接显示参考姿态，
无需 checkpoint。默认 CPU、单机器人、Viser 8081，不占用策略播放的 8080 端口。

```bash
cd /home/ljc/legged_lab_v3
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh \
  scripts/tools/replay_amp_reference.py \
  --run_dir logs/rsl_rl/g1_amp_depth_instinct_v2/2026-09-11_13-01-11_foot_support_v2 \
  --motion C4_-_run_to_walk_a_stageii
```

打开 http://127.0.0.1:8081/。该片段为跑步转走路，从头重放。
去掉 `--motion ...` 则按保存的训练权重抽取片段；用 `--list_motions` 列出可选名称。
本次训练的参考集混合了走、跑、转弯、侧移和倒退动作，并非纯直行。
原 Animation 配置默认指向 deepmimic 目录，不等于本次训练的 amp/walk_and_run。
入口已完成单片段 60 步运行验证。此重放用于核对参考动作本身，不是策略平衡或地形通过测试。

### 检查策略输入与直行

```bash
bash scripts/run_with_rsl5.sh scripts/tools/inspect_amp_depth.py \
  --task LeggedLab-Isaac-AMP-Depth-Instinct-G1-Play-v0 --viz none \
  --num_envs 4 --steps 40 --output_dir .runtime/g1_depth_instinct_check

bash scripts/run_with_rsl5.sh scripts/tools/verify_amp_depth_progress.py \
  --task LeggedLab-Isaac-AMP-Depth-Instinct-G1-v0 --viz none \
  --checkpoint /absolute/path/to/model_1000.pt \
  --num_envs 64 --seed 42 --output /tmp/instinct_eval_seed42.json
```

直行检查固定速度 0.5 m/s、零转向，在自动重置前记录首回合终点；另用 seed=43 复测。
总奖励上升、地形等级变化或网络存在 CNN 均不能单独证明策略依赖视觉。
行走达标后还需比较正常深度与打乱/遮蔽深度的通过率，才能判断视觉是否真正帮助越障。

## 历史流程验证（2026-09-10，不能证明遮挡正确）

下列旧检查只确认了网格数量和运动，没有确认每个网格是否跟对刚体。
2026-09-11 的逐目标位姿和对称姿态检查揭示了错误，因此不应据此认定旧视觉正确。
当前修复与验证见 [修复报告](analysis/depth_fix_20260911/verification.md)。

- 15 项 CPU 回归测试通过，包含精确历史抽帧/延迟、重复读取、部分重置、近距处理、
  actor/critic CNN 梯度与 JIT 导出，以及已有深度、速度命令和 AMP 回放测试。
- 4 环境崎岖地形相机运行 40 步，读到 30 个机器人身体网格和 1 个地形网格，
  命中网格编号包含非地形项，历史重置通过。
- 16 环境崎岖任务完成 3 轮真实 PPO+AMP 更新并保存 `model_2.pt`，
  critic 确认为 CNNModel，动作标准差保持 0.5。
- 4 环境新版平地相机运行 40 步，身体命中像素占比约 1.76%，跟踪网格位置确实更新；
  从上述新版崎岖 checkpoint 恢复到新版平地任务，再完成 2 轮更新，标准差仍为 0.5。
  这只验证两个新版任务的网络兼容性，不是推荐用未收敛的短测模型继续正式训练。

本次日志位于 `/tmp/g1_depth_instinct_alltests.log`、`/tmp/g1_depth_instinct_train.log`、
`/tmp/g1_depth_instinct_resume.log`；相机检查输出位于项目的
`.runtime/g1_depth_instinct_check/` 和 `.runtime/g1_depth_instinct_warmup_check/`。

以上是流程验证，不是训练收敛或崎岖路面通过率验收。

## Instinct 式重放视角与初始化（2026-09-12）

新增 `--instinct_play`，仅适用于 Target 任务，与 `--terrain_showcase` 互斥：

```bash
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python bash scripts/run_with_rsl5.sh \
  scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Depth-Target-G1-Play-v0 \
  --viz viser --instinct_play --device cuda:0 --real-time \
  --checkpoint /home/ljc/legged_lab_v3/logs/rsl_rl/g1_amp_depth_target_v4/2026-09-11_19-20-02_target_v4_scratch/model_32500.pt \
  agent.device=cuda:0
```

参考本机 Instinct 的 PLAY：默认关节站姿、关节速度为零，根位置 XY/朝向 ±0.1，
根速度各分量 ±0.2；10 秒回合；相机从根节点偏移 [4,0.75,1] 近景跟随。
当前适配保留六类现有地形，生成四行难度，各机器人初始选一行，关闭回合间难度课程。
候选目标和速度控制保持原方法。保留本项目摔倒终止（没有照抄 Instinct 关闭高度终止的部分）。
右侧显示地形行、回合时间、时限结束与提前终止次数；时限结束不等于到达目标或通过障碍。

这是独立的站姿初始化测试，训练仍使用参考动作初始化，二者的回合统计不能直接比较。
旧 `--terrain_showcase` 仍保留固定 0.25 难度与原初始化，便于对照。

验证：Python 编译与 diff 检查通过，32500 轮模型在 GPU PhysX 成功加载，
六机器人、四行地形以及新的根/关节重置项已在运行日志确认；浏览器近景、目标距离、
回合计时显示正常。日志 `/tmp/g1_instinct_play.log`。这不是训练步态改善证明。
