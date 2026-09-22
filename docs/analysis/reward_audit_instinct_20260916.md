# 当前楼梯奖励结构与 Instinct 对照审计

日期：2026-09-16。范围：`LeggedLab-Isaac-AMP-Stairs-Long-G1-v0`，当前代码的 v8 奖励与暂停前正式训练配置；参照本机 InstinctLab 的 G1 parkour 源码快照，不声称是远端最新版本或论文全部复现。

**本次只分析、导出配置和读取日志，没有更改奖励，也没有启动训练。**

## 1. 结论

现有结构是“Instinct 风格任务/AMP奖励 + 多代本地修改 + 楼梯专用虚拟边缘与防撞扩展”，不是完整 Instinct 配置。方向合理，但当前最重要的缺口是：

1. 虚拟边缘惩罚乘速度，不能独立限制静止承重踩边；移除旧支撑约束后，尚未补上完整的落脚接触约束。
2. 防脚尖撞立面项可以减少低脚尖向前冲，却没有直接奖励足够的越阶净空与可靠落脚；降低速度也能降低该成本。
3. 课程升级仍只看穿越、未摔倒和巡航速度，没有安全落脚/碰撞的门槛。
4. AMP 的系数是所参照 Instinct 的两倍；实际日志中它占每步净奖励约 77%，值得对照实验，但不能由此断定它占同样比例的梯度或就是主要故障原因。
5. 足滑、承重脚姿态、贴地、关节软约束和全身接触范围与 Instinct 存在明显差异，部分差异是此前继承链留下的，而不是此次刻意设计。

优先修复检测与目标定义，再做小范围权重消融；不建议直接把所有惩罚翻倍。

## 2. 真正进入 PPO 的总奖励

控制周期 dt=0.02 s（50 Hz），物理步长 0.005 s、decimation=4。

`r_task = 0.02 × Σ(weight_i × term_i)`

`r_style = 0.5 × max(1 - 0.25 × (D(s)-1)^2, 0)`

`r_total = r_task + r_style`

- 当前启用 additive，**不是**旧式 task/style lerp。
- 当前 AMP 不乘 dt，style gate 已关闭；不要因为还存在 gate 的源码就认定它正在工作。
- 本机 Instinct 同样是环境奖励乘 dt、AMP 辅助奖励不乘 dt，但 AMP 系数为 0.25。因此这不是误少乘一次 dt 的程序错误，而是任务与风格尺度的设计差异。直接给 AMP 补 dt 会让当前风格强度骤减 50 倍，不是合理的首选修复。
- 当前水平跟踪、转向跟踪、存活三项满分共 +0.16/步；AMP 上限 +0.5/步，实际贡献要看判别器分数。
- 当前失败一次的显式成本为 -4，而不是 -200。失败还会失去后续回报，不能只用 -4 判断摔倒是否“划算”。
- gamma=0.99，对应约 100 控制步/2 s 的折扣时间尺度；不是硬截断。17 级楼梯全过程通常比这个时间尺度长，因此应保留合理密集引导，不能只依赖终点奖励。

来源：[当前 AMP 配置](/home/ljc/legged_lab_v3/logs/rsl_rl/g1_amp_stairs_long/2026-09-16_17-03-41_instinct_edges_toe_v8/params/agent.yaml:94)、[奖励合成](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/rsl_rl/amp/discriminator.py:157)、[Instinct AMP 配置](/home/ljc/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:42)、[Instinct 奖励管理器](/home/ljc/InstinctLab/source/instinctlab/instinctlab/managers/reward_manager.py:120)。

## 3. 当前全部 21 项生效奖励

下面是配置实例完成全部继承后的结果，不是仅抄父类默认值。数值都是未乘 dt 的配置权重。

### 1. track_lin_vel_xy_exp：3

水平速度跟踪：exp(-水平速度误差平方/0.25)。使用去除俯仰/横滚的航向坐标系以及 root_link 速度；Instinct 基础函数使用机体坐标系 root 速度。Instinct 权重 +2，我们 +3。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_scratch_env_cfg.py:31)


### 2. track_ang_vel_z_exp：2

转速跟踪：exp(-偏航角速度误差平方/0.25)。我们取世界 Z 角速度，Instinct 基础函数取机体 Z；权重均 +2。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_scratch_env_cfg.py:36)


### 3. lin_vel_z_l2：-0.2

机体竖直速度平方。Instinct 所核对的 G1Rewards 未启用这一项；爬高、上下阶会有合理的竖直运动，存在抑制必要运动的可能。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:78)


### 4. ang_vel_xy_l2：-0.05

机体横滚/俯仰角速度平方，Instinct 同权重。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:85)


### 5. dof_torques_l2：-1.5e-07

髋、膝、踝实际关节力矩平方和；Instinct 同权重、同类关节范围。很小的数字不能直接理解为无效，因为力矩需要平方。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:138)


### 6. dof_acc_l2：-1.25e-07

当前只选髋和膝关节的加速度平方；Instinct 对全部关节使用同权重。当前踝关节和上肢不在此项内。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:169)


### 7. action_rate_l2：-0.005

相邻控制步动作差平方和，Instinct 同权重。动作单位、动作缩放和控制周期会影响效果，不能脱离机器人设置比较。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:258)


### 8. dof_pos_limits：-1

当前只惩罚踝 pitch/roll 的软关节限位超出；Instinct 对全部关节计算。这不等于我们的物理硬限位不存在。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:192)


### 9. feet_air_time：0.5

恰好单脚支撑时，奖励支撑/摆动计时的较小值；平移或转向指令超过 0.15 才生效。与 Instinct 的实现和权重一致，源码没有显式时长上限。它鼓励单支撑节律，并不直接奖励脚尖抬高。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py:45)


### 10. feet_slide：-0.1

承重脚水平平移速度范数之和，当前权重 -0.1，Instinct -0.4。当前接触门控来自法向合力历史，Instinct 来自总接触力历史；常规承重时接近，但严格说并非完全同一信号。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/rewards.py:71)


### 11. undesired_arm_contacts：-1

只覆盖肘、腕非期望接触，阈值 1 N。Instinct 同权重但覆盖所有非 ankle_roll 身体；当前其他部位是否摔倒由另外的终止条件处理。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:273)


### 12. termination_penalty：-200

真正失败终止给 -200×0.02=-4/次。超时与成功穿越属于 truncation，不触发这项。所对照的 Instinct G1Rewards 没有同名失败奖励，依靠终止和其他成本；不能据此机械删除本项。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:38)


### 13. feet_edge：-4

新 v8 虚拟边缘：5 cm 半径有限圆柱；每脚 100 个三维体积点，取每点最大穿入深度，乘点世界速度范数+1e-6，再求和。权重 -4，与 Instinct 核心公式一致；几何仅覆盖本项目方形楼梯暴露上沿。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/stair_virtual_safety.py:150)


### 14. dont_wait：-0.5

前进指令>0.3 时，速度<0.15、<0、<-0.15 分别累计 1；每份 -0.5。我们额外给开局 0.5 秒缓冲，Instinct 原函数没有该缓冲。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py:152)


### 15. heading_error：-1

负的目标控制器转向指令绝对值；此处指令由目标角误差生成，所以可改善朝向，并不是直接计算 yaw 误差平方。与 Instinct 一致。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py:139)


### 16. stand_still：-0.3

零速且零转向指令下：关节偏离默认站姿的 L1 和减 4，再乘 -0.3。接近默认姿态时可成为正奖励，上限 +1.2/s；与 Instinct 配置一致。它只约束站立指令，并不是要求完成楼梯后站稳。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py:73)


### 17. upright_body：-3

骨盆和躯干分别投影重力，水平分量平方求和，乘 -3；对应 Instinct root/躯干与 pelvis 两项各 -3 的设计意图。根链接配置不同，不宜仅按项数判断强弱。允许 yaw，约束俯仰/横滚。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py:99)


### 18. low_pelvis_height：-6

根位置相对两踝较低者不足 0.55 m 的缺口平方。额外本地项，非 Instinct 同名项；不按真实支撑脚选地面高度，楼梯上可能高估支撑净空。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/locomotion_progress.py:62)


### 19. is_alive：3

未失败时 +3/s，与 Instinct 同权重。不是行进进度奖励，原地保持也可以获得。 [计算源码](/home/ljc/isaaclab6/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:33)


### 20. feet_toe_riser：-2

本地新增防脚尖撞立面项，非 Instinct 原项。每脚选 x>=0.10 m 的 20 个前端体积点，在立面前 10 cm 内且点低于上沿时，对向高侧接近速度和反向水平法向接触力加罚。每脚取最大点成本再求和。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/stair_virtual_safety.py:150)


### 21. descent_overspeed：-0.75

本地下楼专用项：max(实际水平速度范数-指令范数-0.15,0)^2，超过部分截到 2，再乘 -0.75。Instinct 此配置没有此项。 [计算源码](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/stair_speed_course.py:17)


禁用项：旧 `feet_support`、单独的 `flat_orientation_l2`、髋/上肢/腰部关节偏置项。旧 `feet_edge_approach` 已被移除，`feet_edge` 名字保留但算法与 v7 不同。当前没有独立的“成功穿越加分”和“安全落脚中央加分”。

## 4. Instinct 中有、当前没有的配套项

对照的是 `G1Rewards` 的 26 项，不包括其单独的 AMP 辅助奖励。

- **承重脚姿态 feet_flat_ori，-0.4**：接触门控后惩罚脚掌倾斜。可用于减少承重阶段的踮脚/脚尖支撑，但不能在离地摆动、正常蹬离和斜坡上一律要求水平；移植时应使用支撑相位与局部踏面法线。
- **脚掌贴地 feet_at_plane，-0.1**：有接触时根据脚部高度扫描约束脚链接到地面的高度。不是“鼓励摆动脚抬高”的奖励，不能混用。Instinct 的鞋底版高度偏移为 0.058 m，基础版本为 0.035 m；必须按自己的碰撞模型校准。
- **足间侧向距离 feet_close_xy，+0.4**：其函数返回 exp(-缺口/std²)-1，结果不大于零，因此正权重实际在惩罚两脚过近，不能把它误当成“鼓励靠拢”。
- **髋关节默认姿态偏差，-0.5**：当前禁用。可减少内扣/外撇，但过强会妨碍跨阶和转弯。
- **全身关节速度平方，-1e-4**：当前无同等项。
- **电机功率平方 energy，-5e-5**：Instinct 按刚度归一化，当前无同等项；不能仅复制权重到不同执行器模型。
- **上身关节偏离 freeze_upper_body，-0.004**：当前无持续同等项；只在需要抑制过度上身摆动时考虑，不能牺牲必要平衡。
- **全关节速度软限位，-1** 与 **力矩超过 80% 限额，-0.01**：当前无同等项。
- **全关节位置软限位**：双方权重 -1，但当前只覆盖踝，Instinct 覆盖全部关节。
- **足滑**：当前 -0.1，Instinct -0.4；接触信号细节也有差别。
- **非脚部接触**：当前只覆盖肘腕；Instinct 所对照配置覆盖全部非脚链接。

“缺项”不自动等于“必须全部补上”。当前机器人根链接、动作缩放、足部碰撞、执行器、运动参考和地形课程均与 Instinct 不完全相同，应逐项验证。

来源：[Instinct 全部任务奖励](/home/ljc/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:653)、[脚部奖励公式](/home/ljc/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py:114)、[带鞋模型偏移](/home/ljc/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py:96)。

## 5. 避边实现究竟与 Instinct 对齐了多少

### 已对齐

- 半径 5 cm、有限长度圆柱；超出端面不当成圆头胶囊。
- 每脚 10×5×2=100 个体积采样点。
- 每点在多个圆柱中取最大穿入深度。
- 深度乘该点速度范数再求和，速度包含刚体平移和角速度引起的点速度，权重 -4。
- 只是训练中的虚拟安全区域，不改变物理台阶形状，不给机器人增加隐形碰撞支撑。

### 尚未等同

- Instinct 的边缘提取与空间索引可作用于多类地形。我们只处理本项目方形楼梯的暴露上沿，块状地形等没有同一套保护；也不覆盖全部竖直棱线。
- 我们沿用基础脚部采样框；带鞋 Instinct 的 Z 范围不同，不能称完全相同机器人几何。
- 当前采样后端 x=-0.025 m，而既有本项目碰撞核对记录的后跟锚点达到 -0.05 m。**后跟覆盖范围需要重新对照实际 USD 检查**，不能只因采样数量一样就认定覆盖完整。
- 圆柱进入提示只说明进入预留安全带，不说明脚实际碰撞边沿，也不说明真实承重面积。

### 明确的静态承重缺口

代价乘点速度，没有承重力、承重时长或支撑比例。因此静止踩在边上与高速穿过安全带的奖励后果差别很大。

计算示例（不是实测轨迹）：假设 20 个点各穿入 2 cm：

- 点速 1 m/s：每控制步约 -4×0.02×20×0.02=-0.032。
- 点速 0：只剩 epsilon，每控制步约 -0.000000032。

这说明单独使用该项不能实现“承重脚不长期压边”。同时它用速度**范数**，离开安全带时只要仍在带内也会受罚；不同于旧接近项只罚向边缘运动。这个特征与 Instinct 一致，但需要配套奖励平衡。

5 cm 半径也不应只按“越大越安全”理解：30 cm 踏面减去当前 14.5 cm 采样长度与前后各约 5 cm 的保守留白，示意可用平移空间只剩约 5.5 cm。真实三维圆柱与脚倾角会改变范围，这只是解释高精度落脚需求的保守一维估计，不是精确可行域。

来源：[本地实现](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/stair_virtual_safety.py:23)、[Instinct 穿入奖励](/home/ljc/InstinctLab/source/instinctlab/instinctlab/envs/mdp/rewards/volume_points.py:16)、[本项目历史足部碰撞核对](/home/ljc/legged_lab_v3/docs/analysis/foot_support_20260911/verification.md:10)。

## 6. 新防脚尖撞立面项的改进空间

现有项是本地扩展，不是 Instinct 的原始奖励。它已经能在碰撞前提供信号，但有以下限制：

1. **可以用减速规避代价。** 接近成本乘向立面的速度，停下来会减小；所以必须同时有方向正确的行进收益和抬脚净空引导，不能只提高此项负权重。
2. **没有额外抬脚余量。** 当前只要采样点达到踏面上沿高度，就退出此项的低脚尖区域，未要求额外 2–3 cm 的摆动净空。5 cm 圆柱有补充作用，但只覆盖上沿附近。
3. **只看前端 20 点/脚。** 对脚外侧、脚跟、胫部碰撞没有同样完整的几何覆盖；转弯、斜走或倒走时局限更明显。
4. **碰撞力是整个脚链接的法向合力。** 结合脚尖接近立面来推断撞击，不是精确的脚尖接触点定位；需要接触点/接触对数据来验证误判率。
5. **越过立面超过 2 cm 的点退出 active。** 正常刚性接触通常阻止深穿入，但离散采样、接触设置或大速度下不能把这当作全面碰撞检测。
6. **没有单独的脚尖撞击次数、冲量和摆动净空日志。** 非零奖励只能证明函数触发，不能证明减少了实际撞击。

建议拆成三个可解释指标：摆动脚越阶净空不足、立面真实撞击/冲量、落地后安全支撑。前者预测风险，中者统计真实结果，后者限制落脚质量；应分别记录，避免一个总惩罚混合解释。

## 7. 课程与奖励的接口

当前课程：17 级、阶高最高 20 cm；上/下楼分开，高度与 8 cm 速度分别推进。当前阶段至少 100 次且经历一个完整回合时长后，成功率达到 80% 升级、低于 50% 降级。

合格主要看穿越、未失败、巡航时长/跟踪/速度。**当前 `passed` 没有检查虚拟边缘承重比例、脚尖撞击次数或落脚支撑。** 所以“升高了”不能证明“学会不踩边/不撞立面”。

建议：保留用户要求的“穿越即可，不需末端站稳”，增加仅针对过程安全的记录；先记录、标定阈值，再决定是否加入升级门槛。不要把任何安全圆柱穿入都设为失败，也不要要求整个步态全程零临边，否则正常蹬离可能把课程卡死。

额外一致性问题：奖励跟踪使用航向坐标系 root_link 速度，巡航合格/部分诊断使用机体 root 速度。上楼时机体俯仰和竖直运动会让两种数值不同，应统一语义，或明确两者用途并专门测试。

来源：[课程通过判定](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/adaptive_stair_course.py:134)、[巡航统计](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/stair_speed_course.py:82)。

## 8. 暂停前日志证据及限制


### 旧 v7：最后 20 个完整日志块（46400–46419）

- 每步任务奖励：0.100330。
- 每步 AMP：0.304495。
- 每步净总奖励：0.404825；AMP/净总奖励约 75.2%。
- value loss：0.4253。
- 日志 Mean episode length：847.4 步，约 16.95 s。
- 生存项 episode 日志：0.847055；线速度项 0.524870。
- 避边项 episode 日志：-0.023395（旧算法，不能直接和 v8 比成‘踩边变多’）。

[原始日志](/home/ljc/legged_lab_v3/logs/stairs17_crossing_20260916_121132.log)


### 新 v8：最后 20 个完整日志块（46461–46480）

- 每步任务奖励：0.090885。
- 每步 AMP：0.304765。
- 每步净总奖励：0.395635；AMP/净总奖励约 77.0%。
- value loss：0.5216。
- 日志 Mean episode length：652.1 步，约 13.04 s。
- 生存项 episode 日志：0.654555；线速度项 0.401865。
- 避边项 episode 日志：-0.096980；脚尖立面项 -0.047810。

[原始日志](/home/ljc/legged_lab_v3/logs/stairs_instinct_edges_20260916_170334.log)


读数限制：

- v8 正式日志只有 81 轮（46400–46480），外加此前独立 64 环境/20 轮的 smoke；远不足以证明新奖励的长期效果。81×24×0.02≈38.88 s 仿真推进时间/环境，尚未达到迁移后课程最少等待的 60 s；上楼 8 cm、下楼 10 cm 未变化也不能据此判断新奖励失败。
- 最新 v8 保存文件为 model_46400.pt；日志有 46480 不等于保存了 46480。旧播放主要用 46000 旧奖励模型，不能把旧模型行为当成新 v8 的成败证据。
- Episode_Reward 是“已结束回合累计项 / 最大回合时长 60 s”，并经过日志聚合，不是简单每步奖励；不要把这些数和 AMP/per_step 直接相加或比较占比。短回合会让生存项、跟踪项变小，不能直接解释成该项瞬时质量按相同比例变差。
- 每步 AMP/净总奖励约 77% 是尺度信息，不是策略梯度贡献比例，也不是已证明的失败原因。需要消融实验。
- 新旧奖励函数发生变化，环境被重置、课程统计重新积累，采样分布也不同。这组日志不是控制变量评估；只能说明新成本确实生效、价值分布发生变化，不能证明避边改善或退化。

## 9. 建议的改进顺序

### 优先级 1：先确保我们惩罚的是想纠正的行为

- 在代表性场景标定：平地全脚支撑、脚外邻边但未跨边、稳定半脚踩边、正常蹬离、脚尖撞立面、正常高抬脚过阶、斜坡与斜向上下楼。
- 记录实际接触点/接触对或法向冲量、脚部姿态、虚拟穿入、承重状态和最低摆动净空。统一 train/play 同一物理定义，显示“安全带进入”和“实际撞击”两个概念。
- 校准后跟和脚尖采样框；不要盲目套带鞋 Instinct 的 Z 偏移。

### 优先级 2：补齐落脚与抬脚目标，再谈权重

- 保留虚拟圆柱；新增轻量、只在可靠承重阶段有效的边缘支撑成本，不乘脚速度。用承重比例/接触状态门控，并对离地、正常蹬离与缺失数据正确处理；不恢复旧“任一射线未命中就当踩边”的逻辑。
- 引入承重脚与局部踏面法线一致、落地后贴合踏面的约束，参考 Instinct 的 feet_flat_ori / feet_at_plane，但使用自己的接触几何与倾斜地形门控。
- 给接近上一级立面的摆动脚建立 terrain-relative 净空目标，例如初步试验 2–3 cm 额外余量；只在即将越阶的摆动相位生效，避免每一步都高抬腿，也避免让支撑脚因为没抬高而受罚。
- 防撞成本与成功前进协同，避免通过停止或极慢挪脚逃避惩罚。

### 优先级 3：小范围、一次一组的权重消融

建议候选而非已验证最优值：

- AMP：0.5 与 0.25 两档，保留同一时序单位。不要直接乘 dt。
- 足滑：先比较 -0.1 与 -0.2，再决定是否需要 Instinct 的 -0.4；同时保护正常离地滚动。
- 足部姿态/贴地：低强度启动，排查平地和坡面退化，再增加。
- 新避边/防撞：可对权重做逐渐增加的过渡，而不是同时再叠加大量强惩罚。观察 value loss、KL、动作幅度和真实碰撞率；必要时做 critic 适应，不能仅凭回报下降重置整个策略。
- 暂不优先提高 upright、竖直速度或低骨盆惩罚；爬高需要屈膝和竖直运动，容易误伤。

### 优先级 4：课程与完整软安全约束

- 安全过程指标标定后再加入升级标准；保留“穿越无需站稳”。
- 补审全关节软位置/速度/力矩边界及非足部接触；按实际执行器参数设置，不机械照搬。
- 推广虚拟几何到 boxes 等其他地形，但过滤埋藏边、拼接缝和非暴露边，避免旧问题换一种形式重现。

## 10. 推荐评估设计（尚未执行）

从同一个保存点、相同种子和相同预算比较：

A. 现有 v8 基线。
B. 仅改变 AMP 系数为 0.25。
C. 仅补承重落脚约束与摆动净空引导。
D. B+C；前三组无严重退化后再做。

第一阶段固定在 8/10/12 cm 与 0.6/0.8 m/s，上下楼分开、每格建议至少 100 次有效尝试并报告区间。通过后再扩展 14–20 cm、1.0–1.5 m/s；同时保留平地 1.5/2.5/3 m/s 回归测试。

主要指标：穿越率、跌倒率、实际速度/超速、每次穿越的脚尖撞击次数与冲量、承重阶段边缘占用时间、落脚安全余量、滑移距离、摆动最低净空、关节/力矩边界触发比例。回报只作辅助，不以“总奖励更高”代替行为改善。

## 11. 审计产物

- [当前全部奖励及函数源码快照](/home/ljc/legged_lab_v3/docs/analysis/reward_audit_20260916_sources.json)
- [日志窗口汇总和 Instinct 26 项清单](/home/ljc/legged_lab_v3/docs/analysis/reward_audit_20260916_logs.json)

本报告区分代码事实、日志事实与待验证推断。报告给出的候选权重和阈值是实验设计，不是已验证结论。
