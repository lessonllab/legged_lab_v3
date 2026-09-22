# G1 深度视觉运动控制：近两年论文与选题建议

检索日期：2026-09-14。时间范围：2024-09-14 至 2026-09-14，按首次公开时间筛选。研究范围根据当前 legged_lab_v3 项目确定：G1 人形、深度视觉、复杂地形、AMP 与自然步态。本次筛选 18 篇相关论文，阅读摘要并核查关键方法和局限；不是穷尽性系统综述。选题建议是研究假设，不能等同于已证实的首创性。

## 建议

沿现有工程推进，优先验证方向 1“参考步态的可行性与必要偏离”。有稳定的 G1 真机与相机条件时，方向 2“动作条件下的接触风险”值得重点投入。方向 3“记忆失效与撤退”作为周期较长的备选。三条方向择一作为主线，不建议一次堆叠。

普通深度 CNN/GRU、地形条件 AMP、视觉与盲走融合、边缘落脚惩罚、自然走路与楼梯混合训练，都已有接近的工作。论文需要提出一个具体瓶颈，并用公平对照证明新的机制改善了它。

## 当前工程的关联

已阅读 [v7 说明](/home/ljc/legged_lab_v3/docs/g1_target_v7.md)、[风格与任务平衡核对](/home/ljc/legged_lab_v3/docs/analysis/target_v5/instinct_style_balance_20260913.md) 与 [此前研究修改计划](/home/ljc/legged_lab_v3/docs/analysis/g1_depth_20260910/research_modification_plan.md)。

项目已有深度 CNN、视觉历史、遮挡与延迟、AMP、参考动作镜像、六种地形、支撑约束和步态评估。文档记载过持续蹲姿、奖励尺度不一致及判别信号变弱等问题，但其因果贡献尚未由独立消融确定。v7 的最新风格调整仍待新训练验证。工程修复是可靠基线的前提，不自动成为论文贡献。

## 18 篇论文：已有覆盖与选题影响

日期为 arXiv 首次提交日期，不混用修订时间或会议年份。“未核实录用状态”不表示论文未录用。ADD 属物理角色研究，LF2WB 的真机是四足，二者作为相邻工作纳入。下列“影响”为本次分析。

### 1. PIM · 2024-11-21

[Learning Humanoid Locomotion with Perceptive Internal Model](https://arxiv.org/abs/2411.14386)

- 已做：使用机器人中心高程图和内部模型实现感知运动控制。
- 对选题的影响：高度图、本体估计、稳定爬楼梯已是基线能力。
- 状态：arXiv；本次未核实录用状态。

### 2. VB-Com · 2025-02-20

[VB-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception](https://arxiv.org/abs/2502.14814)

- 已做：用可部署回报估计器组合视觉与盲走策略，应对感知失真和动态障碍。
- 对选题的影响：“视觉不可信时切到盲走”已被直接研究。
- 状态：作者项目页标注 ICRA 2026。

### 3. Challenging Terrain · 2025-03-02

[Learning Perceptive Humanoid Locomotion over Challenging Terrain](https://arxiv.org/abs/2503.00692)

- 已做：教师学生训练结合带变分信息瓶颈的世界模型，处理噪声地形估计。
- 对选题的影响：加入世界模型或去噪辅助任务，本身不足以建立新颖性。
- 状态：arXiv；本次未核实正式出版页。

### 4. ADD · 2025-05-08

[Physics-Based Motion Imitation with Adversarial Differential Discriminators](https://arxiv.org/abs/2505.04961)

- 已做：以对抗差分判别器处理多目标优化，应用于物理角色动作模仿。
- 对选题的影响：自动平衡奖励并非空白；这是相邻的仿真角色研究。
- 状态：作者论文标注 SIGGRAPH Asia 2025。

### 5. DPL · 2025-10-08

[DPL: Depth-only Perceptive Humanoid Locomotion via Realistic Depth Synthesis and Cross-Attention Terrain Reconstruction](https://arxiv.org/abs/2510.07152)

- 已做：真实感深度合成、自遮挡建模与交叉注意力地形重建。
- 对选题的影响：深度噪声、自遮挡、地形重建需要作为强对照。
- 状态：RA-L 2026；arXiv v3 2026-08-03。

### 6. Hiking in the Wild · 2026-01-12

[Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids](https://arxiv.org/abs/2601.07718)

- 已做：单阶段深度到动作学习；地形边缘、足体积点与可行目标采样改善安全和训练。
- 对选题的影响：与你当前 Instinct 风格视觉 AMP 工程最接近，优先建立可靠基线。
- 状态：arXiv；本次未核实录用状态。

### 7. PHP · 2026-02-17

[Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching](https://arxiv.org/abs/2602.15827)

- 已做：动作匹配串联人类技能，再把跟踪专家蒸馏成深度视觉多技能策略。
- 对选题的影响：多技能切换、动态跑酷与长序列通过，竞争已很强。
- 状态：arXiv；本次未核实录用状态。

### 8. LF2WB · 2026-03-03

[Look Forward to Walk Backward: Efficient Terrain Memory for Backward Locomotion with Forward Vision](https://arxiv.org/abs/2603.03138)

- 已做：前行时写入紧凑地形记忆，后退时检索；真机是 Lite3 四足。
- 对选题的影响：倒退记忆的思路已有先例；人形迁移还需新的机制与证据。
- 状态：arXiv 标注 ICRA 2026；相邻四足研究。

### 9. PRIOR · 2026-03-19

[PRIOR: Perceptive Learning for Humanoid Locomotion with Reference Gait Priors](https://arxiv.org/abs/2603.18979)

- 已做：参数化步态先验、GRU 深度地形估计与地形适应落脚奖励。
- 对选题的影响：自然步态＋深度视觉＋落脚奖励的组合已有直接竞争者。
- 状态：arXiv；本次未核实录用状态。

### 10. CReF · 2026-03-31

[CReF: Cross-modal and Recurrent Fusion for Depth-conditioned Humanoid Locomotion](https://arxiv.org/abs/2603.29452)

- 已做：本体查询交叉注意力、门控融合和 GRU，直接由前向深度学习控制。
- 对选题的影响：把 CNN 换为注意力、GRU 或加一个融合门控，很难单独立题。
- 状态：arXiv v3 2026-07-27；本次未核实出版页。

### 11. GLAD · 2026-05-30

[Global-Local Attention Decomposition for Terrain Encoding in Humanoid Perceptive Locomotion](https://arxiv.org/abs/2606.00637)

- 已做：分解全局地形与局部落脚注意力；G1 使用机载 LiDAR 部署。
- 对选题的影响：稀疏落脚点与粗细粒度地形编码也已有专门研究。
- 状态：arXiv v3 2026-08-10。

### 12. TAGA · 2026-06-04

[TAGA: Terrain-aware Active Gaze Learning for Generalizable Agile Humanoid Locomotion](https://arxiv.org/abs/2606.05880)

- 已做：融合视觉、本体和指令，选择高度扫描中的信息区域。
- 对选题的影响：主动关注地形已经有人做；论文标题的 gaze 不等于机械转头。
- 状态：arXiv；本次未核实录用状态。

### 13. T-GMP · 2026-06-05

[T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion](https://arxiv.org/abs/2606.06944)

- 已做：用配对状态—地形数据学习 CVAE 运动先验，并使用地形条件判别器。
- 对选题的影响：不能把“地形条件 AMP”直接当新点；可探索不依赖配对数据的必要偏离。
- 状态：arXiv v2 2026-08-29。

### 14. MARCH · 2026-06-09

[MARCH: Model-Assisted Reinforcement Learning for the Perceptive Control of Humanoids over Sparse Footholds](https://arxiv.org/abs/2606.10288)

- 已做：简化模型生成安全参考、CLF 奖励训练教师，再蒸馏视觉学生。
- 对选题的影响：物理安全约束＋视觉落脚控制已有强对照。
- 状态：arXiv；本次未核实录用状态。

### 15. ADP · 2026-07-03

[ADP: Adversarial Dynamics Priors for Physically Grounded Humanoid Locomotion](https://arxiv.org/abs/2607.03454)

- 已做：从轨迹优化参考提取动力学特征，进行对抗正则并研究扰动恢复。
- 对选题的影响：仅把接触力、动量加入判别器，也不足以宣称新颖。
- 状态：arXiv v2 2026-07-16。

### 16. HumoSlope · 2026-07-08

[Physics-Guided Biomechanical Gait Adaptation for Humanoid Locomotion on Extreme Sloped Terrains](https://arxiv.org/abs/2607.07830)

- 已做：坡面 ZMP 与地形门控的生物力学步态约束，改善持续蹲行。
- 对选题的影响：“提高身高奖励解决蹲行”需要超出已有坡面姿态适配。
- 状态：arXiv；本次未核实录用状态。

### 17. Light-Loco-Parkour · 2026-08-01

[Light-Loco-Parkour: Versatile Perceptive Whole-Body Locomotion via Multi-Skill Distillation](https://arxiv.org/abs/2608.02653)

- 已做：从稀疏动作种子扩展地形配对参考，以单一深度策略执行多种全身技能。
- 对选题的影响：少量示范＋多技能蒸馏也已有近期工作。
- 状态：arXiv；本次未核实录用状态。

### 18. CAP · 2026-09-10

[CAP: Continuously Adaptive Perception-Blind Humanoid Locomotion via Learned Denoising](https://arxiv.org/abs/2609.11553)

- 已做：去噪世界模型、本体编码、深度噪声课程与特征 dropout，实现 G1 感知退化适应。
- 对选题的影响：仅做噪声鲁棒视觉融合会直接撞题；应定义更具体的决策风险问题。
- 状态：arXiv 标注已获 CoRL 2026 录用。


录用信息补充来源：[VB-Com 作者项目页](https://renjunli99.github.io/vbcom.github.io/)、[ADD 作者论文](https://add-moo.github.io/static/ADD_SIGGRAPH_ASIA_arxiv.pdf)。DPL、LF2WB、CAP 的状态来自各自 arXiv 页面声明；不将作者的性能数字直接当成可跨论文比较的排行榜。

## 候选方向与实验要求

### 1. 参考步态的可行性与必要偏离

**适配性：** 最贴合当前 AMP 工程；优先做小规模验证。

**研究问题：** 同一段参考走路动作，在台阶、窄支撑和外推扰动下何时不再可行？如何只偏离必要部分，并在风险解除后恢复自然步态？

**方法候选：** 把自然度、任务完成和接触失败分开建模。用短时可行性估计识别与支撑条件冲突的参考约束，再通过受约束优化控制偏离幅度与持续时间。优先尝试使用现有未与地形配对的走跑数据；不能给普通 mocap 随意配地形标签。

**创新边界：** 候选贡献是“无需额外地形配对示范的最小必要偏离与恢复机制”。地形条件判别器、通用奖励自动调权和动力学判别器都已有论文，必须明确超出它们的部分。

**最近邻：** [T-GMP](https://arxiv.org/abs/2606.06944)、[ADP](https://arxiv.org/abs/2607.03454)、[ADD](https://arxiv.org/abs/2505.04961)、[HumoSlope](https://arxiv.org/abs/2607.07830)、[Hiking in the Wild](https://arxiv.org/abs/2601.07718)。

**关键对照：** 先冻结参考数据、奖励单位和预算，扫描固定 AMP 权重，复现通过率—自然度的折中边界。再比较地形条件先验、通用约束/调权法和新方法；在未见地形与扰动时测试，而非只挑一个最优权重。

**评价指标：** 地形通过率、每百米跌倒、速度误差、足滑、接触冲击、按速度匹配的关节与支撑节律分布；记录偏离参考的持续时间和恢复时间。AMP 判别分不能作为唯一自然度指标。

**消融与否定条件：** 固定权重／只有通用约束优化／只有可行性估计／去掉恢复约束／完整方法。若收益仅来自增加高度奖励或降低速度，应否定当前创新假设。

**主要风险：** 实现性价比较高，但算法贡献要求严格。若只是把现有进展门控改成另一条经验曲线，论文说服力不足。当前 v7 属工程方案，不能据此声称研究效果已经验证。

### 2. 视觉失真下的接触风险预测与控制

**适配性：** 有 G1 和深度相机时优先；需要实际退化数据。

**研究问题：** 图像能被重建，不代表下一脚可以安全落下。能否预测特定落脚位置、时刻与速度下的接触失败概率，并据此选择动作？

**方法候选：** 以深度历史、本体历史和候选落脚/动作条件预测短时支撑失效风险。仿真中生成边缘踩空、延迟、遮挡的接触标签；用独立验证集校准风险。控制器在进度要求下调整落脚和速度，防止靠停住获得低风险。

**创新边界：** 候选贡献是“从像素恢复转向动作条件下、经过校准的接触风险”，而不是一般不确定性或视觉—盲走切换。需要证明校准质量会改变决策结果。

**最近邻：** [CAP](https://arxiv.org/abs/2609.11553)、[VB-Com](https://arxiv.org/abs/2502.14814)、[CReF](https://arxiv.org/abs/2603.29452)、[MARCH](https://arxiv.org/abs/2606.10288)、[DPL](https://arxiv.org/abs/2510.07152)。

**关键对照：** 在相同输入、参数规模、训练预算和噪声增强下，对比普通深度策略、去噪策略、切换策略，以及不做校准的风险控制。区分图像退化与地形真实变化，测试训练中未见的失效组合。

**评价指标：** 失败率、每百米跌倒、相同通过时间下的安全性、风险校准曲线/Brier 分数、误报引发的停顿、板载推理延迟。不能只报更低深度重建误差或更慢行走的成功率。

**消融与否定条件：** 去掉动作条件／去掉风险校准／去掉控制中的风险使用／等规模普通网络／完整方法。风险监督不向 actor 泄漏真值地形；部署接触信号只能使用硬件可取得或估计的量。

**主要风险：** CAP 已覆盖部分失真、丢帧和持续融合，因此宽泛的鲁棒感知题目不够。未观测且不可由历史辨识的地形无法保证安全；应把信息边界纳入实验。

### 3. 地形变化后的记忆失效与撤退

**适配性：** 更高风险备选；需要动态地形与非前向测试。

**研究问题：** 机器人看过的落脚点后来被移动了，或者转身后记忆发生漂移，应该何时忘记旧地形、重新观察或撤退？

**方法候选：** 在记忆中显式维护观测时效与可信度，用新的视觉证据和可部署接触残差纠正冲突。把重新观察、有限试探和撤退的决策与低层步态联动，而非仅延长帧堆叠。

**创新边界：** 候选贡献是双足条件下“检测记忆失效并恢复”，而不是已有的前视记忆倒退。LF2WB 的真机是四足，其本身已讨论不确定性，需要避免只把它移到 G1。

**最近邻：** [LF2WB](https://arxiv.org/abs/2603.03138)、[VB-Com](https://arxiv.org/abs/2502.14814)、[TAGA](https://arxiv.org/abs/2606.05880)、[CAP](https://arxiv.org/abs/2609.11553)。

**关键对照：** 看过场景后转向，改变部分支撑面，再执行侧行/倒退。与固定帧堆叠、GRU、关联记忆和无记忆策略比较；必须有普通稳定记忆场景作为对照。

**评价指标：** 旧记忆错误检测率、错误信任次数、完成率、恢复时间、额外观测/试探成本、相同时间预算的撤退成功率。

**消融与否定条件：** 去掉时效／去掉接触纠错／只做随机遗忘／完整方法。没有新观测或接触证据时，不能声称知道隐蔽环境已发生变化。

**主要风险：** 相邻研究与系统集成工作较多，周期较长。G1 是否具备可控视角及实际传感器接口尚未确认，不应默认能靠机械转头实现主动观察。


## 首轮验证计划

1. **固定一个可复现基线。** 冻结奖励单位、输入、参考集、初始化方式和训练预算；区分工程故障与研究瓶颈。当前不能把每次修改配置后的播放观察混作同一实验。
2. **先验证折中问题存在。** 比较无先验与多个固定 AMP 权重，使用同一批地形、指令和预算。自然度按速度/运动模式匹配，避免快慢差异产生假改善。
3. **只新增一个主机制。** 至少 3 个独立训练种子；小规模实验用于筛选，正式结果应报告不确定性。不能用同一训练种子的多个回合作为多个独立训练重复。
4. **进行真正的泛化评估。** 按地形家族、参数区间及失效类型划分训练/验证/测试；评估种子独立。拒绝只报告训练地形或成功视频。
5. **以应用指标决定是否继续。** 同时看成功、速度、足滑、自然度与成本。若只是降低速度、停止移动或增加网络容量获得优势，应重新评价贡献。
6. **条件允许再做真机。** 测量真实传感器退化与端到端延迟；部署只使用可取得的观测。只有仿真时优先强调方法机制、可复现基准与跨仿真器验证，不声称完成 sim-to-real。

建议主图：方向 1 用通过率—自然度折中边界和扰动后恢复曲线；方向 2 用风险校准与相同通过时间的失败率；方向 3 用旧记忆错误信任率与撤退成功率。

## 避免误判新颖性

- T-GMP 已使用地形条件先验；其原文第 7 节明确指出配对状态—地形数据成本和感知不确定性局限。把“降低配对数据依赖”当候选问题有依据，但具体方案仍需单独查新。[T-GMP v2](https://arxiv.org/html/2606.06944v2)
- CAP 在 2026-09-10 已公开部分遮挡、传感器退化下的连续融合方案，不能把宽泛的感知鲁棒性当成空白。[CAP](https://arxiv.org/abs/2609.11553)
- LF2WB 在 Lite3 四足上研究前视记忆倒退，也已提到引入估计不确定性的未来方向。只做“不确定性＋记忆”或移植到 G1，仍可能增量不足。[LF2WB 原文](https://arxiv.org/html/2603.03138v1)
- TAGA 的主动 gaze 涉及对高度扫描区域的选择，不能未经硬件核查就解释成 G1 主动转动相机。[TAGA](https://arxiv.org/abs/2606.05880)
- 身高、低膝屈曲、低能耗、平滑并不分别等同于自然步态。应结合速度匹配的运动分布、接触时序与独立评价；不能让训练判别器自证自然度。

本次未修改训练代码，未启动训练。真机可用性、算力和目标期刊尚未确认，因此未承诺发表级别或完成时间。

