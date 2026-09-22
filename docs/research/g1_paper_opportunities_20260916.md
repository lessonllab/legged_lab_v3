# G1 视觉运动控制：近两年论文核查与可验证选题

检索日期：2026-09-16。主要时间窗：2024-09-16—2026-09-16。用户已确认有 G1 实机，可进行真实地形和传感器实验。

这是面向当前项目的定向调研，不是穷尽性系统综述。下列文献以论文原文、arXiv、作者项目页和会议出版页核查；关键近邻进一步阅读方法或局限。日期采用 arXiv 首次提交日期，正式录用状态单独列出。未核实录用的论文只按预印本讨论。所有新设计均为待验证的研究假设，不代表已证明首创或保证录用。

## 选题判断

最值得先做小规模验证的是三个独立方向：

1. **带平衡约束的主动地形试探**：根据视觉与接触证据决定何时试探、如何施力、何时转移体重，解决“看起来能踩，实际上未必能承重”的问题。
2. **保留补救空间的视觉落脚决策**：落脚前比较继续、调整步、减速和撤回的后果，避免进入只有一次机会的状态。
3. **身体能力随时间变化的视觉通行性**：把关节热状态与可用力矩纳入地形判断，研究长时间运行后的步态、路线与休息决策。

方向 1 的研究上限和实机辨识度较突出，但需要搭建可控接触场景；方向 2 最接近当前踩边和停稳问题，但最近邻竞争很强；方向 3 适合持续运行实验，前提是本机能观测到可重复的热状态与能力变化。先择一，不建议将三者同时拼入第一篇论文。

## 当前项目提供了什么基础

已阅读 README、Instinct 深度历史与网络配置、楼梯长期训练配置，以及 2026-09-16 的检查点评测记录。

- 平台是 G1 29 自由度，Isaac Lab、PPO/AMP；已有走跑参考动作、地形课程、视觉遮挡与延迟、固定场景评测工具。
- 已读的 Instinct 实现使用 64×36 原始深度，经裁剪得到 32×18；37 个控制帧缓冲，稀疏取 8 帧并加入 0/1 帧延迟。actor 与 critic 各自有视觉 CNN，该配置移除了 critic 的理想高度扫描。这些事实应在具体实验分支上再次核对。
- 因此可以直接复用低层运动能力，新增研究模块优先输出候选步态、接触意图或残差；无需一开始重建整个系统。
- 最新本地评测提示：低台阶速度控制改善，并不总伴随支撑质量改善；较高台阶、低速跟踪和高速能力保持仍有问题。这是选题线索，不是因果结论。
- 报告中“临边支撑不足”是射线采样几何代理，不能作为实机压力中心或实际接触面积真值。本次没有重新运行仿真，也没有验证实机性能；用户确认拥有实机不等于当前模型已完成实机验收。

本地证据：[项目说明](/home/ljc/legged_lab_v3/README.md)、[最近评测](/home/ljc/legged_lab_v3/docs/analysis/precontact_40600_20260916.md)、[深度历史实现](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/instinct_depth.py)、[网络配置](/home/ljc/legged_lab_v3/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_depth_instinct_ppo_cfg.py)。

## 已经拥挤的技术区域

“深度图到动作”“自然步态与复杂地形”“视觉失效时盲走”“地形重建或世界模型”“预测能否安全停机”均有直接近邻。研究贡献需要落在新的决策问题、可辨识的机制和公平的实验上。换 CNN/Transformer、增加噪声、提高避边奖励或扩大课程，适合作为基线工程，单独立题通常较弱。

尤其需要修正两种容易过时的判断：

- WM-LOCO 已将递归世界模型用于受限落脚的 G1 视觉运动控制；其状态转移本来就使用过去动作。“动作条件世界模型”这个标签本身不能证明新颖性。[论文](https://arxiv.org/abs/2609.02542)
- PRISM 与 Safe-Stop 已研究策略相关的可安全停机预测。仿真回放打标签、训练风险头、达到阈值切换控制器，这一流程也不是空白。[PRISM](https://arxiv.org/abs/2603.22703)、[Safe-Stop](https://arxiv.org/abs/2609.02358)

## 方向 1：带平衡约束的主动地形试探

**研究问题。** 视觉几何相似的硬地、软支撑、可转动板或低摩擦表面，其物理可通行性不同。机器人能否只在信息不足且影响决策时，进行一次有目的的试探，再决定踩、绕或撤？

**建议题目。** *Probe Before You Commit: Balance-Constrained Active Terrain Identification for Humanoid Locomotion*（暂拟）。

**已有工作与边界。** 2025 年的 Load-bearing Assessment 已在四足上根据计划地面反力包络试探承重，并用 MPC 和状态机维持平衡；它还根据结果改换落脚。因此“力随任务变化”“踩一下再决定”都不应声称首次提出。SPI-Active 也已用主动激励提高机器人参数辨识质量。[承重试探原文](https://arxiv.org/html/2510.21369v1)、[SPI-Active](https://arxiv.org/abs/2505.14266)

**候选新贡献。** 在人形有限支撑条件下联合优化“试探动作的信息价值”和“试探失败后仍能回到已确认支撑的能力”；通过视觉先验决定是否需要试探，并把试探造成的地面状态改变纳入后验。四足方法迁移到 G1 本身不够，必须体现信息获取、负载转移与动态平衡的耦合。

**最小实现。**

1. 冻结或轻度微调现有低层策略，先训练可靠的站立、短步与撤脚行为。
2. 用深度、本体历史、可部署的力矩/接触估计维护地形参数分布，第一版只辨识局部刚度或板的转动响应。
3. 高层选择正常迈步、有限幅值试探、撤回或绕行；信息奖励衡量试探后决策相关不确定性下降，并计入时间、能耗与平衡代价。
4. 从相同外观、不同支撑刚度的场景开始。更难的隐蔽塌陷与摩擦辨识作为后续扩展。

不能凭轻触就判定地面可承受整机重量；轻触可能只提供局部刚度和阻尼信息。应输出已测试载荷范围内的证据或保守区间，允许“不足以判断”。当前项目是深度视觉，“相同外观”的最小定义是相同可见几何；若利用材质外观先验，需要额外 RGB 输入。

**实机实验。** 可替换弹簧支撑板、刚性对照板、有限行程倾转板；同一表面覆盖层下改变支撑参数。记录探测力/位移、重心转移时刻、错误承重、撤脚成功、通过时间和试探次数。第一轮使用有支撑保护的低速短步，不以摔倒视频作为主要验证。

**决定性对照。** 被动 RNN 适应；每步固定试探；按力包络试探；随机试探；相同时间预算的等待/减速；带信息价值但不约束恢复；完整方法。所有方法保持相同传感器、交互次数和可用控制技能。

**继续条件。** 学到“已知硬地少试探、歧义区域有选择地试探”，且在相同通过时间或风险水平下优于固定试探。若收益完全来自普遍减速，或传感信号无法区分目标物理属性，应先缩小问题。

## 方向 2：保留补救空间的视觉落脚

**研究问题。** 两个落脚点都可能成功，但一个失误后还能补步，另一个失误后没有任何可达支撑。策略能否在落脚之前利用这种差异？

**建议题目。** *Keeping Recovery Options Open: Perception-Conditioned Contact Decisions for Humanoids*（暂拟）。

**最近邻。** BeamDojo 与 MARCH 已研究稀疏落脚；WM-LOCO 已学习未来预测表征；PRISM 与 Safe-Stop 已预测固定停机策略是否成功；HWC-Loco 已协调任务策略和恢复策略。仅增加辅助头、风险门控或恢复专家不构成清晰差异。[BeamDojo](https://arxiv.org/abs/2502.10363)、[MARCH](https://arxiv.org/abs/2606.10288)、[HWC-Loco 正式出版页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8c854e7870dc67e0b410a266cd45e0ad-Abstract-Conference.html)

**切入点。** Safe-Stop 的局限明确指出，有些状态立即停止不可行，但继续动作后可能重回可停区域，其框架尚未处理这一情况。可将其发展为复杂地形上的“有期限、受环境约束的恢复通道选择”。这支持一个候选问题，不能据此推导整个问题从未被研究。[原文局限](https://arxiv.org/html/2609.02358v1#S6)

**最小实现。** 给现有策略提供若干短时控制候选（常规前行、短步、侧补步、减速），预测各候选之后，在有限时间内通过指定补救策略到达稳定支撑区域的概率。使用相同模拟状态和受控扰动生成成对 rollout 标签，重点采样边缘、窄支撑和视觉歧义场景。

一个可操作的估计对象为：

\[
q(o_t,u)=P(\text{执行候选 }u\text{ 后，在期限内由指定补救策略到达稳定支撑，且无失败}\mid o_{0:t}).
\]

这里的 q 取决于补救策略、时间界限、扰动分布和观测信息；它不是机器人全部物理可恢复状态的证明。高分头也可能被策略利用，需要独立 rollout 校验与闭环数据更新。

**关键设计。** 将未来接触的后果直接用于动作选择；视觉推断需能识别附近可用补救支撑。部署网络只读真实可得信息，仿真隐变量只用于监督。候选空间先用几个短时技能，避免对 29 维逐帧动作进行昂贵全搜索。

**实验。** 构造“主落脚相同、周边补救支撑不同”的成对地形，以及“几何相同、相机延迟不同”的组合。测通过率、危险误放行率、补步后持续成功、风险—速度曲线和推理时延；把时间到达而非原地停滞纳入成功标准。

**关键对照。** 固定避边/支撑奖励；降速基线；相同网络容量的普通 value/risk head；固定停机可行性监控；不看候选动作的风险估计；不使用未来恢复价值；完整方法。若完整方法不能超过简单可停机监控，则收窄主张。

## 方向 3：热状态改变“哪里还能走”

**研究问题。** 同一机器人与台阶，在冷机和长时间运行后可能具有不同的可用力矩与动态余量。能否在看到接下来的地形时，提前调整步长、领先腿、速度和休息位置，保留完成任务所需的能力？

**建议题目。** *When Terrain Becomes Too Hard: Thermally Conditioned Perceptive Humanoid Locomotion*（暂拟）。

**已有覆盖。** 2026 年已有两篇四足热感知运动控制工作，其中一篇将热状态输入残差策略。TAGA 的局限直接提到长时间运行的电机热负荷会损害精确落脚与动态动作表现。[热感知策略](https://arxiv.org/abs/2603.01631)、[热管理残差策略](https://arxiv.org/abs/2605.27046)、[TAGA 局限](https://arxiv.org/html/2606.05880v1#S7)

**候选贡献。** 建立“机器人当前能力条件下的地形通行性”，让视觉预瞄驱动跨步、跨地形的负载分配；研究某条路线现在能走，但走完之后是否仍能安全完成后续任务。创新不能停留在温度输入和过热惩罚。

**先做的数据验证。** 实机记录可用温度遥测、关节电流/力矩估计、跟踪残差与环境温度。确认传感器测量位置、滞后和本机限流策略；壳温不能直接当绕组温度。用正常作业范围的冷暖状态比较，建立留出序列上的预测误差。

**最小方法。** 在仿真加入经实测标定的低阶热状态，建模能力约束及其不确定性；高层根据未来地形与关节负载预算调整运动。不同关节不对称的热负担可测试换领先腿和分配任务是否有用。

**实机实验。** 连续混合路线；冷/暖起点；相同路线的不同段落顺序；已知总任务下的安全休息点选择。预先规定停止阈值，主要看单位时间完成距离、任务完成率、峰值温度、跟踪误差与人工干预。不能用“几乎不动所以不热”作为胜利。

**对照与否证。** 热状态输入但无视觉预瞄；纯视觉无热状态；静态限速；简单阈值休息；热残差策略；完整方法。如果这台 G1 在目标工作范围内没有可重复的能力变化，就不应强行造出该论文问题。

## 方向 4：为看清而改变运动

**问题。** 当前视觉观测取决于身体运动。能否主动侧移、调整躯干/机身朝向或选择采样时刻，让下一步的支撑区域进入视野并减少自身遮挡？

**边界。** TAGA 的 gaze 主要是选择高度扫描中的信息区域；DPL 已模拟自身遮挡。可探索实际身体动作改变后续可观测性，而非仅在已获取特征中加注意力。[TAGA 方法](https://arxiv.org/html/2606.05880v1)、[DPL](https://arxiv.org/abs/2510.07152)

**设计。** 联合选择运动和感知预算，用未来落脚不确定性降低指导短暂观察动作。G1 相机安装方式和可控关节必须先核实，不能默认存在独立可转动头部。固定相机可通过躯干、机身转向和侧步改变视角。

**判别实验。** 几何可通行但观察角度不同的同一地形；固定视角、被动历史、固定扫描动作、特征注意力和主动身体视角控制对照。匹配时间、帧数与算力，排除单纯多看几帧或减速收益。额外云台或第二相机应另列硬件对照。

**优先级。** 创意较好，工程增量中等；先验证身体改变视角确实能增加任务相关信息，再训练联合策略。

## 方向 5：拿着不同东西，地形的可通行性也不同

**问题。** 携带物体会同时改变惯性、身体外形和相机可见区域；同一通道对空手和抱箱子的机器人是不同任务。

**候选贡献。** 学习机器人—物体联合占据体积和动力学能力条件下的通行性，协同决定物体姿态、身体姿态、步态和路径。第一版用稳定固定的轻质物体，控制质量、尺寸、重心和遮挡，暂不引入复杂抓取。

**近邻与边界。** Gallant 已处理三维受限空间；LALO 已研究负载与上肢运动对行走的耦合。需要证明联合处理几何、惯性和遮挡，超出两个模块简单组合。[Gallant](https://arxiv.org/abs/2511.14625)、[LALO](https://arxiv.org/abs/2603.14308)

**实验。** 相同质量不同尺寸、相同尺寸不同重心、仅改变相机遮挡；持物过窄门与低障碍，留出全新组合。测身体/物体碰撞、物体加速度、通过率和运输时间。对照包含只加质量随机化、只扩充碰撞模型及只加强感知三类。

## 方向 6：把落脚点扩展为“空间与时间窗口”

**问题。** 轻微移动或转动的支撑面，不仅要求选择踩哪里，还要决定何时接触、何时转移载荷，以及是否等待。

**设计。** 估计短时支撑运动及不确定性，输出落脚位置—接触时刻的联合候选；可先研究运动支撑板的等待与短步，不必一开始做高速跳跃。相机运动补偿、板运动辨识和控制延迟需分离。

**边界。** VB-Com 已包含动态地形/感知缺陷；PHP 已展示实时障碍扰动适应。动态地形不是空白场景，应突出预期接触时间窗与等待决策。[VB-Com](https://arxiv.org/abs/2502.14814)、[PHP](https://arxiv.org/abs/2602.15827)

**实验。** 低幅值可控运动平台，未见周期/相位/非周期运动，延迟交叉测试；与静态几何策略、更长历史策略、无时间预测策略对比。必须真实更新视觉和碰撞几何，否则仿真任务不成立。设备成本和辨识成本高于前四项。

## 方向 7：参考动作只在必要处偏离，并在之后恢复

**问题。** 平地走跑先验与窄台阶支撑约束冲突时，如何只放松相关时段和身体部位的模仿要求，在冲突消失后恢复原有自然度？

**设计。** 从当前 AMP 扩展为受接触可行性约束的最小偏离：把偏离持续时间、关节组和恢复自然步态作为显式对象。优先利用既有未与地形配对的走跑参考，减少额外数据要求。

**边界。** T-GMP 已有地形条件生成先验与判别器；在线生成参考动作的整套 G1 方法也已有论文。普通 AMP 调权、条件判别器、Diffusion 参考生成不能独立作为新点。[T-GMP](https://arxiv.org/abs/2606.06944)、[Motion Generation and Tracking](https://arxiv.org/abs/2604.17335)

**实验。** 固定参考数据量、奖励尺度与预算，扫描 AMP 权重形成通过率—自然度折中曲线；使用未参与训练的动作度量评估偏离和恢复时间。若新方法未改善折中边界，只优于一个调得不好的固定权重，证据不足。该方向代码适配最好，但竞争较强。

## 方向 8：知道记忆何时失效的长程撤退

**问题。** 走过的地形记忆可能因物体移动、踩踏变形、位姿漂移而失效。机器人遇到死路时，能否辨别记忆可靠性，重新观察并选择撤退路径？

**设计。** 记忆附带观测时间、空间对应和接触验证证据；用预测—接触不一致触发更新。研究“不能原地掉头、前方不通、来路局部改变”的场景。

**近邻。** LF2WB 已用前向视觉写入记忆支持后退，因此单纯加 GRU、增加历史或训练倒退不足以立题。候选差异是记忆失效检测与有选择地重观测。[LF2WB](https://arxiv.org/abs/2603.03138)

**实验。** 静态/改变来路的对照，匹配地图大小与历史长度，评估无人工重置任务完成率、每百米干预、撤退碰撞和错误记忆采纳。与一直相信记忆、固定遗忘、仅本体反馈对比。

## 四个更发散的储备想法

这些只完成相邻领域初筛，不能据此宣称文献空白。

- **用脚步声修正接触认知。** 接触后短时声学/振动可能帮助分辨空心、松动或软支撑。比较音频对关节/IMU历史的增量信息，防止把“新模态”当贡献；必须做场地、鞋底、环境噪声的跨域测试。
- **允许有目的的非足接触。** 手轻扶墙、扶栏或短暂撑住物体以扩展可行通道。Bracing for Impact 已在仿真研究用手撑墙恢复，PHP 已有全身越障；需要研究视觉选择接触面及可承载性的验证。[撑墙恢复](https://arxiv.org/abs/2505.11495)
- **可被人理解的运动控制。** 在狭窄空间用身体朝向、让行步和速度变化表达通行意图，联结局部运动和人类判断。需要真正的人体实验和清晰行为指标，单纯接入语言指令较弱。
- **因果诊断式评测与训练场景生成。** 构造“同图不同物理”“同几何不同恢复支撑”“同任务不同身体状态”的成对环境，检验策略到底依赖哪些信息；用失败类型引导生成场景。若作为独立论文，需要统一协议、多个强基线、可复现数据与跨场景结论，单纯随机地形生成器不够。

## 统一实验原则

1. 选择一个主要研究问题和一条核心机制。让每个新增模块都有可删除的消融，避免为论文堆组件。
2. 区分三个层次：几何看起来可踩、动力学允许执行、执行失败后仍可补救。分别打标签，不能只看累计 reward。
3. 数据分开训练、调参、校准和最终测试。按地形布局、材料或设备实例划分，防止相邻帧跨集合泄漏。
4. 仿真建议至少 3 个独立训练种子；同一组独立场景种子配对评测。实机先用每条件 10–20 次筛查，再根据效应大小与区间宽度确定正式样本数；这不是可靠性认证样本量。
5. 实机报告分母、失败定义、人工介入与超时。按独立任务/回合而非连续帧计算置信区间；相邻帧不能伪装成大量独立样本。
6. 同时报告成功率、时间/速度、能耗或负载、动作自然度和实际推理延迟。让“始终停下”“始终慢走”失去虚假的指标优势。
7. 视觉噪声、延迟、接触参数和地形改变分开控制，再测试未见组合，才能识别方法在哪类故障上有效。
8. 脚底真实接触可用外部测量验证，例如力板/压力测量或同步运动记录；若使用电流推断接触，说明其估计误差。训练几何代理与实机物理测量不要混写。
9. 风险预测报告可靠性图、Brier 分数、危险状态误放行与覆盖率—成功率曲线；必须比较同样保守程度。分布外测试的经验校准不是形式安全保证。
10. 基线按同传感器、训练交互数、参考数据、参数规模、推理硬件进行比较。无法复现的完整系统应明确标注为方法参考或近似实现。

## 建议的第一轮验证

这是工作分解，不是工期或论文录用承诺。

**先冻结基线。** 留存平地与楼梯模型、配置、参考动作列表和固定评测协议。建立独立研究分支；不将后续修改覆盖到正在训练的配置。核实真实相机链路、时延、可读电流/力矩/温度及是否能部署短步/站稳控制。

**优先验证方向 1 的信息价值。** 先在仿真构建同可见几何、不同局部刚度的成对环境。比较无试探、固定试探和有选择试探的上界；确认一次可控接触确实提供了视觉和已有历史不能提供的信息。然后在实机做低速短步的数据采集。这个问题成立后，再投入完整强化学习训练。

**若物理辨识信号不足，验证方向 2。** 在冻结策略上离线生成候选后果，检查可恢复性是否能区分当前几何风险指标看起来相同的状态。只有预测差异真实存在且影响决策，才开始训练动作选择器。

**若已有明显持续运行瓶颈，方向 3 可以提前。** 用冷暖起点的重复实验，验证地形与热状态的交互效应；先证明问题存在，再增加热模型。

论文贡献应可压缩为：明确的新问题、解决该问题的一个关键机制、排除替代解释的实验。成功率改进本身需要回答“为什么发生”和“在哪些条件下不发生”。

## 文献清单：25 篇核心与邻近工作

以下每项都附原始来源；篇名后的日期为首次 arXiv 提交，HWC-Loco 仅列已核实的正式发表年份。

1. **PIM — Learning Humanoid Locomotion with Perceptive Internal Model**，2024-11-21。机器人中心高程图与内部模型；作者主页列为 ICRA 2025。感知行走的基础对照。[论文](https://arxiv.org/abs/2411.14386)、[作者出版列表](https://oceanpang.github.io/)
2. **BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds**，2025-02-14，RSS 2025。多边形足部采样奖励、双 critic、两阶段训练与 LiDAR 高程图；稀疏落脚不是空白。[论文](https://arxiv.org/abs/2502.10363)、[正式论文](https://www.roboticsproceedings.org/rss21/p068.pdf)
3. **VB-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception**，2025-02-20。视觉/盲走策略选择，应对感知缺陷与动态地形；本文未另核正式出版状态。[论文](https://arxiv.org/abs/2502.14814)
4. **Learning Perceptive Humanoid Locomotion over Challenging Terrain**，2025-03-02。教师学生框架结合世界模型和变分信息瓶颈；去噪世界模型已有基础。[论文](https://arxiv.org/abs/2503.00692)
5. **VideoMimic — Visual Imitation Enables Contextual Humanoid Control**，2025-05-06；作者项目页列 CoRL 2025。视频联合重建人和环境，学习场景相关技能。[论文](https://arxiv.org/abs/2505.03729)、[作者项目页](https://www.videomimic.net/)
6. **Bracing for Impact: Robust Humanoid Push Recovery and Locomotion with Reduced Order Models**，2025-05-16。用环境墙面和手臂辅助恢复；摘要报告仿真实验，不能当作 G1 实机结果。[论文](https://arxiv.org/abs/2505.11495)
7. **SPI-Active — Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning**，2025-05-20。通过信息量驱动的激励采集改善参数辨识；主动探测的相邻方法。[论文](https://arxiv.org/abs/2505.14266)
8. **DPL: Depth-only Perceptive Humanoid Locomotion via Realistic Depth Synthesis and Cross-Attention Terrain Reconstruction**，2025-10-08，RA-L 2026。自身遮挡、深度合成和地形重建。[论文](https://arxiv.org/abs/2510.07152)
9. **Load-bearing Assessment for Safe Locomotion of Quadruped Robots on Collapsing Terrain**，2025-10-24，RA-L 2025。根据计划反力包络试探承重，MPC 与状态机协调；四足的直接近邻。[论文](https://arxiv.org/abs/2510.21369)、[出版页](https://doi.org/10.1109/LRA.2025.3626249)
10. **Gallant: Voxel Grid-based Humanoid Locomotion and Local-navigation across 3D Constrained Terrains**，2025-11-18，CVPR 2026。体素感知、头顶与侧向约束；三维通过性已有强基线。[论文](https://arxiv.org/abs/2511.14625)、[CVF 出版索引](https://openaccess.thecvf.com/content/CVPR2026/html/Ben_Gallant_Voxel_Grid-based_Humanoid_Locomotion_and_Local-navigation_across_3-D_Constrained_CVPR_2026_paper.html)
11. **Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids**，2026-01-12。单阶段深度到动作、地形边缘与足体积点、可行目标采样；与本项目非常接近。[论文](https://arxiv.org/abs/2601.07718)
12. **PHP — Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching**，2026-02-17。动作匹配组合技能，再蒸馏成深度多技能策略，G1 实机。[论文](https://arxiv.org/abs/2602.15827)
13. **Learning Thermal-Aware Locomotion Policies for an Electrically-Actuated Quadruped Robot**，2026-03-02。四足热状态输入与热约束奖励；不能声称首次将温度加入策略。[论文](https://arxiv.org/abs/2603.01631)
14. **LF2WB — Look Forward to Walk Backward: Efficient Terrain Memory for Backward Locomotion with Forward Vision**，2026-03-03；arXiv 标注 ICRA 2026。前向观察写记忆、后退读取；记忆与倒退研究的近邻。[论文](https://arxiv.org/abs/2603.03138)
15. **LALO — Load-Aware Locomotion Control for Humanoid Robots in Industrial Transportation Tasks**，2026-03-15。历史估计负载/上身耦合扰动、残差行走；arXiv 声明投稿 TIE，不能写已录用。[论文](https://arxiv.org/abs/2603.14308)
16. **PRISM — Learning Safe-Stoppability Monitors for Humanoid Robots**，2026-03-24。固定回退策略相关可停机域，重要性采样改进风险边界学习，实机验证。[论文](https://arxiv.org/abs/2603.22703)、[作者项目页](https://intelligent-control-lab.github.io/humanoid_stoppability/)
17. **Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking**，2026-04-19。在线地形相关参考生成、全身跟踪与闭环微调，G1 实机。[论文](https://arxiv.org/abs/2604.17335)
18. **Learning to Balance Motor Thermal Safety and Quadrupedal Locomotion Performance with Residual Policy**，2026-05-26。四足热模型与残差策略，长时间负载行走；与第 13 项是不同论文。[论文](https://arxiv.org/abs/2605.27046)
19. **TAGA: Terrain-aware Active Gaze Learning for Generalizable Agile Humanoid Locomotion**，2026-06-04。根据视觉、本体、指令选择局部高度扫描区域；gaze 不等于物理转头，论文明确讨论热负荷局限。[论文](https://arxiv.org/abs/2606.05880)
20. **T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion**，2026-06-05。地形条件运动先验、条件判别器与落脚惩罚；地形 AMP 的直接近邻。[论文](https://arxiv.org/abs/2606.06944)
21. **MARCH: Model-Assisted Reinforcement Learning for the Perceptive Control of Humanoids over Sparse Footholds**，2026-06-09。简化模型安全参考、CLF 奖励教师、视觉学生，G1 部署。[论文](https://arxiv.org/abs/2606.10288)
22. **Safe-Stop — Humanoid Safe Stop via Learned Stoppability Value**，2026-09-02。停机概率与 reach-avoid 值联合判断；作者明确列出暂不可停但继续可能恢复可停性的问题。[论文](https://arxiv.org/abs/2609.02358)、[全文局限](https://arxiv.org/html/2609.02358v1#S6)
23. **WM-LOCO — World-Model-Augmented Visual Locomotion for Humanoids on Foothold-Constrained Terrain**，2026-09-02。RSSM 与 PPO 联合训练，预测本体/深度/奖励，G1 落脚受限地形。[论文](https://arxiv.org/abs/2609.02542)
24. **CAP: Continuously Adaptive Perception-Blind Humanoid Locomotion via Learned Denoising**，2026-09-10；arXiv 作者声明 CoRL 2026 录用。去噪世界模型、本体编码、噪声课程和特征 dropout；普通视觉退化适应已高度接近。[论文](https://arxiv.org/abs/2609.11553)
25. **HWC-Loco: A Hierarchical Whole-Body Control Approach to Robust Humanoid Locomotion**，ICLR 2026 正式出版。协调任务与安全恢复的层级策略；本文未将会议年份充当首次预印本日期。[会议出版页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8c854e7870dc67e0b410a266cd45e0ad-Abstract-Conference.html)

## 本次检索范围与未解决问题

检索词覆盖 humanoid perceptive locomotion / depth parkour / active probing / load-bearing / recoverability / safe-stoppability / thermal-aware locomotion / payload / active gaze / dynamic terrain，并对初筛近邻追查标题、原文方法和局限。没有将搜索引擎的抓取时间当成首次发表日期，也没有将不同论文的成功率直接组成排行榜。

仍需在立题后做一次聚焦查新：确定方向 1 的双足主动辨识近邻、方向 2 的多策略可恢复域与视觉 reach-avoid 近邻，或方向 3 的人形热约束规划近邻。每条“候选贡献”都有潜在早期相关工作；当前调研支持优先验证顺序，不支持“世界首个”的表述。
