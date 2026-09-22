# 根据原论文与官方代码制定的修改方案

核对日期：2026-09-10。对象为当前 G1 EDU 29DoF 视觉 AMP 基线。本文件是研究结论和实现方案，尚未修改训练超参数，也未启动新训练。

## 已验证的问题与判断修正

第 31200 轮模型在固定 0.5 m/s、零转向下，两种随机种子共 128 个回合均未达到 4 米；113 个存活回合的平均前进速度仅 0.0163 m/s。当前需要解决的是基础前进，而不是放宽课程门槛。详见 `straight_verification.md`。

需要修正此前的解释：前进奖励 1、转向奖励 2、速度核 std=0.5，本身不是已确认的错误。
Isaac Lab 官方 G1 示例采用相同比例。该示例使用 G1_MINIMAL 模型，并非当前完整 29DoF AMP 任务，系数只能作为对照依据，不能保证直接移植有效。
[官方 G1 配置](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py)。

## 优先修改一：固定探索强度

AMP 原论文第 6.2 节采用固定对角协方差；MimicKit 的 G1 AMP 示例也使用固定标准差，并将熵奖励权重设为 0。
当前策略标准差从 0.5 增至约 1.98，训练与确定性播放差距明显，这应先做对照。
[AMP 原论文](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)，[MimicKit G1 AMP 配置](https://github.com/xbpeng/MimicKit/blob/main/data/agents/amp_g1_agent.yaml)。

本项目建议的首个实验：保持初值 0.5，设置 `learn_std=False`，`entropy_coef=0.0`。
0.5 是保留本项目原有初值以控制变量，不是论文提供的最优数值；不要照搬 MimicKit 的 0.05，两个项目动作定义和尺度不同。
固定标准差仍然在训练中采样随机动作，不能把 PPO actor 改成完全确定性分布。

本机 RSL-RL 5.4.1 的 GaussianDistribution 已支持 `learn_std` 和 `std_range`，无需改其库文件。
Isaac Lab 的配置类未显式声明这两个字段，实现时应派生本项目的 distribution 配置类，然后用于 `rsl_rl_depth_ppo_cfg.py`。
旧 checkpoint 中仍有已学到的 std 参数，完整 resume 会恢复这些权重；新实验从头训练最清楚。如做权重迁移，须显式重置 std、优化器状态并记录迁移范围。

## 优先修改二：给“应该走却不走”明确代价

InstinctLab 的 G1 跑酷配置包含 `dont_wait`，权重 -0.5；其实现针对前进指令 >0.3 m/s，在实际前进速度 <0.15 m/s 时开始处罚，倒退时进一步增加处罚。
它同时使用目标点生成的速度指令。不能只复制奖励系数、忽略指令定义。
[奖励实现](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py)，[任务配置](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py)。

本项目的适配建议：

- 先建立固定 0.5 m/s、零横移、零转向的直行训练任务。
- 添加 `dont_wait` 对照项，从 -0.5 开始；仅在明确非零前进指令时生效，静止指令下关闭。
- 可加入重置后的短暂宽限期，避免惩罚正常启动过程。这是本项目建议，不是上述源码原有机制。
- 暂不同时收窄速度核和大幅增大前进奖励，以便识别新增项的实际作用。

该项不是唯一修复：它仍可能诱发滑步或冲刺后摔倒，因此必须一起观察存活、足滑、实际速度和完整回合位移。

## 优先修改三：重新平衡运动代价和辅助步态奖励

当前扭矩惩罚为 -2e-6，而上述官方 G1 示例为 -1.5e-7，系数绝对值相差约 13.3 倍。
当前 feet_air_time 权重 0.75，官方为 0.25。系数差异不等于实际奖励贡献相差相同倍数，实际值还依赖动作、机器人和动力学。

建议单独测试扭矩惩罚 -1.5e-7；若动作过于激烈，再比较中间值 -5e-7。保留力矩限幅、关节限制和摔倒终止条件。
把 feet_air_time 先降回 0.25 作对照；不要同时加强脚离地奖励试图强迫迈步。

本机 `feet_air_time_positive_biped` 根据单脚接触状态和持续时间给分，只用指令速度作开关，没有要求身体实际前进。
所以持续单脚支撑理论上也可获得奖励。结合最近该项约 0.265、理论上限 0.75×0.4=0.3，它值得专项检查；这些数字并不能证明机器人已经在利用单脚静止行为。
如减小权重后仍有问题，再测试以实际沿指令速度作为门控，或使用落脚事件与步长条件；这些是待验证改法。

对应改动位置：独立 warmup/v2 配置，继承 `g1_amp_depth_env_cfg.py`；不直接覆盖所有 rough/flat 任务的奖励。

## AMP：先查输入和记录实际奖励，再改判别器

原 AMP 强调示范与策略使用一致的状态特征、跨训练轮次的历史回放和梯度惩罚。
当前已有缓冲区，但 `construct_algorithm` 将 `max_len` 设成 `num_steps_per_env`，容量仅一轮 rollout；
下一轮采样结束时上一轮的数据已被全部覆盖。因此它没有原论文意义上的跨轮历史回放。这是确定的实现差异，但不是已证明的唯一失败原因。
本地 MimicKit 实现单独记录风格奖励均值/标准差，当前训练日志则缺少同等信息。
[AMP 原论文](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)，[MimicKit AMP 实现](https://github.com/xbpeng/MimicKit/blob/main/mimickit/learning/amp_agent.py)。

实现顺序：

1. 在 PPOAMP 中记录在线 `task_reward`、`style_reward`、`mixed_reward`、零风格奖励比例、在线与示范分数分布。
2. 在相同参考帧下核对关节顺序、角速度坐标系、单位与 0.02 秒采样间隔；确认在线历史和示范均按时间递增。参考窗口是向前取样、策略窗口是过去历史，本身不意味着时间顺序错误。
3. 分别检查归一化后的策略和示范各维分布、异常值、重置后的重复帧。MimicKit 也可只用策略数据更新归一化，不能把“没有混合示范更新”直接判作 bug。
4. 统计当前 walk_and_run 数据里实际速度和转向分布，先选与直行速度范围匹配的有效片段。不要只按文件名筛选，也不要靠减小指令范围假装示范动作已变慢。
5. 将策略历史回放容量与 rollout 长度解耦，首个工程候选为 10 轮容量；采样时同时覆盖当前与历史样本，缓冲区未满时只采有效项。10 轮是本项目的试验起点，不是论文指定数值。更新恢复 checkpoint 的逻辑和有效样本计数，增加跨轮保留测试。
6. 上述通过后才做判别器强度消融。不要盲目继续降低已经为 1e-5 的学习率，或移除梯度惩罚。

当前判别器没有身体线速度输入；核对示范后可把它作为新增特征做独立对照，仅供训练判别器使用。这不是已确认的根因，而且改变输入维度后不能直接恢复旧判别器。

## 训练顺序：保留视觉目标，先建立可验证的基础步态

短期最小改动方案：新增 `DepthWarmup` 实验，保留相同 CNN、本体和深度输入，但地面改为平地、指令固定直行。
平地深度只用于维持输入和 checkpoint 结构，不宣称这时已经学会视觉避障。直行达标后，再转入缓坡、低台阶和低幅起伏，最后加入转向。

更完整的视觉路线是先训练使用地形特权观测的教师，再以学生 rollout 学习深度输入策略。
Extreme Parkour 的公开流程明确区分基础策略训练和带相机的蒸馏阶段；它是四足机器人结果，不能直接承诺对 G1 同样收敛。
另一方面，InstinctLab 的 G1 任务本身包含深度策略输入，说明教师蒸馏并不是视觉训练唯一合法路线。
[Extreme Parkour 官方实现](https://github.com/chengxuxin/extreme-parkour)，[G1 视觉任务配置](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py)。

## 实验与验收

先增加日志和输入一致性检查，再建立固定直行条件。以下实验保持相同任务、种子和初始化方式，逐组比较，避免一次修改所有因素：

- 对照：当前训练设置。
- A：仅固定 std=0.5、关闭熵奖励。
- B：在 A 上降低扭矩惩罚，单独评估。
- C：在较好的基础上加入 dont_wait，并单独测试脚离地奖励降至 0.25 的影响。
- D：单独比较一轮缓冲和跨轮历史回放，观察在线风格奖励、判别器分数分布与实际直行能力。

每 500–1000 轮保存并做确定性固定直行评估。短程训练只作早期筛查，不保证几千轮足够收敛。
本项目建议的阶段验收目标为：两种以上种子、累计至少 128 个回合；存活率与 4 米达标率都达到 80% 以上，平均 XY 速度误差 <0.15 m/s；同时报告所有回合和存活回合，不能只挑成功样本。
这些是工程验收目标，并非文献给出的通用阈值。通过基础步态后，再按地形类别评估并进行深度冻结/打乱消融。

研究结论：先修复可解释的训练差异与观测问题，再建立直行能力；目前证据不足以保证某一组奖励数值一定解决问题，也不支持只降低地形升级门槛。
