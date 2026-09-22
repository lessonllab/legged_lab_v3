# G1 从零起步课程：参考本地 InstinctLab

任务：`LeggedLab-Isaac-AMP-Scratch-G1-v0`。这是独立实验，未更改正在训练的 V7。

## 源码依据与差异

本地 `/home/ljc/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py` 使用：速度指数跟踪权重各 2.0，存活奖励 3.0，目标驱动速度指令。`config/g1/agents/instinct_rl_amp_cfg.py` 使用初始探索标准差 1.0、学习率 1e-3、自适应调度、直接相加的 AMP 风格奖励，以及默认 `resume=False`。该入口不要求预训练模型。

本任务恢复这些起步机制，使用已有视觉网络和机器人模型，不宣称复现其 MoE 网络或全部传感器、动力学、奖励。风格系数按用户要求保留 0.50（Instinct 默认 0.25）；取消 V7 额外的运动/速度风格门控。风格不再取决于是否已经跟好速度。现有参考片段、采样权重、镜像、时间采样和深度输入完全保留。存活奖励按环境步长积分，权重 3.0 对应 0.06/控制步；AMP 风格系数不乘步长。

保留本项目的高度、倾斜、接触终止、脚边缘与低姿态代价。探索标准差额外限制在 0.05~1.5，这是本项目的限制。没有直接使用 V5/V7 actor 初始化。

## 课程

地形共 10 行、10 列，每类地形的第 0 行都是真正的平地。第 1 行起恢复原六类地形，从低难度逐级增加。全部机器人首次从第 0 行开始，出生/关节重置方式仍参考 Instinct 的站姿随机重置。

- 第 0 级：平地，目标在出生点前方 3 米，速度上限采样 0.45~0.60 米/秒。
- 第 1、2 级：低难度崎岖地形，恢复原地形目标点，逐渐放宽速度上限。
- 第 3 级及以上：各地形完整速度范围；台阶/方块最高 0.8、坡面最高 1.2、随机起伏最高 2.0 米/秒。更高等级继续增加地形难度。

Instinct 的配置实际使用整回合归一化速度跟踪分，平移升级阈值 0.6、降级阈值 0.3，角速度升级阈值配置为 0（不是辅助函数默认的 0.5）。本任务保留这些阈值，并增加本项目的限制：完整回合正常超时、没有真实失败、有至少 2 秒移动机会、向目标接近至少 1 米，连续成功 3 回合才升一级。静止即使跟踪分高也不升级。失败或跟踪差则降级，最低为平地；最高级不随机跳回别处。**没有增加脚步计数升级门槛**：最终方案以 Instinct 的速度跟踪课程为主。

每个机器人独立升降级；不存在到某个固定训练轮数就强制升级。日志 `Curriculum/terrain_levels/mean_level`、`Curriculum/terrain_levels/flat_fraction`、`Curriculum/terrain_levels/rough_fraction`、`Curriculum/terrain_levels/full_speed_fraction` 展示课程分布。

模型保存 `scratch_course` 中的地形等级和成功连续次数。恢复时同步出生位置并重置回合，避免用旧位置开始新阶段；初始化重置不计入升级。改变环境数量时，在对应地形列内均匀抽取已有课程状态，避免把台阶的等级错配到坡面。地形行数必须保持一致。旧 V7 checkpoint 不作为该任务的 resume 输入。

## 从零开始

如果旧训练仍在运行，先在其终端按 Ctrl+C。下面的命令只启动新任务，不会自动终止旧训练。

```bash
cd /home/ljc/legged_lab_v3
ISAACLAB_PYTHON=/home/ljc/isaaclab/bin/python \
bash scripts/run_with_rsl5.sh scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Scratch-G1-v0 \
  --viz none --device cuda:0 --num_envs 2048 \
  --max_iterations 1000 --seed 42 --run_name flat_entry_style05 \
  agent.device=cuda:0 agent.resume=false
```

先做 1000 轮观察，不承诺该轮数必然学会走路。应同时比较回合长度、真实失败、目标进展、有效到达和重放步态；风格分不是自然步态准确率。通过后从同一实验继续：添加 `--resume --load_run <新实验目录名> --checkpoint <保存的模型名>`，去掉 `agent.resume=false`。

## 验证边界

逻辑测试检查参考和网络输入不变、平地/崎岖生成边界、速度范围、禁止静止与失败升级、保存恢复课程。小规模真实仿真仅验证程序运行和训练链路，不是训练收敛或自然步态验收。本任务未被证实能避免先前出现的静止局部最优；需要上述阶段性行为评估。

2026-09-14 验证：5 项新课程测试、11 项原目标任务回归、7 项 AMP 奖励回归通过。最终十列布局完成 64 环境、2 轮真实训练，再从保存模型恢复 2 轮；日志确认课程恢复成功且退出码为 0。全部测试机器人初始等级为 0，十列覆盖六类地形。最终测试日志为 `/tmp/g1_scratch_course_final.log` 和 `/tmp/g1_scratch_course_final_resume.log`。这些短测没有验证自然步态或学习后的升级成功率。
