# 支撑转换第二阶段公开证据

实验完成于 2026-10-16，发布于 2026-10-20。对应 4096 环境、15700→16700 的 1000 次 PPO 更新。

- [配对汇总](aggregate_results.json)：两组基线各 192 个桩路案例，64 个已有地图加 128 个新种子、同生成族地图。
- [实验定义](experiment_manifest.json)、[汇总报告数据](summary.json)。
- [训练前关节修正诊断](pretrain_action_diagnosis.json)、[可观测奖励子集](reward_subset_diagnosis.json)。
- [同一 14700 重复校准](repeat_calibration_summary.json)、[平地动作函数检查](final_flat_function_check.json)、[最终完整性检查](final_integrity_checks.json)。
- [视频清单与逐文件 SHA256](publication_manifest.json)、[本地关键帧核查记录](visual_review.json)。

各配对目录的 `results.json` 包含逐回合状态和身体指标，`arm_audit.json` 与 `support_transition_audit.json` 包含臂腿协调和落脚统计。`diagnosis_*` 为零更新的冻结模型关节投影干预；`repeat14700_calibration` 为同一模型重复测量。

2026-10-22 更新后，公开梅花桩录像仅保留16700的正向与斜向演示。原8999演示、包含14700／15700的对比视频及其缩略图已从当前分支移除。新录像取原始记录中16700的独立画面，保留仿真时间和过桩至停车过程；真实Isaac Sim 6物理轨迹重放，MuJoCo仅渲染姿态，不属于MuJoCo动力学复验。[在线播放](https://lessonllab.github.io/legged_lab_v3/#stones)

原始评估 JSON 内容保持不变，来源路径指向实验机器；媒体清单已更新为仅16700，公开副本的内容哈希见清单。完整逐帧张量、训练日志和模型权重没有纳入发布。三种策略均通过本批 192/192 场景，不能据此宣称通过率提升、任意地形泛化或真实传感器效果。重复运行不构成额外独立地图，只有一个训练种子。

[方法、运行命令与结果边界](../../docs/support_transition.md) · [完整实验记录](../../docs/analysis/support_transition_stage2_20261016.md)
