# ljc 项目文档

## 当前主线

- [项目入口与运行命令](../README.md)
- [楼梯专项配置与完成判定](g1_stairs_specialist.md)
- [平地与崎岖地形混合课程](g1_balanced_course.md)
- [2026-09-15：8 个模型对照与噪声复测](analysis/stair_specialist/checkpoint_comparison.md)
- [研究方向记录](research/humanoid_locomotion_directions_20260914.md)

## 历史实现与排查

以下记录对应各自日期的代码和环境，运行命令优先以根 README 为准。

- [深度视觉基础](g1_depth_training.md)、[视觉历史与身体遮挡](g1_depth_instinct.md)
- [视觉遮挡修复验证](analysis/depth_fix_20260911/verification.md)
- [脚底支撑与边缘检测验证](analysis/foot_support_20260911/verification.md)
- [平地预训练](g1_depth_warmup.md)、[Target v7](g1_target_v7.md)、[Scratch 起步课程](g1_scratch_instinct.md)
- [旧楼梯完成判定诊断](analysis/stair_specialist/diagnosis_23200.md)

## 归档与清理

2026-09-15 清除了 Python/测试缓存；不适配当前环境的旧 Docker 工具、旧工具专属 CLAUDE.md 与整理前 README 已保存到 [历史工具归档](archive/legacy_tooling_20260915.tar.gz)。归档用于追溯，不作为当前运行入口。

较大的历史 TensorBoard 导出 `analysis/training_compare_20260911/series.json` 已无损压缩为 `series.json.gz`；原始训练事件、模型和其他评测证据保留。归档与压缩前后均校验内容哈希，详见 [清理记录](archive/cleanup_20260915.json)。

[来源与许可](../NOTICE.md)单独保存，当前维护署名为 ljc。
