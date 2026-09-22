# 来源与许可说明

本项目由 **ljc** 维护和扩展。README 与包元数据中的维护署名用于标识当前衍生版本，不改变上游代码、模型或数据的来源与许可。

## 上游项目

本仓库基于 [Legged Lab](https://github.com/zitongbai/legged_lab) 扩展。原 README 中的上游引用保留如下：

```bibtex
@misc{legged_lab,
  author       = {Zitong Bai},
  title        = {Legged Lab: An Isaac Lab Extension for Legged Robot Reinforcement Learning},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/zitongbai/legged_lab}}
}
```

仓库原有 [LICENCE](LICENCE) 为 Apache License 2.0。旧 README 的 MIT 徽章与该文件不一致，现已移除该徽章；本次整理没有重新许可第三方内容。

## 依赖与参考来源

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)：仿真、任务管理与机器人相关基础设施。
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)：PPO 训练基础。
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)：上游 AMP 实现参考。
- [GMR](https://github.com/YanjieZe/GMR)：运动重定向工具参考。
- [MimicKit](https://github.com/xbpeng/MimicKit)：运动模仿参考。
- [InstinctLab](https://github.com/project-instinct/InstinctLab)：视觉、地形和课程相关参考与适配。部分适配文件明确标注 CC BY-NC 4.0，须保留并遵循对应声明。
- 本地 AME_Locomotion 项目用于梅花桩地形设计参考；具体实现来源按相关文件记录保留。

依赖、机器人资产和动作数据分别沿用其附带许可与来源声明，不因当前项目署名调整而变更。源码中的原版权头、适配说明及第三方许可证不在品牌标签清理范围内。
