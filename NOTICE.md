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
- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3)：G1 全身控制、运动奖励及命令课程适配来源；本项目的 `unitree_official/` 与相关合同记录使用该固定版本。
- [ezoicoder/humanoid-gym](https://github.com/ezoicoder/humanoid-gym/tree/9a47b7dfe85fdcc737c3c8963d80d05997376575)：BeamDojo 地形、局部高程输入和落脚惩罚适配参考；双评论器由本项目适配到原生 RSL-RL 5.4.1。相关来源、固定提交与适配差异记录在任务代码和检查点合同中。
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)：上游 AMP 实现参考。
- [GMR](https://github.com/YanjieZe/GMR)：运动重定向工具参考。
- [MimicKit](https://github.com/xbpeng/MimicKit)：运动模仿参考。
- [InstinctLab](https://github.com/project-instinct/InstinctLab)：视觉、地形和课程相关参考与适配。部分适配文件明确标注 CC BY-NC 4.0，须保留并遵循对应声明。
- 本地 AME_Locomotion 项目用于梅花桩地形设计参考；具体实现来源按相关文件记录保留。

依赖、机器人资产和动作数据分别沿用其附带许可与来源声明，不因当前项目署名调整而变更。源码中的原版权头、适配说明及第三方许可证不在品牌标签清理范围内。
