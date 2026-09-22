# Git 发布范围

对照日期：2026-09-22。上游为 https://github.com/zitongbai/legged_lab ，对照提交为 `69b7f73d912a9d5695a60b5bed41d0785c8b302e`。

## 与上游的对应关系

- 上游 `source/legged_lab/`：提交本项目对应包、任务配置、算法实现、测试、机器人和动作资源，以及新增的视觉、楼梯和 Foothold 模块。
- 上游 `scripts/`：提交本项目训练、播放、重定向、诊断、评估、启动及 sim2sim 脚本。
- 上游 `.vscode/`、格式检查、打包配置：保留当前已跟踪的共享开发配置。
- 上游 `README.md`、`LICENCE`：README 沿用目录、概览、演示、更新、安装、使用、路线图、致谢结构；保留许可证，并补充 `NOTICE.md`。
- 上游 `data/MotionData`、`data/Robots` 使用 LFS：保留相同管理方式；体积计算包含真实资源而非指针。
- 本项目额外提交 `docs/`：使用说明、实验记录和评测证据。历史诊断状态数据不是可直接播放的训练策略。
- 上游 Docker 与代理配置：本机已删除或未保留的旧工具不恢复；历史工具归档在 `docs/archive/`。

## 不提交的内容

- `logs/` 中的训练检查点、TensorBoard 事件及运行输出。
- `.runtime/` 中的本机安装依赖和生成结果。
- Python 缓存、临时目录、私有环境文件和机器本地配置。
- 独立的 `hiking-in-the-wild-sim2sim/` Git 仓库及其编译依赖；按下文重建。

原有本地文件保留；忽略规则不删除磁盘数据。发布文件总量上限按十进制 **1 GB = 1,000,000,000 字节**解释，并额外核查单文件与历史对象体积。

本次暂存快照核查：617 个文件约 226.7 MB，最大单文件约 46.6 MB；当前分支可达历史和暂存 Git 对象按未压缩大小计算，再加去重后的 LFS 资源，约 250.3 MB。该值是保守内容核算，不是网络实际传输量。上游 176 个文件中有 161 个同路径保留；其 50 个机器人和动作资源全部包含。新克隆仍需要 `git lfs pull`。

## MuJoCo 外部资源

在项目根目录执行，仅适用于尚未准备该外部仓库的新克隆：

```bash
git clone https://github.com/jie0110/hiking-in-the-wild-sim2sim.git
git -C hiking-in-the-wild-sim2sim checkout 557d07479d76a6a077b142636fba9d1de58c7e06
git -C hiking-in-the-wild-sim2sim apply ../docs/hiking_scene.patch
```

补丁保存本地短楼梯场景修改。外部仓库 README 中的旧本机说明和旧 DDS 控制器的绝对路径修改不复制；本项目使用 `scripts/run_sim2sim.sh` 直接推理。现有本地外部仓库已经应用场景修改，不应再次应用。

MuJoCo 运行仍需自行准备兼容的策略检查点与旁边的参数目录；完整依赖和验证限制见 [sim2sim_mujoco.md](sim2sim_mujoco.md)。训练同款场景与视觉资源位于 `scripts/sim2sim/`。
