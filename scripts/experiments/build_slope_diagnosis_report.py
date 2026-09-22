"""Build the human-readable report and embedded-data canvas from validated results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/terrain_eval/2026-09-08/diagnosis"
rows = json.loads((OUT / "summary.json").read_text())
def describe(r):
    p = r["protocol"]
    return f'{p["slope_deg"]}°，seed={p["seed"]}，速度 {p["speed_m_s"]} m/s，机器人摩擦 {p["robot_friction"]}，航向反馈{"开" if p["heading_feedback"] else "关"}'

lines = ["# G1 陡坡原因对照 — 2026-09-08", "",
         "固定 model_26800.pt，比较航向反馈、速度和摩擦。先做单因素对照，再补充航向与摩擦联合条件及另一环境随机种子。没有重新训练或改动模型权重。", "",
         f"有效组数：{len(rows)}；有效首次试验：{16*len(rows)}。每组 16 个环境是同一策略的初始扰动样本，不是独立训练种子。", "", "## 关键发现", ""]
by_name = {r["name"]:r for r in rows}
if "slope28-heading-corrected" in by_name:
    lines.append(f'28°坡道，seed=42：加入航向反馈后，到达终点由 {by_name["slope28-baseline"]["successes"]}/16 提高到 {by_name["slope28-heading-corrected"]["successes"]}/16。')
if "slope28-heading-seed43" in by_name:
    lines.append(f'28°坡道，seed=43 成对复测：由 {by_name["slope28-baseline-seed43"]["successes"]}/16 提高到 {by_name["slope28-heading-seed43"]["successes"]}/16。')
if "slope32-heading-friction12" in by_name:
    lines.extend(["", f'32°坡道，seed=42：原条件、单独航向反馈、单独摩擦提高到 1.2 均为 0/16；两者联合达到 {by_name["slope32-heading-friction12"]["successes"]}/16。这说明同一固定策略具备在较好接触条件及方向保持下通过该坡道的能力，但联合条件尚未换环境种子复测。'])
lines.extend(["", "研究问题可收窄为：在原有或更低摩擦条件下，如何利用接触反馈调整步幅、落脚和身体姿态，改善陡坡推进？当前结果为问题定位；还没有证明某种新方法有效，也没有证据证明 AMP 是根因。", "", "## 有效结果", ""])
for r in rows:
    m, o = r["metrics"], r["outcomes"]
    lines.extend([f'### {r["name"]}', "", describe(r), "",
                  f'- 到达 6 米终点：{r["successes"]}/16；全程保持 ±0.8 米通道：{r["route_successes"]}/16。',
                  f'- 退出计分范围：{o.get("left_containment",0)}；达到 28 秒上限：{o.get("horizon_reached",0)}；环境终止：{o.get("terminated",0)}。',
                  f'- 逐试验最大前进距离的平均值：{m["max_forward_m"]:.3f} 米。',
                  f'- 坡段平均绝对航向偏差：{m["slope_mean_abs_heading_deg"]:.3f}°；平均世界坐标前进速度：{m["slope_mean_forward_speed_m_s"]:.3f} m/s。',
                  f'- 支撑时足部质心切向速度：{m["stance_foot_tangent_speed_m_s"]:.3f} m/s；腿部关节近力矩限幅样本占比：{m["leg_joint_time_near_limit_fraction"]*100:.2f}%。',
                  f'- 初始姿态与同角度、同随机种子的基线配对核查：{r["initial_pose_paired"]}。', ""])
lines.extend(["## 如何解释", "",
    "方向控制与地形推进是两个问题。航向反馈只把目标方向保持在世界 x 正方向，未加入横向位置反馈；因此即使方向更准，仍可能横向漂移。", "",
    "32°坡道的基线、单独航向反馈、单独减速、单独提高摩擦均未到达终点。提高摩擦能增加已观察到的最大前进距离，降低摩擦明显增大支撑期足部运动速度。这表明策略对接触条件敏感；尚不能认定摩擦是唯一原因。联合条件应单独查看，不能混同于任一单因素结果。", "",
    "这些结果不能证明 AMP 动作先验导致困难。航向控制是测试接口层面的常规处理，提高摩擦是改变环境条件；二者本身不能包装为论文算法创新。正式研究应在任务接口与摩擦条件固定后，再分析必要的落脚调整、身体姿态和训练约束。", "",
    "训练存档的基础坡度参数为 slope_range=(0,0.4)，源代码定义它为高度变化/水平距离，而不是弧度或角度；0.4 对应单轴倾角约 21.8°。训练地形还有二维形状与 Perlin 扰动，这个数值不能概括所有局部坡度，课程是否达到最高难度也未核查。因此当前 28°、32°坡道不能自动视为训练分布内，失败也可能体现分布变化。依据：训练 params/env.yaml 与 source/legged_lab/legged_lab/terrains/height_field/hf_terrains.py。", "",
    "## 公平比较与测量范围", "",
    "所有有效组最多运行 1400 个控制步（28 秒），包括 0.5 m/s 组的同期限基线。原先 14 秒的地形筛查结果不与本轮直接混算。6 米终点、±1.5 米计分范围、4 米宽地形、2—5 米坡面区间均一致。", "",
    "基线命令为机身前进 1 m/s、横向速度 0、转向速度 0。航向条件使用 wz=clip(wrap(-heading), -1.5,1.5)，增益为 1；修正后从重置第一帧起生效，并逐步断言公式一致。低速条件仅把前进命令改为 0.5 m/s。", "",
    "机器人材料静、动摩擦同时设为 0.4、0.8 或 1.2；地面静、动摩擦保持 1.0，地面合成模式为 multiply，其他材料设置见协议。本轮把实验因子标为机器人材料摩擦值，避免未经所有碰撞材质优先级逐一核查就声称整个接触对的有效值。后续修正和确认运行还保存 robot_materials.npy，并断言材料读回值一致。", "",
    "遥测为每个控制步之前的 50 Hz 样本，只使用 active=True 的首次试验。身体指标限制在躯干前进距离 (2.1,4.9) 米。足部统计限制在足部 x∈(2.05,4.95) 米，并要求沿坡面法向的接触力大于 20 N。各指标先对每次试验求均值，再对试验等权平均；不同条件采到的位置与时长不相同，均值仅用于描述。", "",
    "foot_vel 是踝部连杆质心速度，切向分量包含滚动和转动，不能直接当成脚底接触点的滑移速度。当前安装的 PhysX 后端在 net_forces_w 接口返回 net_normal_forces_w，并输出警告，所以 foot_force 数据只按法向力解读；未计算或声称测得完整摩擦力。telemetry_schema.json 已按后端实际语义注明。", "",
    "力矩来自 applied_torque；隐式执行器可能提供估计值。近限幅指标为腿部关节达到 joint_effort_limits 的 95% 的关节×时间样本比例。该指标不单独证明电机能力不足。", "",
    "## 已排除的调试结果", "",
    "slope32-heading 与 slope28-heading 的初版：发现重置第一帧采样了随机转向命令，第二帧才进入航向反馈，污染了最初的观察历史。两组已标记 valid_evaluation=false；最终统计只使用 heading-corrected 及之后修正的确认组。早期向用户报告的航向数字属于初步结果，以本文件及 summary.json 为准。", "",
    "一次修正试跑在启动阶段因运行时模块过早导入而崩溃，没有产生试验结果。失败日志保存为 slope32-heading-corrected/startup-failure.log；相关导入已移到模拟器启动之后。", "",
    "联合组另经历一次启动停滞，以及一次加入本地定时调用栈诊断后的中途中断；均未生成 results.json，因此不计入有效试验。中断日志为 slope32-heading-friction12/interrupted-diagnostic-run.log。取消诊断包装后，普通运行成功完成；不能仅由中断时机断定底层错误的具体根因。", "",
    "该崩溃曾触发默认诊断上传，自动审批因此拒绝过重跑。后续采用更安全设置：关闭 SimulationApp 的 enable_crashreporter，禁用 crashreporter/enabled、强制诊断上传，并启用 skipOldDumpUpload；安全审查随后允许执行。设置仅作用于本评测进程，没有修改系统安装包。日志只证明触发过上传，不能据此确认此前是否上传成功。", "",
    "## 文件与复现", "",
    "每组保存 command.json、stdout.log、protocol.json、initial_state.json、results.json、trajectory.csv、telemetry.npz 和 telemetry_schema.json。summary.json 为有效汇总，validation.json 为一致性核查。", "",
    "evaluate_terrain_v1_snapshot.py 对应非航向初版有效组；evaluate_terrain_snapshot.py 对应修正后运行。协议中的 evaluator_sha256 与快照匹配；全部检查点哈希一致。", "",
    "在项目根目录使用 /home/ljc/isaaclab/bin/python 运行 scripts/experiments/run_slope_diagnosis.py，--corrected-heading 运行修正航向组，--confirmation 运行联合条件和 seed=43 复测。存在 results.json 的目录会跳过；若要重新采样，使用 evaluate_terrain.py 并指定新的输出目录。", "",
    "汇总顺序：运行 summarize_slope_diagnosis.py，再运行 validate_slope_diagnosis.py，最后运行本报告生成脚本。当前 Isaac Lab / Isaac Sim 软件环境与训练时全部版本尚未严格对齐，结论限于当前固定检查点和仿真设置。"])
(OUT / "README.md").write_text("\n".join(lines)+"\n")
template = Path("/tmp/G1-slope-diagnosis.template")
if template.exists():
    canvas_rows = []
    for row in rows:
        v = {k:row[k] for k in ("name", "successes", "total", "route_successes", "initial_pose_paired", "outcomes", "metrics")}
        v["protocol"] = {k:row["protocol"][k] for k in ("slope_deg", "speed_m_s", "robot_friction", "heading_feedback", "seed")}
        v["trials"] = [{k:t[k] for k in ("env", "outcome", "time_s", "max_forward_m", "position_m")} for t in row["trials"]]
        v["paths"] = [[[round(p[0],4)] for p in trajectory] for trajectory in row["paths"]]
        canvas_rows.append(v)
    Path("/tmp/G1-slope-diagnosis.canvas.tsx").write_text(template.read_text().replace("__DATA__", json.dumps(canvas_rows, ensure_ascii=False, separators=(",", ":"))))
print("WROTE", OUT / "README.md")
