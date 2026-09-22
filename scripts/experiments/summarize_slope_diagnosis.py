"""Equal-trial summaries; COM motion during stance is a slip proxy, not contact-point slip."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/terrain_eval/2026-09-08/diagnosis"
rows = []
for f in sorted(OUT.glob("*/results.json")):
    result = json.loads(f.read_text())
    if not result.get("valid_evaluation") or not (f.parent / "telemetry.npz").exists():
        continue
    protocol = result["protocol"]
    a = np.load(f.parent / "telemetry.npz")
    schema = json.loads((f.parent / "telemetry_schema.json").read_text())
    theta = np.deg2rad(protocol["slope_deg"])
    tangent, normal = np.array([np.cos(theta), 0, np.sin(theta)]), np.array([-np.sin(theta), 0, np.cos(theta)])
    contact_normal_force = a["foot_force"] @ normal
    tangent_speed = np.sqrt((a["foot_vel"] @ tangent)**2 + a["foot_vel"][..., 1]**2)
    stance = (contact_normal_force > 20) & (a["foot_pos"][..., 0] > 2.05) & (a["foot_pos"][..., 0] < 4.95) & a["active"][..., None]
    slope = (a["root_pos"][..., 0] > 2.1) & (a["root_pos"][..., 0] < 4.9) & a["active"]
    q = a["root_quat"]
    tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * (q[..., 0]**2 + q[..., 1]**2), -1, 1)))
    leg_ids = [i for i, n in enumerate(schema["joints"]) if any(k in n for k in ("hip", "knee", "ankle"))]
    effort_limits = a["effort_limit"][..., leg_ids]
    assert np.isfinite(effort_limits).all() and (effort_limits > 0).all()
    effort_ratio = np.abs(a["torque"][..., leg_ids]) / effort_limits
    trials = []
    for i, record in enumerate(result["trials"]):
        active = a["active"][:, i]
        on_slope = slope[:, i]
        valid_stance = stance[:, i]
        velocities = tangent_speed[:, i][valid_stance]
        x = a["root_pos"][:, i, 0][active]
        trial = dict(record, max_forward_m=max(float(x.max()), record["position_m"][0]),
                     slope_samples=int(on_slope.sum()), stance_foot_samples=int(valid_stance.sum()),
                     stance_foot_tangent_speed_m_s=float(velocities.mean()) if velocities.size else None,
                     stance_foot_motion_gt02_fraction=float((velocities > .2).mean()) if velocities.size else None,
                     slope_mean_abs_heading_deg=float(np.rad2deg(np.abs(a["heading"][:, i][on_slope])).mean()) if on_slope.any() else None,
                     slope_mean_tilt_deg=float(tilt[:, i][on_slope].mean()) if on_slope.any() else None,
                     slope_mean_forward_speed_m_s=float(a["root_vel_w"][:, i, 0][on_slope].mean()) if on_slope.any() else None,
                     leg_joint_time_near_limit_fraction=float((effort_ratio[:, i][on_slope] >= .95).mean()) if on_slope.any() else None)
        trials.append(trial)
    metric_names = ["max_forward_m", "stance_foot_tangent_speed_m_s", "stance_foot_motion_gt02_fraction",
                    "slope_mean_abs_heading_deg", "slope_mean_tilt_deg", "slope_mean_forward_speed_m_s", "leg_joint_time_near_limit_fraction"]
    metrics = {k: float(np.mean([t[k] for t in trials if t[k] is not None])) if any(t[k] is not None for t in trials) else None for k in metric_names}
    initial = json.loads((f.parent / "initial_state.json").read_text())
    suffix = "" if protocol["seed"] == 42 else f'-seed{protocol["seed"]}'
    baseline_path = OUT / f'slope{int(protocol["slope_deg"])}-baseline{suffix}' / "initial_state.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    paired = baseline is not None and np.array_equal(initial["joint_pos"], baseline["joint_pos"]) and np.array_equal(
        np.array(initial["root_state"])[:, 3:], np.array(baseline["root_state"])[:, 3:])
    if paired:
        paired = np.allclose(np.array(initial["root_state"])[:, :3] - initial["origins"],
                             np.array(baseline["root_state"])[:, :3] - baseline["origins"], atol=1e-7)
    paths = [a["root_pos"][:, i][a["active"][:, i]][::5].tolist() for i in range(result["total"])]
    rows.append(dict(name=f.parent.name, protocol=protocol, successes=result["successes"], total=result["total"],
                     route_successes=result["route_successes"], outcomes=dict(Counter(t["outcome"] for t in trials)),
                     metrics=metrics, trials=trials, paths=paths, initial_pose_paired=bool(paired)))
for row in rows:
    print(row["name"], row["successes"], row["outcomes"], "paired", row["initial_pose_paired"],
          "metrics", {k: round(v, 3) if v is not None else None for k, v in row["metrics"].items()})
(OUT / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
