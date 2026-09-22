"""Aggregate saved boundary screening without treating censored runs as falls."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/terrain_eval/2026-09-08/boundary"
rows = []
for path in sorted(OUT.glob("*/results.json")):
    result = json.loads(path.read_text())
    if not result.get("valid_evaluation"):
        continue
    p = result["protocol"]
    t = result["trials"]
    trajectory = np.loadtxt(path.parent / "trajectory.csv", delimiter=",", skiprows=1)
    initial = json.loads((path.parent / "initial_state.json").read_text())
    reference_path = OUT / "flat/initial_state.json"
    reference = json.loads(reference_path.read_text())
    paired = p["seed"] == 42 and np.array_equal(initial["joint_pos"], reference["joint_pos"]) and np.array_equal(
        np.array(initial["root_state"])[:, 3:], np.array(reference["root_state"])[:, 3:])
    rows.append(dict(name=path.parent.name, terrain=p["terrain"], seed=p["seed"],
                     slope_deg=p["slope_deg"], step_height_m=p["step_height_m"],
                     total=result["total"], successes=result["successes"],
                     route_successes=result["route_successes"],
                     outcomes=dict(Counter(r["outcome"] for r in t)),
                     termination_terms=dict(Counter(term for r in t for term in r.get("termination_terms", []))),
                     corridor_departures=sum(r["first_corridor_departure"] is not None for r in t),
                     same_initial_pose_as_seed42_flat=bool(paired),
                     video_env=p["video_env"] if list(path.parent.glob("*.mp4")) else None,
                     max_forward_progress_m=[float(trajectory[trajectory[:, 1] == i, 2].max()) for i in range(result["total"])],
                     trials=t, paths=[trajectory[trajectory[:, 1] == i][::5, 2:5].tolist() for i in range(result["total"])]))
rows.sort(key=lambda r: (r["seed"] != 42 or r["video_env"] is not None,
                         {"flat": 0, "slope": 1, "stairs": 2}[r["terrain"]],
                         r["slope_deg"] if r["terrain"] == "slope" else r["step_height_m"], r["name"]))
(OUT / "summary.json").write_text(json.dumps(rows, indent=2))
for r in rows:
    print(r["name"], f'{r["successes"]}/{r["total"]}', 'route', r["route_successes"], r["outcomes"], r["termination_terms"])
