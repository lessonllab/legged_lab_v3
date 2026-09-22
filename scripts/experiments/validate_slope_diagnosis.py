"""Validate protocol, sample pairing, commands, telemetry, and terminal accounting."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/terrain_eval/2026-09-08/diagnosis"
hashes = {hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob("evaluate_terrain*snapshot.py")}
rows = json.loads((OUT / "summary.json").read_text())
reports = []
for r in rows:
    p = OUT / r["name"]
    raw = json.loads((p / "results.json").read_text())
    assert raw["valid_evaluation"] and r["initial_pose_paired"]
    assert r["protocol"]["evaluator_sha256"] in hashes
    assert len(r["trials"]) == r["total"] == 16
    assert sum(r["outcomes"].values()) == 16
    assert r["successes"] == sum(t["outcome"] == "success" for t in r["trials"])
    a = np.load(p / "telemetry.npz")
    for k in a.files:
        assert np.isfinite(a[k]).all(), (r["name"], k)
    trajectory = np.loadtxt(p / "trajectory.csv", delimiter=",", skiprows=1)
    assert len(trajectory) == a["active"].sum()
    for t in raw["trials"]:
        i = t["env"]
        active = a["active"][:, i]
        n = int(active.sum())
        assert active[:n].all() and not active[n:].any()
        assert abs(n * .02 - t["time_s"]) < 1e-8
        assert np.allclose(trajectory[trajectory[:, 1] == i][-1, 2:5], t["position_m"])
    cmd = a["command"]
    assert np.allclose(cmd[..., 0][a["active"]], r["protocol"]["speed_m_s"])
    assert np.allclose(cmd[..., 1][a["active"]], 0)
    if r["protocol"]["heading_feedback"]:
        expected = np.clip(np.arctan2(np.sin(-a["heading"]), np.cos(-a["heading"])), -1.5, 1.5)
        assert np.allclose(cmd[..., 2][a["active"]], expected[a["active"]], atol=2e-6)
    else:
        assert np.allclose(cmd[..., 2][a["active"]], 0)
    materials = p / "robot_materials.npy"
    if materials.exists():
        assert np.allclose(np.load(materials)[..., :2], r["protocol"]["robot_friction"])
    reports.append(dict(name=r["name"], trials=16, valid=True, first_frame_command_checked=True,
                        material_readback_checked=materials.exists(), evaluator_hash_checked=True))
assert len({r["protocol"]["checkpoint_sha256"] for r in rows}) == 1
(OUT / "validation.json").write_text(json.dumps(dict(cases=reports, valid_trials=16*len(reports)), indent=2))
print("VALIDATED", len(reports), "cases", len(reports)*16, "trials")
