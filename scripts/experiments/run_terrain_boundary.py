"""Sequential, bounded screening; each case saves its command and full output."""
import json
import os
from pathlib import Path
import subprocess
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/terrain_eval/2026-09-08/boundary"
parser = argparse.ArgumentParser()
parser.add_argument("--followup", action="store_true")
args = parser.parse_args()
if args.followup:
    cases = [("slope-28", "slope", ["--slope-deg", "28"]),
             ("slope-32-seed43", "slope", ["--slope-deg", "32", "--seed", "43"]),
             ("stairs-32cm", "stairs", ["--step-height", ".32"]),
             ("slope-32-video", "slope", ["--slope-deg", "32", "--video", "--video-env", "5"])]
else:
    cases = [("flat", "flat", [])]
    cases += [(f"slope-{d}", "slope", ["--slope-deg", str(d)]) for d in (8, 16, 24, 32)]
    cases += [(f"stairs-{h:02d}cm", "stairs", ["--step-height", str(h / 100)]) for h in (8, 16, 24)]
env = os.environ.copy()
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    env.pop(key, None)
env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
for name, terrain, extra in cases:
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "results.json").exists():
        print("EXISTS", name, flush=True)
        continue
    cmd = [sys.executable, str(ROOT / "scripts/experiments/evaluate_terrain.py"),
           "--terrain", terrain, "--checkpoint", str(ROOT / "logs/rsl_rl/g1_amp_rough/2026-08-06_08-53-15/model_26800.pt"),
           "--output", str(dest), "--num-envs", "16", "--steps", "700", "--continue-after-corridor", *extra]
    (dest / "command.json").write_text(json.dumps(cmd, indent=2))
    print("START", name, flush=True)
    with (dest / "stdout.log").open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    if result.returncode:
        print("FAILED", name, result.returncode, flush=True)
        sys.exit(result.returncode)
    data = json.loads((dest / "results.json").read_text())
    print("DONE", name, "success", data["successes"], "route", data["route_successes"],
          "outcomes", [t["outcome"] for t in data["trials"]], flush=True)
