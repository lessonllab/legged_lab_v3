"""One-factor-at-a-time inference interventions with a common 28 second horizon."""
import json
import os
from pathlib import Path
import subprocess
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/terrain_eval/2026-09-08/diagnosis"
cases = [
    ("slope32-baseline", 32, []),
    ("slope32-heading", 32, ["--heading-feedback"]),
    ("slope32-speed05", 32, ["--speed", ".5"]),
    ("slope32-friction04", 32, ["--friction", ".4"]),
    ("slope32-friction12", 32, ["--friction", "1.2"]),
    ("slope28-baseline", 28, []),
    ("slope28-heading", 28, ["--heading-feedback"]),
]
parser = argparse.ArgumentParser()
parser.add_argument("--corrected-heading", action="store_true")
parser.add_argument("--confirmation", action="store_true")
args = parser.parse_args()
if args.corrected_heading:
    cases = [("slope32-heading-corrected", 32, ["--heading-feedback"]),
             ("slope28-heading-corrected", 28, ["--heading-feedback"])]
if args.confirmation:
    cases = [("slope32-heading-friction12", 32, ["--heading-feedback", "--friction", "1.2"]),
             ("slope28-baseline-seed43", 28, ["--seed", "43"]),
             ("slope28-heading-seed43", 28, ["--heading-feedback", "--seed", "43"])]
env = os.environ.copy()
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    env.pop(key, None)
env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
for name, angle, extra in cases:
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "results.json").exists():
        print("EXISTS", name, flush=True)
        continue
    cmd = [sys.executable, str(ROOT / "scripts/experiments/evaluate_terrain.py"),
           "--terrain", "slope", "--slope-deg", str(angle),
           "--checkpoint", str(ROOT / "logs/rsl_rl/g1_amp_rough/2026-08-06_08-53-15/model_26800.pt"),
           "--output", str(dest), "--num-envs", "16", "--steps", "1400", "--continue-after-corridor", "--telemetry", *extra]
    (dest / "command.json").write_text(json.dumps(cmd, indent=2))
    print("START", name, flush=True)
    with (dest / "stdout.log").open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=360)
    if result.returncode:
        print("FAILED", name, result.returncode, flush=True)
        sys.exit(result.returncode)
    data = json.loads((dest / "results.json").read_text())
    print("DONE", name, "success", data["successes"], "route", data["route_successes"],
          "outcomes", [t["outcome"] for t in data["trials"]], flush=True)
