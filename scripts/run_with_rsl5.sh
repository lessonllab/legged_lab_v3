#!/usr/bin/env bash
# Run this project with RSL-RL 5 without replacing another project's installation.
# Activate the desired Isaac Lab environment first, or set ISAACLAB_PYTHON.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$project_dir/.runtime/rsl_rl_5_4_1"
python_bin="${ISAACLAB_PYTHON:-python}"

if [[ ! -f "$runtime_dir/rsl_rl/models/cnn_model.py" ]]; then
    printf 'Install the isolated runtime first:\n%s -m pip install --no-deps --target "%s" rsl-rl-lib==5.4.1\n' \
        "$python_bin" "$runtime_dir" >&2
    exit 2
fi
export PYTHONPATH="$runtime_dir:$project_dir/source/legged_lab${PYTHONPATH:+:$PYTHONPATH}"
# Kit's platform-info worker can fork during startup. The bundled OpenBLAS
# thread-pool shutdown hook crashes there (exit 139); set this before Python
# imports NumPy/SciPy. This changes only the launched process and its children.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "$project_dir"
exec "$python_bin" "$@"
