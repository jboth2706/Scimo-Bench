#!/usr/bin/env bash
# Entry point: forwards all arguments as Hydra overrides (see README.md). Runs from the
# caller's directory, so relative paths (e.g. models_file=) are relative to where you are.
# SCIMO_BENCH_ENV picks the pixi environment (default: `default`); running models locally
# (site=local steps=[predict,...]) needs the model env: SCIMO_BENCH_ENV=perf ./run.sh ...
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The quality path is single-threaded numpy; an OpenBLAS pool sized for all cores of a
# shared node (whose cgroup grants a few) makes every BLAS call crawl.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
pixi run --manifest-path "$REPO/pixi.toml" -e "${SCIMO_BENCH_ENV:-default}" python -m scimo_bench "$@"
