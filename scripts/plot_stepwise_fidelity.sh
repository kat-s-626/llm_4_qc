#!/usr/bin/env bash
# plot_stepwise_fidelity.sh
#
# Regenerates test_1 step-wise fidelity figures for both sets:
#
#   non_parametric_set_test1_stepwise_fidelity.png
#       plot_step_by_step.py  ← grover_sets/sft/test_1_summary.txt
#
#   parametric_set_test1_stepwise_fidelity.png
#       plot_step_by_step.py  ← random_sets/sft/test_1_summary.txt
#
# Usage (from repo root):
#   bash scripts/plot_stepwise_fidelity.sh
#
# Optional overrides (env vars):
#   GROVER_SFT_DIR   – directory with grover_sets SFT summary txts
#   RANDOM_SFT_DIR   – directory with random_sets SFT summary txts
#   FIG_DIR          – output directory for figures

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GROVER_SFT_DIR="${GROVER_SFT_DIR:-/scratch3/ip004/data/results/eval/grover_sets/sft}"
RANDOM_SFT_DIR="${RANDOM_SFT_DIR:-/scratch3/ip004/data/results/eval/random_sets/sft}"
FIG_DIR="${FIG_DIR:-visualization/figures}"

PYTHON="$(source venv/bin/activate 2>/dev/null; which python)"

echo "=== Step-wise fidelity plots ==="
echo "  GROVER_SFT_DIR : $GROVER_SFT_DIR"
echo "  RANDOM_SFT_DIR : $RANDOM_SFT_DIR"
echo "  FIG_DIR        : $FIG_DIR"
echo

# ── 1. non_parametric_set_test1_stepwise_fidelity.png ────────────────────────
# Parse just test_1_summary.txt for the grover (non-parameterized) set, then plot.
echo "[1/2] non_parametric_set_test1_stepwise_fidelity.png"
TMPDIR_GROVER="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_GROVER"' EXIT

cp "$GROVER_SFT_DIR/test_1_summary.txt" "$TMPDIR_GROVER/"

"$PYTHON" -m visualization.utils.sft_output_parser \
    --log-dir    "$TMPDIR_GROVER" \
    --output-dir "$TMPDIR_GROVER" \
    --csv-file   non_parametric_test1_stepwise.csv

"$PYTHON" -m visualization.plot_step_by_step \
    --csv-path   "$TMPDIR_GROVER/non_parametric_test1_stepwise.csv" \
    --output-dir "$FIG_DIR" \
    --output-file non_parametric_set_test1_stepwise_fidelity.png

echo "  -> $FIG_DIR/non_parametric_set_test1_stepwise_fidelity.png"

# ── 2. parametric_set_test1_stepwise_fidelity.png ────────────────────────────
# Parse just test_1_summary.txt for the random (parameterized) set, then plot.
echo "[2/2] parametric_set_test1_stepwise_fidelity.png"
TMPDIR_RANDOM="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RANDOM"' EXIT

cp "$RANDOM_SFT_DIR/test_1_summary.txt" "$TMPDIR_RANDOM/"

"$PYTHON" -m visualization.utils.sft_output_parser \
    --log-dir    "$TMPDIR_RANDOM" \
    --output-dir "$TMPDIR_RANDOM" \
    --csv-file   parametric_test1_stepwise.csv

"$PYTHON" -m visualization.plot_step_by_step \
    --csv-path   "$TMPDIR_RANDOM/parametric_test1_stepwise.csv" \
    --output-dir "$FIG_DIR" \
    --output-file parametric_set_test1_stepwise_fidelity.png

echo "  -> $FIG_DIR/parametric_set_test1_stepwise_fidelity.png"

echo
echo "Done."
