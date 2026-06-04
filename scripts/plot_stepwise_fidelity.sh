#!/usr/bin/env bash
# plot_stepwise_fidelity.sh
#
# Regenerates test_1 step-wise fidelity figures for both sets:
#
#   non_parametric_set_test1_stepwise_fidelity.png
#       plot_step_by_step.py         ← grover_sets/sft/test_1_summary.txt
#
#   parametric_set_test1_stepwise_fidelity.png
#       plot_step_by_step.py         ← random_sets/sft/test_1_summary.txt
#
#   combined_test1_stepwise_fidelity.png
#       plot_step_by_step_combined.py ← both sets side by side, shared legend
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

# Single shared tmpdir – cleaned up on exit so all three steps can share CSVs.
TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT

echo "=== Step-wise fidelity plots ==="
echo "  GROVER_SFT_DIR : $GROVER_SFT_DIR"
echo "  RANDOM_SFT_DIR : $RANDOM_SFT_DIR"
echo "  FIG_DIR        : $FIG_DIR"
echo

# ── 1. non_parametric_set_test1_stepwise_fidelity.png ────────────────────────
echo "[1/3] non_parametric_set_test1_stepwise_fidelity.png"
mkdir -p "$TMPDIR_WORK/grover"
cp "$GROVER_SFT_DIR/test_1_summary.txt" "$TMPDIR_WORK/grover/"

"$PYTHON" -m visualization.utils.sft_output_parser \
    --log-dir    "$TMPDIR_WORK/grover" \
    --output-dir "$TMPDIR_WORK" \
    --csv-file   non_parametric_test1_stepwise.csv

"$PYTHON" -m visualization.plot_step_by_step \
    --csv-path   "$TMPDIR_WORK/non_parametric_test1_stepwise.csv" \
    --output-dir "$FIG_DIR" \
    --output-file non_parametric_set_test1_stepwise_fidelity.png

echo "  -> $FIG_DIR/non_parametric_set_test1_stepwise_fidelity.png"

# ── 2. parametric_set_test1_stepwise_fidelity.png ────────────────────────────
echo "[2/3] parametric_set_test1_stepwise_fidelity.png"
mkdir -p "$TMPDIR_WORK/random"
cp "$RANDOM_SFT_DIR/test_1_summary.txt" "$TMPDIR_WORK/random/"

"$PYTHON" -m visualization.utils.sft_output_parser \
    --log-dir    "$TMPDIR_WORK/random" \
    --output-dir "$TMPDIR_WORK" \
    --csv-file   parametric_test1_stepwise.csv

"$PYTHON" -m visualization.plot_step_by_step \
    --csv-path   "$TMPDIR_WORK/parametric_test1_stepwise.csv" \
    --output-dir "$FIG_DIR" \
    --output-file parametric_set_test1_stepwise_fidelity.png

echo "  -> $FIG_DIR/parametric_set_test1_stepwise_fidelity.png"

# ── 3. combined_test1_stepwise_fidelity.png ───────────────────────────────────
echo "[3/3] combined_test1_stepwise_fidelity.png"
"$PYTHON" -m visualization.plot_step_by_step_combined \
    --non-param-csv "$TMPDIR_WORK/non_parametric_test1_stepwise.csv" \
    --param-csv     "$TMPDIR_WORK/parametric_test1_stepwise.csv" \
    --output-dir    "$FIG_DIR" \
    --output-file   combined_test1_stepwise_fidelity.png

echo "  -> $FIG_DIR/combined_test1_stepwise_fidelity.png"

echo
echo "Done."
