#!/bin/bash
source constants.sh

# Generate jsonl datasets of random circuits
cd "$project_root" || {
    echo "Error: Failed to change directory to $project_root."
    exit 1
}

source $venv || {
    echo "Error: Failed to activate virtual environment at $venv."
    exit 1
}

# mkdir -p "$parsed_logs/non_parametric_set_stage_1_mae"

# python -m visualization.utils.grpo_log_parser \
#     --log-dir "$logs/non_parametric_set_stage_1_mae" \
#     --output-dir "$parsed_logs/non_parametric_set_stage_1_mae" \

# python -m visualization.plot_grpo_train \
#     --csv-path "$parsed_logs/non_parametric_set_stage_1_mae/grpo_metrics_aggregated.csv" \
#     --output-dir "$figs/non_parametric_set_stage_1_mae" \

mkdir -p "$parsed_logs/parametric_set_sftgrpo"

python -m visualization.utils.grpo_log_parser \
    --log-dir "$logs/parametric_set_sftgrpo" \
    --output-dir "$parsed_logs/parametric_set_sftgrpo" \

python -m visualization.plot_grpo_train \
    --csv-path "$parsed_logs/parametric_set_sftgrpo/grpo_metrics_aggregated.csv" \
    --output-dir "$figs/parametric_set_sftgrpo" \