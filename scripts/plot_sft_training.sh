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

logs/non_parameterized_set_sftgrop_sft

mkdir -p "$parsed_logs/non_parameterized_set_sftgrpo_sft"

python -m visualization.utils.train_valid_loss_log_parser \
    --log-dir "$logs/non_parameterized_set_sftgrpo_sft" \
    --output-dir "$parsed_logs/non_parameterized_set_sftgrpo_sft" \

python -m visualization.plot_sft_train \
    --csv-path "$parsed_logs/non_parameterized_set_sftgrpo_sft/train_valid_loss_aggregated.csv" \
    --output-dir "$figs/non_parameterized_set_sftgrpo_sft" \