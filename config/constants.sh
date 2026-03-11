#!/bin/bash

# project paths

export verl="$HOME/rl_experiment/verl"
export inference="$HOME/inference"
export data="$HOME/data"
export random_generator="$HOME/random_circuit_generator"

# venv
export venv="venv/bin/activate"

# data paths
export grover_set_path="$data/grover_sets"
export results_path="$data/results"
export train_10_20="$data/grover_sets/sft_datasets/train_filtered_by_gate_count_10_20.parquet"
export grover_set_grpo="$data/grover_sets/grpo_datasets"

# model paths
export qwen_8b_special_sft_model_grover="$HOME/rl_experiment/verl/merged_models/groverset_sft_17500/qwen3_8b_special"
export qwen_8b_special_sft_model_rotation="$HOME/rl_experiment/verl/merged_models/rotationset_sft_17500/qwen3_8b_special"
export qwen_8b_base="$HOME/rl_experiment/verl/Qwen/Qwen3-8B"
export qwen_8b_special="$HOME/rl_experiment/verl/Qwen/Qwen3-8B-special"
export gptoss_120b="$HOME/rl_experiment/verl/gpt-oss-120b"

export grpo_grover_stage_1_97="$HOME/rl_experiment/verl/merged_models/grpo_qwen3_grover_set_stage_1_97"
export grpo_grover_stage_1_194="$HOME/rl_experiment/verl/merged_models/grpo_qwen3_grover_set_stage_1_194"

# dir names
export qwen_8b_special_dir="qwen3_8b_special"
export gptoss_120b_dir="baseline_gptoss"
export qwen_8b_base_dir="baseline_qwen3_8b"