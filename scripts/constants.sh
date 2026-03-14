#!/bin/bash

# project paths
export project_root="$HOME/llm_4_qc"
export dataset_generator="$project_root/dataset_generator"
export verl="$project_root/verl"
export inference="$project_root/inference"
export data="$project_root/data"

# venv
export venv="$project_root/venv/bin/activate"

# data paths
export grover_set_dir="$data/grover_sets"
export random_set_dir="$data/random_sets"
export results_dir="$data/results"
export train_10_20="$grover_set_dir/sft_datasets/train_filtered_by_gate_count_10_20.parquet"
export grover_set_grpo="$grover_set_dir/grpo_datasets"

# model paths
export qwen_8b_special_sft_model_grover="$project_root/verl/merged_models/groverset_sft_17500/qwen3_8b_special"
export qwen_8b_special_sft_model_rotation="$project_root/verl/merged_models/rotationset_sft_17500/qwen3_8b_special"
export qwen_8b_base="$project_root/verl/Qwen/Qwen3-8B"
export qwen_8b_special="$project_root/verl/Qwen/Qwen3-8B-special"
export gptoss_120b="$project_root/verl/gpt-oss-120b"

export grpo_grover_stage_1_97="$project_root/verl/merged_models/grpo_qwen3_grover_set_stage_1_97"
export grpo_grover_stage_1_194="$project_root/verl/merged_models/grpo_qwen3_grover_set_stage_1_194"

# dir names
export qwen_8b_special_dir="qwen3_8b_special"
export gptoss_120b_dir="baseline_gptoss"
export qwen_8b_base_dir="baseline_qwen3_8b"