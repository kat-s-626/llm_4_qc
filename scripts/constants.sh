#!/bin/bash

# project paths
export project_root="$HOME/llm_4_qc"
export dataset_generator="$project_root/dataset_generator"
export verl="$project_root/verl"
export inference="$project_root/inference"
export data="$project_root/data"
export visualization="$project_root/visualization"
export logs="$project_root/logs"
export parsed_logs="$visualization/parsed_logs"
export figs="$visualization/figures"
export models="$project_root/models"

# venv
export venv="$project_root/venv/bin/activate"

# data paths
export non_parametric_set_dir="$data/non_parametric_sets"
export parametric_set_dir="$data/parametric_sets"
export grover_gpt_replication_dir="$data/grover_gpt_replication"
export results_dir="$data/results"
export train_10_20="$non_parametric_set_dir/sft_datasets/train_filtered_by_gate_count_10_20.parquet"
export non_parametric_set_grpo="$non_parametric_set_dir/grpo_datasets"

# model paths
export qwen_8b_special_sft_model_non_parametric="$project_root/verl/merged_models/non_parametric_set_sft_17500/qwen3_8b_special"
export qwen_8b_special_sft_model_parametric="$project_root/verl/merged_models/parametric_set_sft_17500/qwen3_8b_special"
export qwen_8b_base="$project_root/verl/Qwen/Qwen3-8B"
export qwen_8b_special="$project_root/verl/Qwen/Qwen3-8B-special"
export gptoss_120b="$project_root/verl/gpt-oss-120b"

export grpo_non_parametric_stage_1_97="$project_root/verl/merged_models/grpo_qwen3_non_parametric_set_stage_1_97"
export grpo_non_parametric_stage_1_194="$project_root/verl/merged_models/grpo_qwen3_non_parametric_set_stage_1_194"

# dir names
export qwen_8b_special_dir="qwen3_8b_special"
export gptoss_120b_dir="baseline_gptoss"
export qwen_8b_base_dir="baseline_qwen3_8b"