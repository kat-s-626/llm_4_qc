# Finetuning Large Language Models for Quantum Circuit Reasoning
This repository contains code for finetuning large language models (LLMs) to enhance their reasoning capabilities in the context of quantum circuits. The project focuses on improving the performance of LLMs in understanding and analyzing quantum circuits, which is crucial for advancing quantum computing research and applications.

## Repository Structure
```text
llm_4_qc/
├── config/              # Shared constants and path configuration
├── dataset_generator/   # Dataset generation and preprocessing pipeline
├── inference/           # Inference and evaluation scripts
├── scripts/             # Shell workflows 
├── visualization/       # Plotting and result-analysis scripts
└── verl/                # Verl framework and training recipes
    └── experiments/       # Scripts for SFT and GRPO experiments
```

## Getting Started
To replicate the finetuning process, follow these steps:
1. Install the required dependencies:
   ```bash
   pip install -r verl/requirements.txt
   ```
2. Generate datasets of quantum circuits and their reasoning traces. This script will create training and testing parquets files that can be used for verl training:
    ```bash
    bash scripts/generate_dataset.sh
    ```

3. LLM finetuning:
     - [Reward function](verl/verl/utils/reward_score/state_pred.py)
         - Defines the reward function used by GRPO to evaluate predicted quantum states against ground truth (format + MAE).
     - [SFT Training](verl/examples/sft/run_sft_experiment.sh)
         - Launches SFT trainer with Verl; edit experiment config and run to start SFT training and save checkpoints/logs.
     - [GRPO Training](verl/examples/grpo_trainer/run_grpo_experiment.sh)
         - Launches GRPO trainer with Verl; edit experiment config and run to start GRPO training and save checkpoints/logs.

Model used for fine-tuning: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) ([Qwen Team, 2025](https://arxiv.org/abs/2505.09388)) with special tokens:
-  `<circuit_reasoning>` and `</circuit_reasoning>` to denote the start and end of the reasoning process; 
- `<quantum_state>` and `</quantum_state>` to denote the start and end of the quantum state representation.