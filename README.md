# Finetuning Large Language Models for Quantum Circuit Reasoning
This repository contains code for finetuning large language models (LLMs) to enhance their reasoning capabilities in the context of quantum circuits. The project focuses on improving the performance of LLMs in understanding and analyzing quantum circuits, which is crucial for advancing quantum computing research and applications.

## Repository Structure
- `dataset_generation/`: Contains scripts for generating datasets of quantum circuits and their corresponding reasoning traces.
- `config/`: Contains paths and constants used across the project.
- `inference/`: Contains code for performing inference with the finetuned models, including evaluation scripts and utilities.
- `verl/`: Contains LLM finetuning scripts using [Verl](https://github.com/verl-project/verl).
- `visualization/`: Contains scripts for visualizing the results of the finetuning and evaluation processes, such as plotting metrics and generating reports.

## Getting Started
To replicate the finetuning process, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/kat-s-626/llm_4_qc.git
    ```
2. Install the required dependencies:
   ```bash
   pip install -r verl/requirements.txt
   ```
3. Prepare your dataset and run LLM finetuning with Verl:
     - [Data Preprocessing](verl/examples/data_preprocess/state_pred.py)
         - Processes raw quantum circuit jsonl files and prepares parquet files for Verl SFT training.
     - [Reward function](verl/verl/utils/reward_score/state_pred.py)
         - Defines the reward function used by GRPO to evaluate predicted quantum states against ground truth (format + MAE).
     - [SFT Training](verl/examples/sft/run_sft_experiment.sh)
         - Launches SFT trainer with Verl; edit experiment config and run to start SFT training and save checkpoints/logs.
     - [GRPO Training](verl/examples/grpo_trainer/run_grpo_experiment.sh)
         - Launches GRPO trainer with Verl; edit experiment config and run to start GRPO training and save checkpoints/logs.

Model used for fine-tuning: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) ([Qwen Team, 2025](https://arxiv.org/abs/2505.09388)) with special tokens:
-  `<circuit_reasoning>` and `</circuit_reasoning>` to denote the start and end of the reasoning process; 
- `<quantum_state>` and `</quantum_state>` to denote the start and end of the quantum state representation.