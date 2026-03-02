# Finetuning Large Language Models for Quantum Circuit Reasoning
This repository contains code for finetuning large language models (LLMs) to enhance their reasoning capabilities in the context of quantum circuits. The project focuses on improving the performance of LLMs in understanding and analyzing quantum circuits, which is crucial for advancing quantum computing research and applications.

## Repository Structure
- `inference/`: Contains code for performing inference with the finetuned models, including evaluation scripts and utilities.
- `verl/`: Contains LLM finetuning scripts using [Verl](https://github.com/verl-project/verl).
- `scripts/`: Contains bash scripts for automating the data processing, model training, and evaluation workflows.

### LLM Finetuning
The finetuning process is implemented using Verl.

Corresponding scripts for:
 
 - [Data Preprocessing](verl/examples/data_preprocess/state_pred.py)
 - [Reward function](verl/verl/utils/reward_score/state_pred.py)