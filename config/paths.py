"""Project-wide path constants."""
import os
from pathlib import Path

HOME = os.environ.get("HOME")
PROJECT_DIR = os.path.join(HOME, "llm_4_qc")

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
QWEN_8B_DIR = os.path.join(MODELS_DIR, "Qwen/Qwen3-8B")
GROVER_SFT_DIR = os.path.join(DATA_DIR, "grover_sets/sft_datasets")
ROTATION_SFT_DIR = os.path.join(DATA_DIR, "rotation_sets/sft_datasets")
RANDOM_SFT_DIR = os.path.join(DATA_DIR, "random_sets/sft_datasets")
DATASET_GENERATION_DIR = os.path.join(PROJECT_DIR, "dataset_generation")
