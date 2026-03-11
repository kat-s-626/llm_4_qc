"""Project-wide path constants."""
import os
from pathlib import Path

home = os.environ.get("HOME")

DATA_DIR = os.path.join(home, "data")
MODELS_DIR = os.path.join(home, "models")
GROVER_SFT_DIR = os.path.join(DATA_DIR, "grover_sets/sft_datasets")
ROTATION_SFT_DIR = os.path.join(DATA_DIR, "rotation_sets/sft_datasets")
DATASET_GENERATION_DIR = Path.cwd() / "dataset_generation"
