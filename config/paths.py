"""Project-wide path constants."""
import os
from pathlib import Path

HOME = os.environ.get("HOME")
PROJECT_DIR = os.path.join(HOME, "llm_4_qc")

# Repo-relative default so it works regardless of clone location.
# Override with the EVAL_RESULTS_DIR env var when running on a remote cluster.
_REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = os.environ.get("EVAL_RESULTS_DIR", str(_REPO_ROOT / "data" / "results"))

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
QWEN_8B_DIR = os.path.join(MODELS_DIR, "Qwen/Qwen3-8B")
NON_PARAMETERISED_SFT_DIR = os.path.join(DATA_DIR, "non_parametric_sets/sft_datasets")
PARAMETERISED_SFT_DIR = os.path.join(DATA_DIR, "parametric_sets/sft_datasets")
GROVER_SFT_DIR = os.path.join(DATA_DIR, "grover_gpt_replication/sft_datasets")
ROTATION_SFT_DIR = os.path.join(DATA_DIR, "rotation_sets/sft_datasets")
DATASET_GENERATION_DIR = os.path.join(PROJECT_DIR, "dataset_generation")
VISUALIZATION_DIR = os.path.join(PROJECT_DIR, "visualization")
FIG_DIR = os.path.join(VISUALIZATION_DIR, "figures")
