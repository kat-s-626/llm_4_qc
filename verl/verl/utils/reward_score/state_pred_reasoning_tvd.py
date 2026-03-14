import re
import json
import numpy as np
from typing import Dict, Any, Optional

_SOLUTION_CLIP_CHARS = 500
MAE_THRESHOLD = 0.01


def extract_probability_distribution(solution_str: str) -> Optional[Dict[str, float]]:
    """
    Extract the probability distribution from the model's output.
    Looks for JSON format after #### delimiter.
    """

    # Cut after </think> if present
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1]
    else:
        return None

    # Optimization: clip to last N characters for efficiency
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    
    try:
        # Try to find JSON pattern
        json_match = re.search(r'\{[^}]+\}', solution_str)
        if json_match:
            predicted_dist = json.loads(json_match.group())
            
            # Validate that it's a proper probability distribution
            if isinstance(predicted_dist, dict) and all(
                isinstance(k, str) and isinstance(v, (int, float)) and v >= 0
                for k, v in predicted_dist.items()
            ):
                return predicted_dist
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    
    return None


def compute_mae_reward(pred_dist: Dict[str, float], truth_dist: Dict[str, float], threshold: float = 0.01
) -> float:
    if not pred_dist:
        return 0.0
    
    if sum(pred_dist.values()) > 1.01:
        return 0.0
    
    top_15_truth = set(sorted(truth_dist.keys(), key=lambda x: truth_dist[x], reverse=True)[:15])
    num_qubits = len(next(iter(truth_dist.keys()))) if truth_dist else 0
    # get set of all states to consider (predicted + top 15 truth)
    all_states = set(pred_dist.keys()) | top_15_truth
    
    mae = 0.0
    for state in all_states:
        pred_prob = pred_dist.get(state, 0.0)
        true_prob = truth_dist.get(state, 0.0)
        mae += abs(pred_prob - true_prob)
    
    # Average MAE over the number of states
    # State not in the set is assumed to have 0 probability, so we consider all states in the union of predicted and top-15 truth
    mae /= 2 ** num_qubits  
    
    # Binary reward: 1.0 if within threshold, 0.0 otherwise/scratch3/ip004/bash_scripts/eval/eval_grover_grpo_step_1_97.sh
    return 1.0 if mae <= threshold else 0.0

def compute_tvd_reward(pred_dist: Dict[str, float], truth_dist: Dict[str, float], threshold: float = 0.01) -> float:
    if not pred_dist:
        return 0.0
    
    if sum(pred_dist.values()) > 1.01:
        return 0.0
    
    top_15_truth = set(sorted(truth_dist.keys(), key=lambda x: truth_dist[x], reverse=True)[:15])
    # get set of all states to consider (predicted + top 15 truth)
    all_states = set(pred_dist.keys()) | top_15_truth
    
    tvd = 0.0
    for state in all_states:
        pred_prob = pred_dist.get(state, 0.0)
        true_prob = truth_dist.get(state, 0.0)
        tvd += abs(pred_prob - true_prob)
    
    # Total Variation Distance is half the L1 distance
    tvd /= 2
    
    # Binary reward: 1.0 if within threshold, 0.0 otherwise
    return 1.0 if tvd <= threshold else 0.0


def compute_score(data_source: Any, solution_str: str, ground_truth: str, extra_info: Optional[Dict] = None) -> float:
    """
    Compute reward score for quantum circuit simulation with two-level scoring:
    - Level 1: Format score (1.0) if valid distribution extracted
    - Level 2: TVD reward (0-1) only if format is correct
    
    Args:
        data_source: The data source (unused but required by framework)
        solution_str: Model's output including reasoning and probability distribution
        ground_truth: Ground truth as JSON string or dict
        extra_info: Additional information (optional)
    
    Returns:
        Reward score in range [0, 1]
    """
    
    # Parse ground truth if it's a string
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            # If ground_truth is not valid JSON, return 0
            return 0.0
    
    # Extract predicted distribution
    predicted_dist = extract_probability_distribution(solution_str)
    
    # Level 1: Format check - if no valid distribution found, return 0
    if predicted_dist is None:
        return 0.0

    # Compute TVD reward
    tvd_reward = compute_tvd_reward(predicted_dist, ground_truth, threshold=MAE_THRESHOLD)
    
    return tvd_reward
