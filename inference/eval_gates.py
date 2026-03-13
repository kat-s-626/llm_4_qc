from collections import defaultdict
import pandas as pd
import numpy as np
import json
import re
from transformers import AutoTokenizer 
from typing import Dict, Optional, List, Tuple, Any
import argparse
import os
import sys
import tqdm
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import COMMON_SQRT_VALUES

"""
    Things to evaluate for quantum gates:
    1. Number of tokens used
    2. Fidelity (absolute square of inner product) - per step and aggregated
    3. Classical Fidelity (probability distribution comparison)
    4. F1-score (for probability distribution comparison)
    5. MAE (Mean Absolute Error for probability distributions)
    6. TVD top-k renormalized (Total Variation Distance over top-k states)
    7. Parse success rate
    8. Perfect fidelity rate (fidelity = 1.0)
    9. Format correctness (quantum state format)
    10. Reasoning format correctness (circuit_reasoning structure and sequential steps)
"""

""" 
    Dataframe columns:
    df["responses"] : model generated response
    df["extra_info"]["formatted_completion"] : ground truth quantum state
    df["extra_info"]["msb_measurement_probabilities"] : ground truth probability distribution (full)
"""

FIDELITY_THRESHOLD = 0.99
CLASSICAL_FIDELITY_THRESHOLD = 0.99
MAE_THRESHOLD = 0.005
_SOLUTION_CLIP_CHARS = 500

def extract_probability_distribution_from_json(text: str) -> Optional[Dict[str, float]]:
    """
    Extract the probability distribution from JSON format after </circuit_reasoning> tag.
    This is for extracting the final answer in the format: {"00000": 0.25, "01000": 0.25, ...}
    
    Args:
        text: Text containing probability distribution in JSON format
        
    Returns:
        Dictionary mapping bitstrings to probabilities, or None if parsing fails
    """
    # Optimization: clip to last N characters for efficiency
    solution_str = text
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    
    # Cut after </circuit_reasoning> if present using regex to handle spacing issues
    reasoning_end_pattern = r'</circuit_reasoning>\s*'
    reasoning_end_match = re.search(reasoning_end_pattern, solution_str)
    if reasoning_end_match:
        solution_str = solution_str[reasoning_end_match.end():]
    
    # Strip markdown code fences (e.g. ```json ... ```) before matching
    solution_str = re.sub(r'```(?:json)?\s*', '', solution_str).strip()

    try:
        # Try to find JSON pattern - get the last one if multiple exist
        # re.DOTALL allows { ... } to span multiple lines
        json_matches = re.findall(r'\{[^{}]+\}', solution_str, re.DOTALL)
        if json_matches:
            # Take the last match
            last_json = json_matches[-1]
            predicted_dist = json.loads(last_json)
            
            # Validate that it's a proper probability distribution
            if isinstance(predicted_dist, dict) and all(
                isinstance(k, str) and isinstance(v, (int, float)) and v >= 0
                for k, v in predicted_dist.items()
            ):
                return predicted_dist
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    
    return None

def search_acc(pred_dist: Dict[str, float], marked_states: list) -> float:
    """
    Measures the model's ability to identify marked states correctly.
    Computes the intersection between top-k predicted states and marked states.
    
    Args:
        pred_dist: Predicted probability distribution
        marked_states: List of marked states to check for
        
    Returns:
        Search accuracy value between 0 and 1
    """
    top_k = len(marked_states)
    marked_states = set(marked_states)

    # flip order for each marked state from 001 to 100
    flipped_states = {state[::-1] for state in marked_states}

    print(f"Flipped marked states: {flipped_states}")

    # Get the top-k states from both distributions
    top_pred = set(sorted(pred_dist.keys(), key=lambda x: pred_dist[x], reverse=True)[:len(flipped_states)])

    print(f"Top predicted states: {top_pred}")

    if not top_pred:
        return 0.0

    # Compute the intersection
    intersection = top_pred & flipped_states
    return len(intersection) / top_k

def compute_f1_score(pred_dist: Dict[str, float], truth_dist: Dict[str, float]) -> float:
    """
    Compute F1 score between predicted and ground truth probability distributions.
    Compares the set of bitstrings (keys) in both distributions.
    
    Args:
        pred_dist: Predicted probability distribution
        truth_dist: Ground truth probability distribution
        
    Returns:
        F1 score value between 0 and 1
    """
    if not pred_dist or not truth_dist:
        # print("One of the distributions is empty, returning F1 score of 0.0")
        return 0.0
    
    pred_keys = set(pred_dist.keys())
    truth_keys = set(truth_dist.keys())
    
    if not truth_keys or not pred_keys:
        return 0.0
    
    true_positives = len(pred_keys & truth_keys)
    precision = true_positives / len(pred_keys) if pred_keys else 0.0
    recall = true_positives / len(truth_keys) if truth_keys else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    # print(f"Computed F1 score: {f1} (Precision: {precision}, Recall: {recall})")
    return f1

def compute_mae(pred_dist: Dict[str, float], truth_dist: Dict[str, float], num_qubits) -> float:
    """
    Compute Mean Absolute Error between predicted and ground truth probability distributions.
    
    Args:
        pred_dist: Predicted probability distribution
        truth_dist: Ground truth probability distribution
        
    Returns:
        MAE value (lower is better)
    """
    if not pred_dist or not truth_dist:
        return 1
    
    # Get all bitstrings from both distributions
    all_bitstrings = set(pred_dist.keys()) | set(truth_dist.keys())

    # if the basis length does not match num_qubits, return 1
    for bitstring in all_bitstrings:
        if len(bitstring) != num_qubits:
            return 1
    
    if not all_bitstrings:
        return 1
    
    # Calculate MAE over all states
    mae = 0.0
    for bitstring in all_bitstrings:
        pred_prob = pred_dist.get(bitstring, 0.0)
        true_prob = truth_dist.get(bitstring, 0.0)
        mae += abs(pred_prob - true_prob)
    
    return mae / 2**num_qubits

def tvd_topk_renormalized(pred_dist: Dict[str, float], true_dist: Dict[str, float], k: int = 15) -> float:
    """
    TVD over top-k states after renormalizing both distributions.
    Measures: "How well does the model distribute probability 
    among the most important states?"
    
    Args:
        pred_dist: Predicted probability distribution
        true_dist: Ground truth probability distribution
        k: Number of top states to consider
        
    Returns:
        TVD value (lower is better, range [0, 1])
    """
    # Get union of top-k from both
    all_topk_states = set(pred_dist.keys()) | set(true_dist.keys())
    
    # Renormalize both to sum to 1.0
    pred_sum = sum(pred_dist.values())
    true_sum = sum(true_dist.values())
    
    tvd = 0.0
    for state in all_topk_states:
        p_pred = pred_dist.get(state, 0.0) / pred_sum if pred_sum > 0 else 0.0
        p_true = true_dist.get(state, 0.0) / true_sum if true_sum > 0 else 0.0
        tvd += abs(p_pred - p_true)
    
    return tvd / 2.0

def reasoning_format_accuracy(response: str, ground_truth: str = '', num_qubits: int = 0) -> tuple[float, list[bool]]:
    """
    Check if the reasoning format is correct.
    Returns (overall_accuracy, criteria_list)
    
    Criteria:
    1. Contains <circuit_reasoning>...</circuit_reasoning> tags
    2. Each intermediate reasoning step contains <quantum_state>...</quantum_state> tags
    3. Measurement outcome probability calculation section is present before the final answer
    4. Each quantum state length equals num_qubits**2 (2^num_qubits)
    """
    criteria_list = [False] * 4
   
    criteria_list[0] = len(re.findall(r'</circuit_reasoning>', response)) == 1
    
    criteria_list[1] = len(re.findall(r'<quantum_state>', response)) == len(re.findall(r'</quantum_state>', response))
        
    # Criterion 3: Measurement outcome probability calculation section is present
    # Look for patterns like "The probability distribution of measurement outcome is:"
    # and "|00000>: |1|^2 = 1.000" before the final answer
    prob_keywords = [
        'The probability distribution of measurement outcome is:',
        r'\|[01]+>\s*:.*?=\s*[0-9]*\.?[0-9]+'  # Pattern like |1>: |1|^2 = 1.000
    ]
    
    criteria_list[2] = all(re.search(pattern, response) for pattern in prob_keywords)
    
    # Criterion 4: Each quantum state length equals 2^num_qubits
    # AND the states successfully parse to match ground truth structure
    pred_states = parse_all_quantum_states(response)
    truth_states = parse_all_quantum_states(ground_truth)
    
    if num_qubits > 0 and pred_states and truth_states:
        expected_length = 2 ** num_qubits
        criteria_list[3] = all(
            [len(state) == expected_length for state in pred_states if state is not None]
        )
    else:
        # If states don't parse or don't match structure, this criterion fails
        criteria_list[3] = False
    
    
    return float(all(criteria_list)), criteria_list

def format_accuracy(response: str, num_qubits: int) -> tuple[float, list[bool]]:
    """
    Check if the predicted probability distribution (final output) is well-formed.
    Returns (overall_accuracy, criteria_list)
    
    Criteria:
    1. All basis lengths match num_qubits
    2. Basis format is binary string
    3. No duplicate basis states (inherent in dict)
    4. All probabilities sum in the range [0, 1]
    5. All probabilities non-negative
    6. Probability has ≤ 3 decimal places
    7. No more than 15 entries in the distribution
    """
    criteria_list = [False] * 7

    # First, try to extract just the JSON object to avoid matching other numeric values
    # Look for the last JSON-like pattern in the response
    json_pattern = re.compile(r'\{[^{}]*\}')
    json_matches = json_pattern.findall(response)
    
    # Use the last JSON match if found, otherwise use the whole response
    search_text = json_matches[-1] if json_matches else response

    key_pattern = re.compile(r'"([01]+)"\s*:')  # Match binary string keys
    keys_found = key_pattern.findall(search_text)

    if not keys_found:
        criteria_list[0] = False
        criteria_list[1] = False
        criteria_list[2] = False

    value_pattern = re.compile(r'"[01]+"\s*:\s*([0-9]+\.?[0-9]*)')  # Match numeric values after binary keys
    values_found = value_pattern.findall(search_text)
    if not values_found:
        criteria_list[3] = False
        criteria_list[4] = False
        criteria_list[5] = False

    # Criterion 1: All basis lengths match num_qubits
    criteria_list[0] = all(len(basis) == num_qubits for basis in keys_found)

    # Criterion 2: Basis format is binary string
    criteria_list[1] = all(re.fullmatch(r'[01]+', basis) for basis in keys_found)

    # Criterion 3: No duplicate basis states (check in original response)
    criteria_list[2] = len(keys_found) == len(set(keys_found))

    try:
        values_found = [float(v) for v in values_found]

        # Criterion 4: Probabilities sum in range [0, 1]
        total_prob = sum(values_found)
        criteria_list[3] = 0 <= total_prob <= 1.0001  # Small epsilon for float precision
        
        # Criterion 5: All probabilities non-negative
        criteria_list[4] = all(p >= 0 for p in values_found)

        # Criterion 6: Probability has ≤ 3 decimal places
        criteria_list[5] = all(
            len(str(v).split('.')[-1]) <= 3 if '.' in str(v) else True
            for v in values_found
        )
    except ValueError:
        criteria_list[3] = False
        criteria_list[4] = False
        criteria_list[5] = False
    
    # Criterion 7: No more than 15 entries
    criteria_list[6] = len(keys_found) <= 15
    
    return float(all(criteria_list)), criteria_list

def parse_quantum_state(text: str) -> Optional[List[complex]]:
    """
    Extract quantum state array from text wrapped between <quantum_state> tags.
    Handles formats like: [1/√2, -0.71] or [0, 1] or [-1.00, 0] or arrays of any length
    Also converts symbolic values using COMMON_SQRT_VALUES mapping.
    
    Args:
        text: Text containing quantum state
        
    Returns:
        List of complex numbers representing the quantum state, or None if parsing fails
    """
    # Find content between <quantum_state> tags
    pattern = r'<quantum_state>\s*:?\s*\[(.*?)\]'
    match = re.search(pattern, text)

    if not match:
        return None
    
    state_str = match.group(1)
    
    # Split by comma
    components = [s.strip() for s in state_str.split(',')]
    
    if len(components) < 1:
        return None
    
    try:
        parsed_components = []
        for comp in components:
            value = parse_component_value(comp)
            if value is None:
                return None
            parsed_components.append(complex(value, 0))
        
        return parsed_components
    
    except (ValueError, AttributeError) as e:
        return None

def parse_real_value(value_str: str) -> Optional[float]:
    """
    Parse a real number that might be symbolic (e.g., "1/√2", "√3/2", "1/8") or numeric.
    
    Args:
        value_str: String representation of a real number
        
    Returns:
        Float value or None if parsing fails
    """
    value_str = value_str.strip()
    
    # Handle negative values
    is_negative = False
    if value_str.startswith('-'):
        is_negative = True
        value_str = value_str[1:].strip()
    elif value_str.startswith('+'):
        value_str = value_str[1:].strip()
    
    # Check if it matches a symbolic value from COMMON_SQRT_VALUES
    for numeric_value, symbolic_str in COMMON_SQRT_VALUES.items():
        if value_str == symbolic_str:
            return -numeric_value if is_negative else numeric_value

    # Handle generic a/√b forms like "2/√2", "3/√8", "0.5/√2"
    sqrt_fraction_match = re.fullmatch(
        r'([0-9]*\.?[0-9]+)\s*/\s*√\s*([0-9]*\.?[0-9]+)',
        value_str
    )
    if sqrt_fraction_match:
        try:
            numerator = float(sqrt_fraction_match.group(1))
            radicand = float(sqrt_fraction_match.group(2))
            if radicand > 0:
                result = numerator / np.sqrt(radicand)
                return -result if is_negative else result
        except ValueError:
            pass

    # Examples: "2/(2√2)", "3/(4√5)", "2/(√2)", "2/2√2"
    sqrt_den_match = re.fullmatch(
        r'([0-9]*\.?[0-9]+)\s*/\s*\(?\s*([0-9]*\.?[0-9]+)?\s*√\s*([0-9]*\.?[0-9]+)\s*\)?',
        value_str
    )
    if sqrt_den_match:
        try:
            numerator = float(sqrt_den_match.group(1))
            coeff_str = sqrt_den_match.group(2)
            coeff = float(coeff_str) if coeff_str else 1.0
            radicand = float(sqrt_den_match.group(3))

            denominator = coeff * np.sqrt(radicand)
            if denominator != 0:
                result = numerator / denominator
                return -result if is_negative else result
        except ValueError:
            pass

    # Examples: "√2/2", "3√2/4", "0.5√8/2"
    sqrt_num_match = re.fullmatch(
        r'([0-9]*\.?[0-9]+)?\s*√\s*([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)',
        value_str
    )
    if sqrt_num_match:
        try:
            coeff_str = sqrt_num_match.group(1)
            coeff = float(coeff_str) if coeff_str else 1.0
            radicand = float(sqrt_num_match.group(2))
            denominator = float(sqrt_num_match.group(3))
            if radicand > 0 and denominator != 0:
                result = (coeff * np.sqrt(radicand)) / denominator
                return -result if is_negative else result
        except ValueError:
            pass

    # Handle parenthesized numerator forms
    # Examples: "(√2)/2", "(3√2)/4"
    sqrt_num_paren_match = re.fullmatch(
        r'\(\s*([0-9]*\.?[0-9]+)?\s*√\s*([0-9]*\.?[0-9]+)\s*\)\s*/\s*([0-9]*\.?[0-9]+)',
        value_str
    )
    if sqrt_num_paren_match:
        try:
            coeff_str = sqrt_num_paren_match.group(1)
            coeff = float(coeff_str) if coeff_str else 1.0
            radicand = float(sqrt_num_paren_match.group(2))
            denominator = float(sqrt_num_paren_match.group(3))
            if radicand > 0 and denominator != 0:
                result = (coeff * np.sqrt(radicand)) / denominator
                return -result if is_negative else result
        except ValueError:
            pass

    # Try to parse as a regular float
    try:
        result = float(value_str)
        return -result if is_negative else result
    except ValueError:
        pass
    
    # Try to evaluate simple fractions like "1/8"
    if '/' in value_str and '√' not in value_str:
        try:
            parts = value_str.split('/')
            if len(parts) == 2:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator != 0:
                    result = numerator / denominator
                    return -result if is_negative else result
        except ValueError:
            pass
    
    return None

def parse_component_value(comp: str) -> Optional[complex]:
    """
    Parse a single component value, handling symbolic representations and complex numbers.
    
    Args:
        comp: Component string (e.g., "1/√2", "-0.71", "√3/2", "1/2", "0.5+0.3j", "√3/2j", "1/8i", "-0.71 + 1/√2j")
        
    Returns:
        Complex number or None if parsing fails
    """
    comp = comp.strip()
    comp = comp.replace('i', 'j')  # Normalize imaginary unit to 'j'
    
    # Check for complex numbers with both real and imaginary parts (contains + or - in middle)
    # Pattern: find + or - that's not at the start and is followed by something ending in 'j'
    # This handles cases like "-0.71 + 1/√2j" or "0.5-0.3j"
    for i in range(1, len(comp)):  # Start from 1 to skip leading sign
        if comp[i] in ['+', '-']:
            # Check if this splits into real and imaginary parts
            real_part = comp[:i].strip()
            imag_part = comp[i:].strip()
            
            # If imaginary part ends with 'j', try to parse as complex
            if imag_part.endswith('j'):
                try:
                    # Parse real part
                    real_value = parse_real_value(real_part)
                    if real_value is None:
                        continue
                    
                    # Parse imaginary part (remove 'j' first)
                    imag_coeff = imag_part[:-1].strip()
                    
                    # Handle cases like "+1/√2" or "-1/√2"
                    if not imag_coeff or imag_coeff == '+':
                        imag_value = 1.0
                    elif imag_coeff == '-':
                        imag_value = -1.0
                    else:
                        imag_value = parse_real_value(imag_coeff)
                        if imag_value is None:
                            continue
                    
                    return complex(real_value, imag_value)
                except (ValueError, AttributeError):
                    continue
    
    # Check if this is purely imaginary (ends with 'j' and no +/- in the middle found above)
    if comp.endswith('j'):
        coeff_str = comp[:-1].strip()
        
        # Handle standalone 'j' (coefficient is 1)
        if not coeff_str or coeff_str == '+':
            return complex(0, 1)
        if coeff_str == '-':
            return complex(0, -1)
        
        # Parse the coefficient as a real value
        coeff_value = parse_real_value(coeff_str)
        if coeff_value is not None:
            return complex(0, coeff_value)
        else:
            print(f"Failed to parse imaginary coefficient: {coeff_str}")
            return None
    
    # Otherwise, try to parse as a real number
    real_value = parse_real_value(comp)
    if real_value is not None:
        return complex(real_value, 0)
    
    print(f"Failed to parse component value: {comp}")
    return None


def parse_all_quantum_states(text: str) -> List[Optional[List[complex]]]:
    """
    Extract all quantum state arrays from text (for multiple steps).
    Handles multiple formats:
    1. <quantum_state>: [...]
    2. "The current state becomes [...]" or "The current state becomes: [...]"
    3. "Current state: [...]"
    
    Detects which pattern exists in the text and uses only that pattern for parsing.
    Only returns successfully parsed states (skips those that fail to parse).
    
    Args:
        text: Text containing multiple quantum states
        
    Returns:
        List of quantum states (only successfully parsed states, no None values)
    """
    states = []
    
    # Define all patterns
    # TODO: Update this pattern later on 
    pattern1 = r'<quantum_state>\s*:?\s*\[(.*?)\]'  

    
    # Check which pattern exists in the text (in priority order)
    selected_pattern = None
    if re.search(pattern1, text):
        selected_pattern = pattern1
    
    # If no pattern found, return empty list
    if selected_pattern is None:
        return states
    
    # Find all matches for the selected pattern
    matches = re.findall(selected_pattern, text)
    
    # Parse each match, only keep successfully parsed states
    for match_content in matches:
        components = [s.strip() for s in match_content.split(',')]        
        if len(components) < 1:
            continue  # Skip empty states
        
        try:
            parsed_components = []
            parse_success = True
            for comp in components:
                value = parse_component_value(comp)
                if value is None:
                    print(f"Failed to parse component '{comp}'")
                    parse_success = False
                    break
                parsed_components.append(value)

            # Only append if parsing was successful
            if parse_success:
                states.append(parsed_components)
            else:
                print(f"Skipping state due to parse failure: {match_content}")
        except (ValueError, AttributeError):
            print(f"Exception occurred while parsing state: {match_content}")
            continue  # Skip states that raise exceptions
    
    return states

def compute_fidelity(state1: List[complex], state2: List[complex]) -> float:
    """
    Compute fidelity between two quantum states as the absolute square of their inner product.
    
    Fidelity = |<ψ₁|ψ₂>|²
    
    Args:
        state1: First quantum state as list of complex numbers
        state2: Second quantum state as list of complex numbers
        
    Returns:
        Fidelity value between 0 and 1
    """
    if len(state1) != len(state2):
        return 0.0
    
    # Convert to numpy arrays
    psi1 = np.array(state1, dtype=complex)
    psi2 = np.array(state2, dtype=complex)
    
    # Normalize states
    norm1 = np.linalg.norm(psi1)
    norm2 = np.linalg.norm(psi2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    psi1_normalized = psi1 / norm1
    psi2_normalized = psi2 / norm2
    
    # Compute inner product
    inner_product = np.vdot(psi1_normalized, psi2_normalized)
    
    # Fidelity is the absolute square
    fidelity = np.abs(inner_product) ** 2
    
    return float(fidelity)

def parse_probability_distribution(text: str) -> Optional[Dict[str, float]]:
    """
    Parse probability distribution from model output.
    Looks for patterns like:
    |00000>: |1/2|^2 = 0.250
    |01000>: |-0.50|^2 = 0.250
    
    Args:
        text: Text containing probability distribution
        
    Returns:
        Dictionary mapping bitstrings to probabilities, or None if parsing fails
    """
    # Pattern to match lines like |00000>: |1/2|^2 = 0.250
    pattern = r'\|([01]+)>\s*:.*?=\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, text)
    # print(f"parse_probability_distribution matches: {matches}")
    
    if not matches:
        return None
    
    prob_dist = {}
    for bitstring, prob_str in matches:
        try:
            prob = float(prob_str)
            prob_dist[bitstring] = prob
        except ValueError:
            print(f"Failed to parse probability value: {prob_str}")
            continue
    
    return prob_dist if prob_dist else None

def compute_classical_fidelity(pred_dist: Dict[str, float], truth_dist: Dict[str, float]) -> float:
    """
    Compute classical fidelity between two probability distributions.
    Classical fidelity = (Σ √(p_i * q_i))
    
    Args:
        pred_dist: Predicted probability distribution
        truth_dist: Ground truth probability distribution
        
    Returns:
        Classical fidelity value between 0 and 1
    """
    if not pred_dist or not truth_dist:
        return 0.0
    
    # Get all bitstrings from both distributions
    all_bitstrings = set(pred_dist.keys()) | set(truth_dist.keys())
    
    # Compute classical fidelity
    fidelity_sum = 0.0
    for bitstring in all_bitstrings:
        p = pred_dist.get(bitstring, 0.0)
        q = truth_dist.get(bitstring, 0.0)
        fidelity_sum += np.sqrt(p * q)
    
    classical_fidelity = fidelity_sum
    
    return float(classical_fidelity)

def load_eval_results(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df

def count_tokens(response, tokenizer):
    encoded_input = tokenizer.encode(response)
    num_tokens = len(encoded_input)
    return num_tokens

def track_metrics_by_group(metrics_dict, key, metric_values):
    """
    Track multiple metrics for a given grouping key.
    
    Args:
        metrics_dict: Dictionary to store metrics
        key: The grouping key (e.g., depth bin, num_qubits)
        metric_values: Dictionary of metric names to values
    """
    if key not in metrics_dict:
        metrics_dict[key] = {name: [] for name in metric_values.keys()}
    
    for metric_name, value in metric_values.items():
        if value is not None:
            metrics_dict[key][metric_name].append(value)

def bin_value(value, num_bins=10):
    """
    Bin a value into discrete bins for stratification.
    Returns bin identifier like "1-10", "11-20", etc.
    
    Args:
        value: The value to bin
        num_bins: Number of items per bin (default: 10)
        
    Returns:
        String representing the bin range
    """
    if value is None:
        return None
    bin_start = ((value - 1) // num_bins) * num_bins + 1
    bin_end = bin_start + num_bins - 1
    return f"{bin_start}-{bin_end}"


def _parse_extra_info_field(value: Any) -> Any:
    """
    Parse extra_info fields that may be a JSON string or already a Python object.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _extract_extra_info_dict(row: pd.Series) -> Dict[str, Any]:
    """
    Extract extra_info as a dictionary from a dataframe row.
    """
    extra_info = row.get('extra_info', {})
    extra_info = _parse_extra_info_field(extra_info)
    return extra_info if isinstance(extra_info, dict) else {}


def _extract_ground_truth_prob_dist(extra_info: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[str]]:
    """
    Extract ground-truth probability distribution with support for updated and legacy key names.

    Priority:
    1) msb_measurement_probabilities (new)
    2) probability_distribution (legacy)
    3) ground_truth (legacy)
    """
    key_priority = ['msb_measurement_probabilities', 'probability_distribution', 'ground_truth']
    source_key = None
    raw_dist = None

    for key in key_priority:
        candidate = extra_info.get(key)
        if candidate not in (None, '', {}):
            source_key = key
            raw_dist = candidate
            break

    if raw_dist in (None, '', {}):
        return {}, source_key

    parsed_dist = _parse_extra_info_field(raw_dist)
    if not isinstance(parsed_dist, dict):
        return {}, source_key

    try:
        normalized = {str(k): float(v) for k, v in parsed_dist.items()}
    except (TypeError, ValueError):
        return {}, source_key

    return normalized, source_key

def compute_step_by_step_fidelity(pred_states, 
                                  truth_states, 
                                  num_qubits):
    step_fidelity_dict = defaultdict(list)
    step_fidelity_by_qubits_dict = defaultdict(lambda: defaultdict(list))
    individual_updates = {
        'parse_success': False,
        'step_fidelities': [],
        'fidelity': 0.0,
        'final_fidelity': 0.0,
        'perfect_fidelity': False,
        'perfect_final_fidelity': False,
    }

    parse_success_increment = 0
    perfect_fidelity_increment = 0
    perfect_final_fidelity_increment = 0
    fidelity_score = 0.0
    final_fidelity_score = 0.0
    
    if pred_states and truth_states:
        parse_success_increment = 1
        individual_updates['parse_success'] = True
        
        # Compute fidelity for each step
        step_fidelities = []
        
        for step_idx, truth_state in enumerate(truth_states):
            pred_state = pred_states[step_idx] if step_idx < len(pred_states) else None

            fidelity = compute_fidelity(pred_state, truth_state) if pred_state else 0
            step_fidelities.append(fidelity)
                
            # Track per-step fidelity (overall)
            step_fidelity_dict[step_idx].append(fidelity)
            
            # Track per-step fidelity by number of qubits
            if num_qubits is not None:
                qubits_key = f"{num_qubits}_qubits"
                step_fidelity_by_qubits_dict[qubits_key][step_idx].append(fidelity)

        all_perfect = all(f >= FIDELITY_THRESHOLD for f in step_fidelities if f is not None)
        individual_updates['step_fidelities'] = step_fidelities
        
        # Average fidelity across all steps for this sample
        valid_fidelities = [f for f in step_fidelities if f is not None]
        if valid_fidelities:
            avg_sample_fidelity = np.mean(valid_fidelities)
            fidelity_score = float(avg_sample_fidelity)
            individual_updates['fidelity'] = fidelity_score
            
            # Check for perfect fidelity across all steps
            if all_perfect and len(valid_fidelities) == len(step_fidelities):
                perfect_fidelity_increment = 1
                individual_updates['perfect_fidelity'] = True
            
            # Compute final step fidelity (last quantum state)
            if step_fidelities:
                final_fidelity = step_fidelities[-1]
                final_fidelity_score = float(final_fidelity)
                individual_updates['final_fidelity'] = final_fidelity_score
                    
                # Check for perfect final fidelity
                if final_fidelity >= FIDELITY_THRESHOLD:
                    perfect_final_fidelity_increment = 1
                    individual_updates['perfect_final_fidelity'] = True
    else:
        individual_updates['fidelity'] = 0.0
        individual_updates['final_fidelity'] = 0.0
    
    return {
        'fidelity_score': fidelity_score,
        'final_fidelity_score': final_fidelity_score,
        'individual_updates': individual_updates,
        'parse_success_increment': parse_success_increment,
        'perfect_fidelity_increment': perfect_fidelity_increment,
        'perfect_final_fidelity_increment': perfect_final_fidelity_increment,
        'step_fidelity_dict': step_fidelity_dict,
        'step_fidelity_by_qubits_dict': step_fidelity_by_qubits_dict,
    }

def evaluate_model_performance(df: pd.DataFrame, model_name='Qwen/Qwen3-0.6B', store_individual=True) -> Dict[str, float]:
    """
    Comprehensive evaluation of model performance for quantum gate predictions.
    
    Args:
        df: DataFrame with columns 'responses' and 'extra_info'
        model_name: Model name for tokenizer
        store_individual: Whether to store individual sample results
        
    Returns:
        Dictionary with evaluation metrics and optionally individual results
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, max_length=40960)
    
    results = {
        'total_samples': len(df),
        'avg_tokens': 0.0,
        'avg_fidelity': 0.0,
        'avg_final_fidelity': 0.0,  # Fidelity of only the last quantum state
        'avg_classical_fidelity': 0.0,
        'avg_f1_score': 0.0,
        'avg_mae': 0.0,
        'avg_tvd_topk': 0.0,  # TVD top-k renormalized metric
        'avg_search_accuracy': 0.0,  # Marked states search accuracy
        'avg_format_accuracy': 0.0,
        'avg_reasoning_format_accuracy': 0.0,
        'parse_success_rate': 0.0,
        'classical_parse_success_rate': 0.0,
        'json_parse_success_rate': 0.0,
        'perfect_fidelity_count': 0,
        'perfect_fidelity_rate': 0.0,
        'perfect_final_fidelity_count': 0,  # Count of perfect final step fidelity
        'perfect_final_fidelity_rate': 0.0,
        'perfect_classical_fidelity_count': 0,
        'perfect_classical_fidelity_rate': 0.0,
        'perfect_match_count': 0,
        'perfect_match_rate': 0.0,
        'perfect_match_count_mae_0_01': 0,
        'perfect_match_rate_mae_0_01': 0.0,
        'perfect_match_count_mae_0_05': 0,
        'perfect_match_rate_mae_0_05': 0.0,
        'perfect_match_count_mae_0_1': 0,
        'perfect_match_rate_mae_0_1': 0.0,
        'perfect_match_count_tvd_0_01': 0,
        'perfect_match_rate_tvd_0_01': 0.0,
        'perfect_match_count_tvd_0_05': 0,
        'perfect_match_rate_tvd_0_05': 0.0,
        'perfect_match_count_tvd_0_1': 0,
        'perfect_match_rate_tvd_0_1': 0.0,
        'perfect_match_count_tvd_0_2': 0,
        'perfect_match_rate_tvd_0_2': 0.0,
        'perfect_match_count_tvd_0': 0,
        'perfect_match_rate_tvd_0': 0.0,
        'token_efficiency_mae_0_005': 0.0,
        'token_efficiency_mae_0_01': 0.0,
        'token_efficiency_mae_0_05': 0.0,
        'token_efficiency_mae_0_1': 0.0,
        'token_efficiency_tvd_0_01': 0.0,
        'token_efficiency_tvd_0_05': 0.0,
        'token_efficiency_tvd_0_1': 0.0,
        'token_efficiency_tvd_0_2': 0.0,
        'token_efficiency_tvd_0': 0.0,
        'step_fidelities': {},  # Will store average fidelity per step
        'step_fidelities_by_num_qubits': {},  # Will store average step-wise fidelity per number of qubits
        'metrics_by_circuit_depth': {},  # All metrics by circuit depth (binned)
        'metrics_by_num_qubits': {},  # All metrics by number of qubits
        'metrics_by_num_gates': {},  # All metrics by number of gates (binned)
    }
    
    # Store individual results if requested
    if store_individual:
        results['individual_results'] = []
    
    token_counts = []
    fidelity_scores = []
    final_fidelity_scores = []  # Track only final step fidelity
    classical_fidelity_scores = []
    f1_scores = []
    mae_scores = []
    tvd_topk_scores = []  # Track TVD top-k scores
    search_acc_scores = []  # Track search accuracy scores
    format_acc_scores = []
    format_criteria_list = []
    reasoning_format_acc_scores = []
    reasoning_format_criteria_list = []
    parse_success_count = 0
    classical_parse_success_count = 0
    json_parse_success_count = 0
    perfect_fidelity_count = 0
    perfect_final_fidelity_count = 0  # Count of perfect final fidelity
    perfect_classical_fidelity_count = 0
    perfect_match_count = 0
    perfect_match_count_mae_0_01 = 0
    perfect_match_count_mae_0_05 = 0
    perfect_match_count_mae_0_1 = 0
    perfect_match_count_tvd_0_01 = 0
    perfect_match_count_tvd_0_05 = 0
    perfect_match_count_tvd_0_1 = 0
    perfect_match_count_tvd_0_2 = 0
    perfect_match_count_tvd_0 = 0
    step_fidelity_dict = defaultdict(list)  # Track fidelities for each step position
    step_fidelity_by_qubits_dict = defaultdict(lambda: defaultdict(list))  # Track step-wise fidelities by number of qubits
    
    # Track all metrics by circuit characteristics
    metrics_by_depth = {}  # Track all metrics by circuit depth
    metrics_by_qubits = {}  # Track all metrics by number of qubits
    metrics_by_gates = {}  # Track all metrics by number of gates
    
    # Track parse statistics by number of qubits
    parse_stats_by_qubits = {}  # {num_qubits: {'total': count, 'parsed': count}}
    
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        prompt = row.get('prompt', '')
        response = row.get('responses', '')
        
        # Get ground truth from extra_info
        extra_info = _extract_extra_info_dict(row)
    
        ground_truth = row.get('completion', 'N/A') 
        ground_truth_prob, prob_source_key = _extract_ground_truth_prob_dist(extra_info)
        

        extra_info = extra_info

        circuit_depth = extra_info.get('circuit_depth', None)
        num_qubits_info = extra_info.get('num_qubits', None)
        if num_qubits_info is not None:
            num_qubits_info = int(num_qubits_info)  
        num_gates = extra_info.get('num_gates', None)
        if num_gates is not None:
            num_gates = int(num_gates)  
        marked_states = extra_info.get('marked_states', [])
        
        # Handle if marked_states is stored as JSON string
        if isinstance(marked_states, str):
            try:
                marked_states = json.loads(marked_states)
            except json.JSONDecodeError:
                marked_states = []
       
        # Initialize individual result record
        individual_result = {
            'index': idx,
            'tokens': 0,
            'fidelity': None,
            'final_fidelity': None,  # Fidelity of only the last quantum state
            'classical_fidelity': None,
            'f1_score': None,
            'mae': None,
            'tvd_topk': None,  # TVD top-k renormalized score
            'search_acc': None,  # Marked states search accuracy
            'format_accuracy': None,
            'format_criteria': None,
            'reasoning_format_accuracy': None,
            'reasoning_format_criteria': None,
            'parse_success': False,
            'classical_parse_success': False,
            'json_parse_success': False,
            'perfect_fidelity': False,
            'perfect_final_fidelity': False,  # Perfect final step fidelity
            'perfect_classical_fidelity': False,
            'perfect_match': False,
            'perfect_match_mae_0_01': False,
            'perfect_match_mae_0_05': False,
            'perfect_match_mae_0_1': False,
            'perfect_match_tvd_0_01': False,
            'perfect_match_tvd_0_05': False,
            'perfect_match_tvd_0_1': False,
            'perfect_match_tvd_0_2': False,
            'perfect_match_tvd_0': False,
            'step_fidelities': [],  # Store fidelity for each step
            'circuit_depth': circuit_depth,
            'num_qubits': num_qubits_info,
            'num_gates': num_gates,
            'marked_states': marked_states,
            'prompt': prompt,
            'response': response,
            'ground_truth': ground_truth,
            'ground_truth_prob_source': prob_source_key,
        }
        
        if not response:
            print(f"Skipping sample {idx} due to missing response or ground truth.")
            if store_individual:
                results['individual_results'].append(individual_result)
            continue
        
        # Count tokens
        num_tokens = count_tokens(response, tokenizer)
        token_counts.append(num_tokens)
        individual_result['tokens'] = num_tokens
        
        # Determine num_qubits from ground_truth_prob
        num_qubits = 0

        if ground_truth_prob:
            # Get num_qubits from the first key in ground_truth_prob
            first_key = next(iter(ground_truth_prob.keys()), None)
            if first_key:
                num_qubits = len(first_key)
        
        # Compute format accuracy
        format_acc_value, criteria_list = format_accuracy(response, num_qubits)
        individual_result['format_accuracy'] = format_acc_value
        individual_result['format_criteria'] = criteria_list
        format_acc_scores.append(format_acc_value)
        format_criteria_list.append(criteria_list)
        
        # Compute reasoning format accuracy
        reasoning_format_acc_value, reasoning_criteria_list = reasoning_format_accuracy(response, ground_truth, num_qubits)
        individual_result['reasoning_format_accuracy'] = reasoning_format_acc_value
        individual_result['reasoning_format_criteria'] = reasoning_criteria_list
        reasoning_format_acc_scores.append(reasoning_format_acc_value)
        reasoning_format_criteria_list.append(reasoning_criteria_list)
        
        # Parse all quantum states (for multi-step responses)
        pred_states = parse_all_quantum_states(response)
        truth_states = parse_all_quantum_states(ground_truth)

        # Track parse statistics by number of qubits
        if num_qubits_info is not None:
            if num_qubits_info not in parse_stats_by_qubits:
                parse_stats_by_qubits[num_qubits_info] = {'total': 0, 'parsed': 0}
            parse_stats_by_qubits[num_qubits_info]['total'] += 1
            if pred_states and truth_states and len(pred_states) == len(truth_states):
                parse_stats_by_qubits[num_qubits_info]['parsed'] += 1
              
        fidelity_result = compute_step_by_step_fidelity(
            pred_states=pred_states,
            truth_states=truth_states,
            num_qubits=num_qubits_info
        )

        fidelity_scores.append(fidelity_result['fidelity_score'])
        final_fidelity_scores.append(fidelity_result['final_fidelity_score'])
        parse_success_count += fidelity_result['parse_success_increment']
        perfect_fidelity_count += fidelity_result['perfect_fidelity_increment']
        perfect_final_fidelity_count += fidelity_result['perfect_final_fidelity_increment']
        individual_result.update(fidelity_result['individual_updates'])

        # Merge per-step fidelity tracking (overall)
        for step_idx, fidelity_values in fidelity_result['step_fidelity_dict'].items():
            step_fidelity_dict[step_idx].extend(fidelity_values)

        # Merge per-step fidelity tracking by number of qubits
        for qubits_key, step_dict in fidelity_result['step_fidelity_by_qubits_dict'].items():
            for step_idx, fidelity_values in step_dict.items():
                step_fidelity_by_qubits_dict[qubits_key][step_idx].extend(fidelity_values)

        if not fidelity_result['parse_success_increment']:
            # Format is incorrect for quantum states - penalize with zero fidelity
            print(f"Skipping fidelity computation for sample {idx} due to parse mismatch. \n"
                  f"pred_states: {len(pred_states)}, truth_states: {len(truth_states)}")
        
        # Compute classical fidelity if ground truth probability is available
        if ground_truth_prob:
            pred_prob_dist = parse_probability_distribution(response)
 
            if pred_prob_dist:
                classical_parse_success_count += 1
                individual_result['classical_parse_success'] = True
                
                # Compute classical fidelity
                classical_fid = compute_classical_fidelity(pred_prob_dist, ground_truth_prob)
                classical_fidelity_scores.append(classical_fid)
                individual_result['classical_fidelity'] = classical_fid
                
                # Check for perfect classical fidelity
                if classical_fid >= CLASSICAL_FIDELITY_THRESHOLD:
                    perfect_classical_fidelity_count += 1
                    individual_result['perfect_classical_fidelity'] = True
            else:
                # Format is incorrect for classical fidelity - penalize with zero
                classical_fidelity_scores.append(0.0)
                individual_result['classical_fidelity'] = 0.0
                    
        # Also try to parse JSON format for F1 and MAE computation
        if ground_truth_prob:
            pred_json_dist = extract_probability_distribution_from_json(response)

            if pred_json_dist:
                json_parse_success_count += 1
                individual_result['json_parse_success'] = True
                
                # Compute search accuracy if marked_states are available
                if len(marked_states) > 0:
                    search_accuracy = search_acc(pred_json_dist, marked_states)
                    search_acc_scores.append(search_accuracy)
                    individual_result['search_acc'] = search_accuracy
                
                # Compute F1 score
                f1 = compute_f1_score(pred_json_dist, ground_truth_prob)

                f1_scores.append(f1)
                individual_result['f1_score'] = f1
                
                # Compute MAE
                mae = compute_mae(pred_json_dist, ground_truth_prob, num_qubits)
                mae_scores.append(mae)
                individual_result['mae'] = mae
                
                # Get top-k states sorted by probability (descending)
                pred_states_sorted = sorted(pred_json_dist.items(), key=lambda x: x[1], reverse=True)
                true_states_sorted = sorted(ground_truth_prob.items(), key=lambda x: x[1], reverse=True)
                
                # Compute TVD top-k renormalized
                # Get top-k from both distributions
                k = 15
                pred_topk = dict(pred_states_sorted[:k])
                true_topk = dict(true_states_sorted[:k])
                
                tvd_topk = tvd_topk_renormalized(pred_topk, true_topk, k=k)
                tvd_topk_scores.append(tvd_topk)
                individual_result['tvd_topk'] = tvd_topk
                
                # Check for perfect match (F1 = 1.0 and MAE < threshold AND format correct)
                if f1 == 1.0 and mae < MAE_THRESHOLD and format_acc_value == 1.0:
                    perfect_match_count += 1
                    individual_result['perfect_match'] = True
                
                # Check for perfect match with MAE < 0.01
                if f1 == 1.0 and mae < 0.01 and format_acc_value == 1.0:
                    perfect_match_count_mae_0_01 += 1
                    individual_result['perfect_match_mae_0_01'] = True
                
                # Check for perfect match with MAE < 0.05
                if f1 == 1.0 and mae < 0.05 and format_acc_value == 1.0:
                    perfect_match_count_mae_0_05 += 1
                    individual_result['perfect_match_mae_0_05'] = True
                
                # Check for perfect match with MAE < 0.1
                if f1 == 1.0 and mae < 0.1 and format_acc_value == 1.0:
                    perfect_match_count_mae_0_1 += 1
                    individual_result['perfect_match_mae_0_1'] = True
                
                # Check for perfect match with TVD < 0.01
                if f1 == 1.0 and tvd_topk < 0.01 and format_acc_value == 1.0:
                    perfect_match_count_tvd_0_01 += 1
                    individual_result['perfect_match_tvd_0_01'] = True
                
                # Check for perfect match with TVD < 0.05
                if f1 == 1.0 and tvd_topk < 0.05 and format_acc_value == 1.0:
                    perfect_match_count_tvd_0_05 += 1
                    individual_result['perfect_match_tvd_0_05'] = True
                
                # Check for perfect match with TVD < 0.1
                if f1 == 1.0 and tvd_topk < 0.1 and format_acc_value == 1.0:
                    perfect_match_count_tvd_0_1 += 1
                    individual_result['perfect_match_tvd_0_1'] = True
                
                # Check for perfect match with TVD < 0.2
                if f1 == 1.0 and tvd_topk < 0.2 and format_acc_value == 1.0:
                    perfect_match_count_tvd_0_2 += 1
                    individual_result['perfect_match_tvd_0_2'] = True
                
                # Check for perfect match with TVD = 0 (exact match)
                if f1 == 1.0 and tvd_topk == 0.0 and format_acc_value == 1.0:
                    perfect_match_count_tvd_0 += 1
                    individual_result['perfect_match_tvd_0'] = True
            else:
                # Format is incorrect - penalize with worst possible scores
                f1_scores.append(0.0)
                individual_result['f1_score'] = 0.0
                
                mae_scores.append(1.0)
                individual_result['mae'] = 1.0
                
                tvd_topk_scores.append(1.0)
                individual_result['tvd_topk'] = 1.0
        
        # Track all metrics by circuit characteristics
        metric_values = {
            'fidelity': individual_result.get('fidelity'),
            'final_fidelity': individual_result.get('final_fidelity'),
            'classical_fidelity': individual_result.get('classical_fidelity'),
            'f1_score': individual_result.get('f1_score'),
            'mae': individual_result.get('mae'),
            'tvd_topk': individual_result.get('tvd_topk'),
            'search_acc': individual_result.get('search_acc'),
            'format_accuracy': individual_result.get('format_accuracy'),
            'reasoning_format_accuracy': individual_result.get('reasoning_format_accuracy'),
            'tokens': individual_result.get('tokens'),
        }
        
        # Track by circuit depth (binned)
        if circuit_depth is not None:
            depth_bin = bin_value(circuit_depth, num_bins=10)
            if depth_bin:
                track_metrics_by_group(metrics_by_depth, depth_bin, metric_values)
        
        # Track by number of qubits (not binned - usually small range)
        if num_qubits_info is not None:
            track_metrics_by_group(metrics_by_qubits, f"{num_qubits_info}_qubits", metric_values)
        
        # Track by number of gates (binned)
        if num_gates is not None:
            gates_bin = bin_value(num_gates, num_bins=10)
            if gates_bin:
                track_metrics_by_group(metrics_by_gates, gates_bin, metric_values)
        
        if store_individual:
            results['individual_results'].append(individual_result)
    
    # Compute overall format criteria accuracy
    if format_criteria_list:
        criteria_array = np.array(format_criteria_list)
        criteria_means = criteria_array.mean(axis=0)
        for i, mean in enumerate(criteria_means):
            results[f'format_criteria_{i+1}_accuracy'] = mean
    
    # Compute overall reasoning format criteria accuracy
    if reasoning_format_criteria_list:
        reasoning_criteria_array = np.array(reasoning_format_criteria_list)
        reasoning_criteria_means = reasoning_criteria_array.mean(axis=0)
        for i, mean in enumerate(reasoning_criteria_means):
            results[f'reasoning_format_criteria_{i+1}_accuracy'] = mean
    
    # Calculate averages
    if token_counts:
        results['avg_tokens'] = np.mean(token_counts)
    
    if fidelity_scores:
        results['avg_fidelity'] = np.mean(fidelity_scores)
    
    if final_fidelity_scores:
        results['avg_final_fidelity'] = np.mean(final_fidelity_scores)
    
    if classical_fidelity_scores:
        results['avg_classical_fidelity'] = np.mean(classical_fidelity_scores)
    
    if f1_scores:
        results['avg_f1_score'] = np.mean(f1_scores)
    
    if mae_scores:
        results['avg_mae'] = np.mean(mae_scores)
    
    if tvd_topk_scores:
        results['avg_tvd_topk'] = np.mean(tvd_topk_scores)
    
    if search_acc_scores:
        results['avg_search_accuracy'] = np.mean(search_acc_scores)
    
    if format_acc_scores:
        results['avg_format_accuracy'] = np.mean(format_acc_scores)
    
    if reasoning_format_acc_scores:
        results['avg_reasoning_format_accuracy'] = np.mean(reasoning_format_acc_scores)
    
    # Calculate per-step average fidelities
    for step_idx, fidelities in step_fidelity_dict.items():
        results['step_fidelities'][f'step_{step_idx + 1}'] = np.mean(fidelities)
        results['step_fidelities'][f'step_{step_idx + 1}_count'] = len(fidelities)
    
    # Calculate per-step average fidelities by number of qubits
    for qubits_key, step_dict in sorted(step_fidelity_by_qubits_dict.items(), key=lambda x: int(x[0].split('_')[0])):
        results['step_fidelities_by_num_qubits'][qubits_key] = {}
        for step_idx, fidelities in step_dict.items():
            results['step_fidelities_by_num_qubits'][qubits_key][f'step_{step_idx + 1}'] = np.mean(fidelities)
            results['step_fidelities_by_num_qubits'][qubits_key][f'step_{step_idx + 1}_count'] = len(fidelities)
    
    # Calculate average metrics by circuit depth (binned)
    for depth_bin, metrics in sorted(metrics_by_depth.items()):
        results['metrics_by_circuit_depth'][depth_bin] = {}
        for metric_name, values in metrics.items():
            if values:
                results['metrics_by_circuit_depth'][depth_bin][f'avg_{metric_name}'] = np.mean(values)
        results['metrics_by_circuit_depth'][depth_bin]['count'] = len(metrics.get('fidelity', metrics.get('tokens', [])))
    
    # Calculate average metrics by number of qubits
    for qubits_key, metrics in sorted(metrics_by_qubits.items(), key=lambda x: int(x[0].split('_')[0])):
        results['metrics_by_num_qubits'][qubits_key] = {}
        for metric_name, values in metrics.items():
            if values:
                results['metrics_by_num_qubits'][qubits_key][f'avg_{metric_name}'] = np.mean(values)
        results['metrics_by_num_qubits'][qubits_key]['count'] = len(metrics.get('fidelity', metrics.get('tokens', [])))
    
    # Calculate average metrics by number of gates (binned)
    for gates_bin, metrics in sorted(metrics_by_gates.items()):
        results['metrics_by_num_gates'][gates_bin] = {}
        for metric_name, values in metrics.items():
            if values:
                results['metrics_by_num_gates'][gates_bin][f'avg_{metric_name}'] = np.mean(values)
        results['metrics_by_num_gates'][gates_bin]['count'] = len(metrics.get('fidelity', metrics.get('tokens', [])))
    
    # Store parse statistics by number of qubits
    results['parse_stats_by_qubits'] = {}
    for num_qubits, stats in sorted(parse_stats_by_qubits.items()):
        results['parse_stats_by_qubits'][f'{num_qubits}_qubits'] = {
            'total': stats['total'],
            'parsed': stats['parsed'],
            'parse_rate': stats['parsed'] / stats['total'] if stats['total'] > 0 else 0.0
        }
    
    results['parse_success_rate'] = parse_success_count / len(df) if len(df) > 0 else 0.0
    results['classical_parse_success_rate'] = classical_parse_success_count / len(df) if len(df) > 0 else 0.0
    results['json_parse_success_rate'] = json_parse_success_count / len(df) if len(df) > 0 else 0.0
    results['perfect_fidelity_count'] = perfect_fidelity_count
    results['perfect_fidelity_rate'] = perfect_fidelity_count / len(df) if len(df) > 0 else 0.0
    results['perfect_final_fidelity_count'] = perfect_final_fidelity_count
    results['perfect_final_fidelity_rate'] = perfect_final_fidelity_count / len(df) if len(df) > 0 else 0.0
    results['perfect_classical_fidelity_count'] = perfect_classical_fidelity_count
    results['perfect_classical_fidelity_rate'] = perfect_classical_fidelity_count / len(df) if len(df) > 0 else 0.0
    results['perfect_match_count'] = perfect_match_count
    results['perfect_match_rate'] = perfect_match_count / len(df) if len(df) > 0 else 0.0
    results['perfect_match_count_mae_0_01'] = perfect_match_count_mae_0_01
    results['perfect_match_rate_mae_0_01'] = perfect_match_count_mae_0_01 / len(df) if len(df) > 0 else 0.0
    results['perfect_match_count_mae_0_05'] = perfect_match_count_mae_0_05
    results['perfect_match_rate_mae_0_05'] = perfect_match_count_mae_0_05 / len(df) if len(df) > 0 else 0.0
    results['perfect_match_count_mae_0_1'] = perfect_match_count_mae_0_1
    results['perfect_match_rate_mae_0_1'] = perfect_match_count_mae_0_1 / len(df) if len(df) > 0 else 0.0
    results['perfect_match_count_tvd_0_01'] = perfect_match_count_tvd_0_01
    results['perfect_match_rate_tvd_0_01'] = perfect_match_count_tvd_0_01 / len(df) if len(df) > 0 else 0.0
    results['perfect_match_count_tvd_0_05'] = perfect_match_count_tvd_0_05
    results['perfect_match_rate_tvd_0_05'] = perfect_match_count_tvd_0_05 / len(df) if len(df) > 0 else 0.0
    results['perfect_match_count_tvd_0_1'] = perfect_match_count_tvd_0_1
    results['perfect_match_rate_tvd_0_1'] = perfect_match_count_tvd_0_1 / len(df) if len(df) > 0 else 0.0
    results['perfect_match_count_tvd_0_2'] = perfect_match_count_tvd_0_2
    results['perfect_match_rate_tvd_0_2'] = perfect_match_count_tvd_0_2 / len(df) if len(df) > 0 else 0.0
    results['perfect_match_count_tvd_0'] = perfect_match_count_tvd_0
    results['perfect_match_rate_tvd_0'] = perfect_match_count_tvd_0 / len(df) if len(df) > 0 else 0.0
    
    # Calculate token efficiency: (Perfect Match % / Avg Tokens per 1000)
    if results['avg_tokens'] > 0:
        results['token_efficiency'] = (results['perfect_match_rate']) / (results['avg_tokens'] / 1000)
        results['token_efficiency_mae_0_005'] = (results['perfect_match_rate']) / (results['avg_tokens'] / 1000)
        results['token_efficiency_mae_0_01'] = (results['perfect_match_rate_mae_0_01']) / (results['avg_tokens'] / 1000)
        results['token_efficiency_mae_0_05'] = (results['perfect_match_rate_mae_0_05']) / (results['avg_tokens'] / 1000)
        results['token_efficiency_mae_0_1'] = (results['perfect_match_rate_mae_0_1']) / (results['avg_tokens'] / 1000)
        results['token_efficiency_tvd_0_01'] = (results['perfect_match_rate_tvd_0_01']) / (results['avg_tokens'] / 1000)
        results['token_efficiency_tvd_0_05'] = (results['perfect_match_rate_tvd_0_05']) / (results['avg_tokens'] / 1000)
        results['token_efficiency_tvd_0_1'] = (results['perfect_match_rate_tvd_0_1']) / (results['avg_tokens'] / 1000)
        results['token_efficiency_tvd_0_2'] = (results['perfect_match_rate_tvd_0_2']) / (results['avg_tokens'] / 1000)
        results['token_efficiency_tvd_0'] = (results['perfect_match_rate_tvd_0']) / (results['avg_tokens'] / 1000)
    else:
        results['token_efficiency'] = 0.0
        results['token_efficiency_mae_0_005'] = 0.0
        results['token_efficiency_mae_0_01'] = 0.0
        results['token_efficiency_mae_0_05'] = 0.0
        results['token_efficiency_mae_0_1'] = 0.0
        results['token_efficiency_tvd_0_01'] = 0.0
        results['token_efficiency_tvd_0_05'] = 0.0
        results['token_efficiency_tvd_0_1'] = 0.0
        results['token_efficiency_tvd_0_2'] = 0.0
        results['token_efficiency_tvd_0'] = 0.0
    
    return results

def save_individual_results(results: Dict, output_path: str, parquet_path: str, model_name: str):
    """
    Save three files:
    1. Individual prompt/response/ground truth file
    2. CSV file with metrics
    3. Text file with metadata and aggregated results
    
    Args:
        results: Dictionary containing evaluation results with 'individual_results' key
        output_path: Base path for output files (without extension)
        parquet_path: Path to the parquet file used for evaluation
        model_name: Name of the model used for evaluation
    """
    import os
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # 1. Save individual prompt/response/ground truth to text file
    responses_file = f"{output_path}_responses.txt"
    with open(responses_file, 'w', encoding='utf-8') as f:
        for res in results.get('individual_results', []):
            f.write("="*80 + "\n")
            f.write(f"Index: {res.get('index', '')}\n")
            f.write("-"*80 + "\n")
            f.write(f"PROMPT:\n{res.get('prompt', '')}\n\n")
            f.write(f"RESPONSE:\n{res.get('response', '')}\n\n")
            f.write(f"GROUND TRUTH:\n{res.get('ground_truth', '')}\n")
            f.write(f"Fidelity: {res.get('fidelity', 'N/A')}\n")
            f.write(f"Classical Fidelity: {res.get('classical_fidelity', 'N/A')}\n")
            f.write("="*80 + "\n\n")
    print(f"Individual responses saved to {responses_file}")
    
    # 2. Save CSV with metrics (without prompt/response/ground_truth)
    csv_file = f"{output_path}.csv"
    if 'individual_results' in results and results['individual_results']:
        individual_df = pd.DataFrame(results['individual_results'])
        
        # Expand step_fidelities list into separate columns
        if 'step_fidelities' in individual_df.columns:
            max_steps = max(len(sf) if isinstance(sf, list) else 0 for sf in individual_df['step_fidelities'])
            for step_idx in range(max_steps):
                individual_df[f'step_{step_idx + 1}_fidelity'] = individual_df['step_fidelities'].apply(
                    lambda x: x[step_idx] if isinstance(x, list) and step_idx < len(x) else None
                )
            # Drop the original list column
            individual_df = individual_df.drop(columns=['step_fidelities'])
        
        individual_df = individual_df.drop(columns=['response', 'ground_truth', 'prompt'], errors='ignore')
        individual_df.to_csv(csv_file, index=False)
        print(f"Metrics CSV saved to {csv_file}")
    
    # 3. Save metadata and aggregated results to text file
    summary_file = f"{output_path}_summary.txt"
    # print(results)
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("QUANTUM GATE EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("METADATA:\n")
        f.write(f"  Parquet Path: {parquet_path}\n")
        f.write(f"  Model Name: {model_name}\n")
        f.write(f"  Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("AGGREGATED RESULTS:\n")
        f.write(f"  Total Samples: {results['total_samples']}\n")
        f.write(f"  Average Tokens: {results['avg_tokens']:.2f}\n")
        f.write(f"  Average Fidelity (All Steps): {results['avg_fidelity']:.6f}\n")
        f.write(f"  Average Final Fidelity (Last Step): {results['avg_final_fidelity']:.6f}\n")
        f.write(f"  Average Classical Fidelity: {results['avg_classical_fidelity']:.6f}\n")
        f.write(f"  Average F1 Score: {results['avg_f1_score']:.6f}\n")
        f.write(f"  Average MAE: {results['avg_mae']:.6f}\n")
        f.write(f"  Average TVD Top-k (k=15): {results['avg_tvd_topk']:.6f}\n")
        f.write(f"  Average Search Accuracy (Marked States): {results['avg_search_accuracy']:.2%}\n")
        f.write(f"  Average Format Accuracy: {results['avg_format_accuracy']:.2%}\n")
        f.write(f"  Average Reasoning Format Accuracy: {results['avg_reasoning_format_accuracy']:.2%}\n")
        
        # Format criteria breakdown
        criteria_names = [
            "All basis lengths match num_qubits",
            "Basis format is binary string",
            "No duplicate basis states",
            "All probabilities sum in range [0, 1]",
            "All probabilities non-negative",
            "Probability has ≤ 3 decimal places",
            "No more than 15 entries in the distribution"
        ]
        for i in range(1, 8):
            if f'format_criteria_{i}_accuracy' in results:
                f.write(f"  Format Criteria {i} ({criteria_names[i-1]}) Accuracy: {results[f'format_criteria_{i}_accuracy']:.2%}\n")
        
        # Reasoning format criteria breakdown
        reasoning_criteria_names = [
            "Wrapped in <circuit_reasoning>...</circuit_reasoning> tags",
            "Each step contains <quantum_state> tags",
            "Measurement outcome probability section present",
            "Each quantum state length equals 2^num_qubits"
        ]
        for i in range(1, 5):
            if f'reasoning_format_criteria_{i}_accuracy' in results:
                f.write(f"  Reasoning Format Criteria {i} ({reasoning_criteria_names[i-1]}) Accuracy: {results[f'reasoning_format_criteria_{i}_accuracy']:.2%}\n")
        
        f.write(f"  Parse Success Rate: {results['parse_success_rate']:.2%}\n")
        f.write(f"  Classical Parse Success Rate: {results['classical_parse_success_rate']:.2%}\n")
        f.write(f"  JSON Parse Success Rate: {results['json_parse_success_rate']:.2%}\n")
        f.write(f"  Perfect Fidelity (All Steps): {results['perfect_fidelity_count']} ({results['perfect_fidelity_rate']:.2%})\n")
        f.write(f"  Perfect Final Fidelity (Last Step): {results['perfect_final_fidelity_count']} ({results['perfect_final_fidelity_rate']:.2%})\n")
        f.write(f"  Perfect Classical Fidelity: {results['perfect_classical_fidelity_count']} ({results['perfect_classical_fidelity_rate']:.2%})\n")
        f.write(f"  Perfect Match (F1=1.0 & MAE<{MAE_THRESHOLD}): {results['perfect_match_count']} ({results['perfect_match_rate']:.2%})\n")
        f.write(f"  Perfect Match (F1=1.0 & MAE<0.01): {results['perfect_match_count_mae_0_01']} ({results['perfect_match_rate_mae_0_01']:.2%})\n")
        f.write(f"  Perfect Match (F1=1.0 & MAE<0.05): {results['perfect_match_count_mae_0_05']} ({results['perfect_match_rate_mae_0_05']:.2%})\n")
        f.write(f"  Perfect Match (F1=1.0 & MAE<0.1): {results['perfect_match_count_mae_0_1']} ({results['perfect_match_rate_mae_0_1']:.2%})\n")
        f.write(f"  Perfect Match (F1=1.0 & TVD<0.01): {results['perfect_match_count_tvd_0_01']} ({results['perfect_match_rate_tvd_0_01']:.2%})\n")
        f.write(f"  Perfect Match (F1=1.0 & TVD<0.05): {results['perfect_match_count_tvd_0_05']} ({results['perfect_match_rate_tvd_0_05']:.2%})\n")
        f.write(f"  Perfect Match (F1=1.0 & TVD<0.1): {results['perfect_match_count_tvd_0_1']} ({results['perfect_match_rate_tvd_0_1']:.2%})\n")
        f.write(f"  Perfect Match (F1=1.0 & TVD<0.2): {results['perfect_match_count_tvd_0_2']} ({results['perfect_match_rate_tvd_0_2']:.2%})\n")
        f.write(f"  Perfect Match (F1=1.0 & TVD=0): {results['perfect_match_count_tvd_0']} ({results['perfect_match_rate_tvd_0']:.2%})\n")
        f.write(f"\n  TOKEN EFFICIENCY (Perfect Match % / Avg Tokens per 1K):\n")
        f.write(f"    MAE<0.005: {results.get('token_efficiency_mae_0_005', 0.0):.4f}\n")
        f.write(f"    MAE<0.01: {results.get('token_efficiency_mae_0_01', 0.0):.4f}\n")
        f.write(f"    MAE<0.05: {results.get('token_efficiency_mae_0_05', 0.0):.4f}\n")
        f.write(f"    MAE<0.1: {results.get('token_efficiency_mae_0_1', 0.0):.4f}\n")
        f.write(f"    TVD<0.01: {results.get('token_efficiency_tvd_0_01', 0.0):.4f}\n")
        f.write(f"    TVD<0.05: {results.get('token_efficiency_tvd_0_05', 0.0):.4f}\n")
        f.write(f"    TVD<0.1: {results.get('token_efficiency_tvd_0_1', 0.0):.4f}\n")
        f.write(f"    TVD<0.2: {results.get('token_efficiency_tvd_0_2', 0.0):.4f}\n")
        f.write(f"    TVD=0: {results.get('token_efficiency_tvd_0', 0.0):.4f}\n")
        
        # Per-step fidelity
        if 'step_fidelities' in results and results['step_fidelities']:
            f.write("\n  PER-STEP FIDELITY:\n")
            for step_name in sorted([k for k in results['step_fidelities'].keys() if not k.endswith('_count')]):
                fidelity = results['step_fidelities'][step_name]
                count = results['step_fidelities'].get(f'{step_name}_count', 0)
                f.write(f"    {step_name}: {fidelity:.6f} (n={count})\n")
        
        # Per-step fidelity by number of qubits
        if 'step_fidelities_by_num_qubits' in results and results['step_fidelities_by_num_qubits']:
            f.write("\n  PER-STEP FIDELITY BY NUMBER OF QUBITS:\n")
            for qubits_key, step_fidelities in sorted(results['step_fidelities_by_num_qubits'].items(), key=lambda x: int(x[0].split('_')[0])):
                f.write(f"    {qubits_key}:\n")
                for step_name in sorted([k for k in step_fidelities.keys() if not k.endswith('_count')]):
                    fidelity = step_fidelities[step_name]
                    count = step_fidelities.get(f'{step_name}_count', 0)
                    f.write(f"      {step_name}: {fidelity:.6f} (n={count})\n")
        
        # Parse statistics by number of qubits
        if 'parse_stats_by_qubits' in results and results['parse_stats_by_qubits']:
            f.write("\n  QUANTUM STATE PARSE STATISTICS BY NUMBER OF QUBITS:\n")
            for qubits_key, stats in sorted(results['parse_stats_by_qubits'].items(), key=lambda x: int(x[0].split('_')[0])):
                f.write(f"    {qubits_key}: {stats['parsed']}/{stats['total']} ({stats['parse_rate']:.2%})\n")
        
        # Metrics by circuit depth
        if 'metrics_by_circuit_depth' in results and results['metrics_by_circuit_depth']:
            f.write("\n  METRICS BY CIRCUIT DEPTH:\n")
            for depth_bin, metrics in sorted(results['metrics_by_circuit_depth'].items()):
                f.write(f"    Depth {depth_bin} (n={metrics.get('count', 0)}):\n")
                if 'avg_fidelity' in metrics:
                    f.write(f"      Avg Fidelity: {metrics['avg_fidelity']:.6f}\n")
                if 'avg_final_fidelity' in metrics:
                    f.write(f"      Avg Final Fidelity: {metrics['avg_final_fidelity']:.6f}\n")
                if 'avg_classical_fidelity' in metrics:
                    f.write(f"      Avg Classical Fidelity: {metrics['avg_classical_fidelity']:.6f}\n")
                if 'avg_f1_score' in metrics:
                    f.write(f"      Avg F1 Score: {metrics['avg_f1_score']:.6f}\n")
                if 'avg_mae' in metrics:
                    f.write(f"      Avg MAE: {metrics['avg_mae']:.6f}\n")
                if 'avg_tvd_topk' in metrics:
                    f.write(f"      Avg TVD Top-k: {metrics['avg_tvd_topk']:.6f}\n")
                if 'avg_search_acc' in metrics:
                    f.write(f"      Avg Search Accuracy: {metrics['avg_search_acc']:.2%}\n")
        
        # Metrics by number of qubits
        if 'metrics_by_num_qubits' in results and results['metrics_by_num_qubits']:
            f.write("\n  METRICS BY NUMBER OF QUBITS:\n")
            for qubits_key, metrics in sorted(results['metrics_by_num_qubits'].items(), key=lambda x: int(x[0].split('_')[0])):
                f.write(f"    {qubits_key} (n={metrics.get('count', 0)}):\n")
                if 'avg_fidelity' in metrics:
                    f.write(f"      Avg Fidelity: {metrics['avg_fidelity']:.6f}\n")
                if 'avg_final_fidelity' in metrics:
                    f.write(f"      Avg Final Fidelity: {metrics['avg_final_fidelity']:.6f}\n")
                if 'avg_classical_fidelity' in metrics:
                    f.write(f"      Avg Classical Fidelity: {metrics['avg_classical_fidelity']:.6f}\n")
                if 'avg_f1_score' in metrics:
                    f.write(f"      Avg F1 Score: {metrics['avg_f1_score']:.6f}\n")
                if 'avg_mae' in metrics:
                    f.write(f"      Avg MAE: {metrics['avg_mae']:.6f}\n")
                if 'avg_tvd_topk' in metrics:
                    f.write(f"      Avg TVD Top-k: {metrics['avg_tvd_topk']:.6f}\n")
                if 'avg_search_acc' in metrics:
                    f.write(f"      Avg Search Accuracy: {metrics['avg_search_acc']:.2%}\n")
        
        # Metrics by number of gates
        if 'metrics_by_num_gates' in results and results['metrics_by_num_gates']:
            f.write("\n  METRICS BY NUMBER OF GATES:\n")
            for gates_bin, metrics in sorted(results['metrics_by_num_gates'].items()):
                f.write(f"    Gates {gates_bin} (n={metrics.get('count', 0)}):\n")
                if 'avg_fidelity' in metrics:
                    f.write(f"      Avg Fidelity: {metrics['avg_fidelity']:.6f}\n")
                if 'avg_final_fidelity' in metrics:
                    f.write(f"      Avg Final Fidelity: {metrics['avg_final_fidelity']:.6f}\n")
                if 'avg_classical_fidelity' in metrics:
                    f.write(f"      Avg Classical Fidelity: {metrics['avg_classical_fidelity']:.6f}\n")
                if 'avg_f1_score' in metrics:
                    f.write(f"      Avg F1 Score: {metrics['avg_f1_score']:.6f}\n")
                if 'avg_mae' in metrics:
                    f.write(f"      Avg MAE: {metrics['avg_mae']:.6f}\n")
                if 'avg_tvd_topk' in metrics:
                    f.write(f"      Avg TVD Top-k: {metrics['avg_tvd_topk']:.6f}\n")
                if 'avg_search_acc' in metrics:
                    f.write(f"      Avg Search Accuracy: {metrics['avg_search_acc']:.2%}\n")
    print(f"Summary saved to {summary_file}")

def print_evaluation_results(results: Dict[str, float]):
    """
    Print formatted evaluation results.
    
    Args:
        results: Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("QUANTUM GATE EVALUATION RESULTS")
    print("="*60)
    print(f"  Total Samples: {results['total_samples']}")
    print(f"  Average Tokens: {results['avg_tokens']:.2f}")
    print(f"  Average Fidelity (All Steps): {results['avg_fidelity']:.6f}")
    print(f"  Average Final Fidelity (Last Step): {results['avg_final_fidelity']:.6f}")
    print(f"  Average Classical Fidelity: {results['avg_classical_fidelity']:.6f}")
    print(f"  Average F1 Score: {results['avg_f1_score']:.6f}")
    print(f"  Average MAE: {results['avg_mae']:.6f}")
    print(f"  Average TVD Top-k (k=15): {results['avg_tvd_topk']:.6f}")
    print(f"  Average Search Accuracy (Marked States): {results['avg_search_accuracy']:.2%}")
    print(f"  Average Format Accuracy: {results['avg_format_accuracy']:.2%}")
    print(f"  Average Reasoning Format Accuracy: {results['avg_reasoning_format_accuracy']:.2%}")
    
    # Format criteria breakdown
    criteria_names = [
        "All basis lengths match num_qubits",
        "Basis format is binary string",
        "No duplicate basis states",
        "All probabilities sum in range [0, 1]",
        "All probabilities non-negative",
        "Probability has ≤ 3 decimal places",
        "No more than 15 entries in the distribution"
    ]
    for i in range(1, 8):
        if f'format_criteria_{i}_accuracy' in results:
            print(f"  Format Criteria {i} ({criteria_names[i-1]}) Accuracy: {results[f'format_criteria_{i}_accuracy']:.2%}")
    
    # Reasoning format criteria breakdown
    reasoning_criteria_names = [
        "Wrapped in <circuit_reasoning>...</circuit_reasoning> tags",
        "Each step contains <quantum_state> tags",
        "Measurement outcome probability section present",
        "Each quantum state length equals 2^num_qubits"
    ]
    for i in range(1, 5):
        if f'reasoning_format_criteria_{i}_accuracy' in results:
            print(f"  Reasoning Format Criteria {i} ({reasoning_criteria_names[i-1]}) Accuracy: {results[f'reasoning_format_criteria_{i}_accuracy']:.2%}")
    
    print(f"  Parse Success Rate: {results['parse_success_rate']:.2%}")
    print(f"  Classical Parse Success Rate: {results['classical_parse_success_rate']:.2%}")
    print(f"  JSON Parse Success Rate: {results['json_parse_success_rate']:.2%}")
    print(f"  Perfect Fidelity (All Steps): {results['perfect_fidelity_count']} ({results['perfect_fidelity_rate']:.2%})")
    print(f"  Perfect Final Fidelity (Last Step): {results['perfect_final_fidelity_count']} ({results['perfect_final_fidelity_rate']:.2%})")
    print(f"  Perfect Classical Fidelity: {results['perfect_classical_fidelity_count']} ({results['perfect_classical_fidelity_rate']:.2%})")
    print(f"  Perfect Match (F1=1.0 & MAE<{MAE_THRESHOLD}): {results['perfect_match_count']} ({results['perfect_match_rate']:.2%})")
    print(f"  Perfect Match (F1=1.0 & MAE<0.01): {results['perfect_match_count_mae_0_01']} ({results['perfect_match_rate_mae_0_01']:.2%})")
    print(f"  Perfect Match (F1=1.0 & MAE<0.05): {results['perfect_match_count_mae_0_05']} ({results['perfect_match_rate_mae_0_05']:.2%})")
    print(f"  Perfect Match (F1=1.0 & MAE<0.1): {results['perfect_match_count_mae_0_1']} ({results['perfect_match_rate_mae_0_1']:.2%})")
    print(f"  Perfect Match (F1=1.0 & TVD<0.01): {results['perfect_match_count_tvd_0_01']} ({results['perfect_match_rate_tvd_0_01']:.2%})")
    print(f"  Perfect Match (F1=1.0 & TVD<0.05): {results['perfect_match_count_tvd_0_05']} ({results['perfect_match_rate_tvd_0_05']:.2%})")
    print(f"  Perfect Match (F1=1.0 & TVD<0.1): {results['perfect_match_count_tvd_0_1']} ({results['perfect_match_rate_tvd_0_1']:.2%})")
    print(f"  Perfect Match (F1=1.0 & TVD<0.2): {results['perfect_match_count_tvd_0_2']} ({results['perfect_match_rate_tvd_0_2']:.2%})")
    print(f"  Perfect Match (F1=1.0 & TVD=0): {results['perfect_match_count_tvd_0']} ({results['perfect_match_rate_tvd_0']:.2%})")
    print(f"\n  TOKEN EFFICIENCY (Perfect Match % / Avg Tokens per 1K):")
    print(f"    MAE<0.005: {results.get('token_efficiency_mae_0_005', 0.0):.4f}")
    print(f"    MAE<0.01: {results.get('token_efficiency_mae_0_01', 0.0):.4f}")
    print(f"    MAE<0.05: {results.get('token_efficiency_mae_0_05', 0.0):.4f}")
    print(f"    MAE<0.1: {results.get('token_efficiency_mae_0_1', 0.0):.4f}")
    print(f"    TVD<0.01: {results.get('token_efficiency_tvd_0_01', 0.0):.4f}")
    print(f"    TVD<0.05: {results.get('token_efficiency_tvd_0_05', 0.0):.4f}")
    print(f"    TVD<0.1: {results.get('token_efficiency_tvd_0_1', 0.0):.4f}")
    print(f"    TVD<0.2: {results.get('token_efficiency_tvd_0_2', 0.0):.4f}")
    print(f"    TVD=0: {results.get('token_efficiency_tvd_0', 0.0):.4f}")
    
    # Per-step fidelity
    if 'step_fidelities' in results and results['step_fidelities']:
        print("\n  PER-STEP FIDELITY:")
        for step_name in sorted([k for k in results['step_fidelities'].keys() if not k.endswith('_count')]):
            fidelity = results['step_fidelities'][step_name]
            count = results['step_fidelities'].get(f'{step_name}_count', 0)
            print(f"    {step_name}: {fidelity:.6f} (n={count})")
    
    # Per-step fidelity by number of qubits
    if 'step_fidelities_by_num_qubits' in results and results['step_fidelities_by_num_qubits']:
        print("\n  PER-STEP FIDELITY BY NUMBER OF QUBITS:")
        for qubits_key, step_fidelities in sorted(results['step_fidelities_by_num_qubits'].items(), key=lambda x: int(x[0].split('_')[0])):
            print(f"    {qubits_key}:")

            for step_name in sorted([k for k in step_fidelities.keys() if not k.endswith('_count')]):
                fidelity = step_fidelities[step_name]
                count = step_fidelities.get(f'{step_name}_count', 0)
                print(f"      {step_name}: {fidelity:.6f} (n={count})")
    
    # Parse statistics by number of qubits
    if 'parse_stats_by_qubits' in results and results['parse_stats_by_qubits']:
        print("\n  QUANTUM STATE PARSE STATISTICS BY NUMBER OF QUBITS:")
        for qubits_key, stats in sorted(results['parse_stats_by_qubits'].items(), key=lambda x: int(x[0].split('_')[0])):
            print(f"    {qubits_key}: {stats['parsed']}/{stats['total']} ({stats['parse_rate']:.2%})")
    
    # Metrics by circuit depth
    if 'metrics_by_circuit_depth' in results and results['metrics_by_circuit_depth']:
        print("\n  METRICS BY CIRCUIT DEPTH:")
        for depth_bin, metrics in sorted(results['metrics_by_circuit_depth'].items()):
            print(f"    Depth {depth_bin} (n={metrics.get('count', 0)}):")
            if 'avg_fidelity' in metrics:
                print(f"      Avg Fidelity: {metrics['avg_fidelity']:.6f}")
            if 'avg_final_fidelity' in metrics:
                print(f"      Avg Final Fidelity: {metrics['avg_final_fidelity']:.6f}")
            if 'avg_classical_fidelity' in metrics:
                print(f"      Avg Classical Fidelity: {metrics['avg_classical_fidelity']:.6f}")
            if 'avg_f1_score' in metrics:
                print(f"      Avg F1 Score: {metrics['avg_f1_score']:.6f}")
            if 'avg_mae' in metrics:
                print(f"      Avg MAE: {metrics['avg_mae']:.6f}")
            if 'avg_tvd_topk' in metrics:
                print(f"      Avg TVD Top-k: {metrics['avg_tvd_topk']:.6f}")
            if 'avg_search_acc' in metrics:
                print(f"      Avg Search Accuracy: {metrics['avg_search_acc']:.2%}")
    
    # Metrics by number of qubits
    if 'metrics_by_num_qubits' in results and results['metrics_by_num_qubits']:
        print("\n  METRICS BY NUMBER OF QUBITS:")
        for qubits_key, metrics in sorted(results['metrics_by_num_qubits'].items(), key=lambda x: int(x[0].split('_')[0])):
            print(f"    {qubits_key} (n={metrics.get('count', 0)}):")
            if 'avg_fidelity' in metrics:
                print(f"      Avg Fidelity: {metrics['avg_fidelity']:.6f}")
            if 'avg_final_fidelity' in metrics:
                print(f"      Avg Final Fidelity: {metrics['avg_final_fidelity']:.6f}")
            if 'avg_classical_fidelity' in metrics:
                print(f"      Avg Classical Fidelity: {metrics['avg_classical_fidelity']:.6f}")
            if 'avg_f1_score' in metrics:
                print(f"      Avg F1 Score: {metrics['avg_f1_score']:.6f}")
            if 'avg_mae' in metrics:
                print(f"      Avg MAE: {metrics['avg_mae']:.6f}")
            if 'avg_tvd_topk' in metrics:
                print(f"      Avg TVD Top-k: {metrics['avg_tvd_topk']:.6f}")
            if 'avg_search_acc' in metrics:
                print(f"      Avg Search Accuracy: {metrics['avg_search_acc']:.2%}")
    
    # Metrics by number of gates
    if 'metrics_by_num_gates' in results and results['metrics_by_num_gates']:
        print("\n  METRICS BY NUMBER OF GATES:")
        for gates_bin, metrics in sorted(results['metrics_by_num_gates'].items()):
            print(f"    Gates {gates_bin} (n={metrics.get('count', 0)}):")
            if 'avg_fidelity' in metrics:
                print(f"      Avg Fidelity: {metrics['avg_fidelity']:.6f}")
            if 'avg_final_fidelity' in metrics:
                print(f"      Avg Final Fidelity: {metrics['avg_final_fidelity']:.6f}")
            if 'avg_classical_fidelity' in metrics:
                print(f"      Avg Classical Fidelity: {metrics['avg_classical_fidelity']:.6f}")
            if 'avg_f1_score' in metrics:
                print(f"      Avg F1 Score: {metrics['avg_f1_score']:.6f}")
            if 'avg_mae' in metrics:
                print(f"      Avg MAE: {metrics['avg_mae']:.6f}")
            if 'avg_tvd_topk' in metrics:
                print(f"      Avg TVD Top-k: {metrics['avg_tvd_topk']:.6f}")
            if 'avg_search_acc' in metrics:
                print(f"      Avg Search Accuracy: {metrics['avg_search_acc']:.2%}")

def main():
    """
    Main function to evaluate quantum gate prediction results.
    """
    parser = argparse.ArgumentParser(description="Evaluate quantum gate prediction results.")
    parser.add_argument('--input_path', '--parquet_path', dest='input_path', type=str, default="/scratch3/ip004/inference/results/grover_gates_py_combined_qwen3_8b/grover_gates_py_test_1.parquet", help='Path to the input parquet file with evaluation results.')
    parser.add_argument('--output_path', type=str, required=True, help='Output path prefix for generated files.')
    parser.add_argument('--model_name', type=str, default="/scratch3/ip004/rl_experiment/verl/Qwen/Qwen3-8B-special", help='Model name for tokenizer.')
    args = parser.parse_args()
    
    # Load data
    df = load_eval_results(args.input_path)
    
    # Evaluate overall performance
    results = evaluate_model_performance(df, model_name=args.model_name, store_individual=True)
    
    # Print results to console
    print_evaluation_results(results)
    
    # Normalize output prefix from output_path
    output_prefix, ext = os.path.splitext(args.output_path)
    if ext.lower() not in {"", ".csv"}:
        output_prefix = args.output_path
    if not output_prefix:
        output_prefix = args.output_path
    
    # Save three files: responses, CSV, and summary
    save_individual_results(results, output_prefix, args.input_path, args.model_name)
    
    print(f"\nEvaluation complete! Files saved with prefix: {output_prefix}")

if __name__ == "__main__":
    main()
