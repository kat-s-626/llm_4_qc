import pytest
import re
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.eval import (compute_fidelity, 
                        parse_all_quantum_states, 
                        parse_probability_distribution, 
                        compute_classical_fidelity, 
                        parse_component_value,
                        reasoning_format_accuracy,
                        compute_step_by_step_fidelity,
                        parse_real_value)
import pandas as pd
import dotenv
import os



dotenv.load_dotenv()
RESULT_DIR = os.getenv("RESULT_DIR")
TESTING_OUTPUT_FILE = f"{RESULT_DIR}/grover_sets/sft_datasets/qwen3_8b_special/test_1_output_0.parquet"


def data_instance():
    df = pd.read_parquet(TESTING_OUTPUT_FILE)
    return df.iloc[0].to_dict()

class TestFormatCriteria:
    test_cases = [
    pytest.param({"input": "<circuit_reasoning><quantum_state>[0.707, 0, 0, 0.707]</quantum_state></quantum_states></circuit_reasoning>{\"\"}",
                  "expected_accuracy": 0.0, "num_qubits": 2, "expected_criteria": [True, True, False, True]}, id="missing_final_calculation"),

    pytest.param({"input": "<circuit_reasoning><quantum_state>[0.707, 0, 0, 0.707]</quantum_state>",
                  "expected_accuracy": 0.0, "num_qubits": 2, "expected_criteria": [False, True, False, True]}, id="missing_closing_tag"),
    pytest.param({"input": "<circuit_reasoning><quantum_state>[1, 0, 0, 0]</quantum_state>The probability distribution of measurement outcome is: |00>: |1|^2 = 1.000</circuit_reasoning>",
                  "expected_accuracy": 1.0, "num_qubits": 2, "expected_criteria": [True, True, True, True]}, id="Correct_case")
    ]

    dataset_case = [
        pytest.param({"input": data_instance()["responses"], "expected_accuracy": 0.0, "num_qubits": 5, "expected_criteria": [True, True, True, False]}, id="dataset_case")
    ]
    
    @pytest.mark.parametrize("test_case", test_cases + dataset_case)
    def test_reasoning_format_accuracy(self, test_case):
        response = test_case["input"]
        expected_accuracy = test_case["expected_accuracy"]
        overall_accuracy, criteria = reasoning_format_accuracy(response, ground_truth=response, num_qubits=test_case["num_qubits"]) 

        assert isinstance(criteria, list) and len(criteria) == 4, "Criteria list should be a list of 4 booleans"
        assert all(isinstance(c, bool) for c in criteria), "Each criteria entry should be a boolean"

        assert criteria == test_case["expected_criteria"], f"Criteria should be {test_case['expected_criteria']} but got {criteria}"    
        assert overall_accuracy == expected_accuracy, f"Accuracy should be {expected_accuracy} but got {overall_accuracy}"



class TestEvalFidelity:
    test_cases = [
        pytest.param({
            "response": "<circuit_reasoning><quantum_state>[1, 0, 0, 0]</quantum_state>The probability distribution of measurement outcome is: |00>: |1|^2 = 1.000</circuit_reasoning>",
            "ground_truth": "<circuit_reasoning><quantum_state>[1, 0, 0, 0]</quantum_state>The probability distribution of measurement outcome is: |00>: |1|^2 = 1.000</circuit_reasoning>",
            "expected_state_count": 1,
            "expected_fidelity": 1.0,
            "expected_prob_dist": {"00": 1.0},
            "expected_classical_fidelity": 1.0,
        }, id="perfect_match"),
        pytest.param({
            "response": "<circuit_reasoning><quantum_state>[0, 1, 0, 0]</quantum_state>The probability distribution of measurement outcome is: |01>: |1|^2 = 1.000</circuit_reasoning>",
            "ground_truth": "<circuit_reasoning><quantum_state>[1, 0, 0, 0]</quantum_state>The probability distribution of measurement outcome is: |00>: |1|^2 = 1.000</circuit_reasoning>",
            "expected_state_count": 1,
            "expected_fidelity": 0.0,
            "expected_prob_dist": {"01": 1.0},
            "expected_classical_fidelity": 0.0,
        }, id="orthogonal_states"),
    ]

    dataset_case = [
        pytest.param({
            "response": data_instance()["responses"],
            "ground_truth": data_instance()["completion"],
            "expected_state_count": data_instance()["reward_model"]["num_gates"],
            "expected_fidelity_min": 0.0,
            "expected_fidelity_max": 1.01,
            "expected_has_prob_dist": True,
            "expected_classical_fidelity_min": 0.0,
            "expected_classical_fidelity_max": 1.0,
        }, id="dataset_case"),
    ]

    @pytest.mark.parametrize("test_case", test_cases + dataset_case)
    def test_parse_all_quantum_states(self, test_case):
        response = test_case["response"]
        states = parse_all_quantum_states(response)

        assert isinstance(states, list), "Output should be a list"
        assert len(states) == test_case["expected_state_count"], "Number of parsed states should match expected state count"

    @pytest.mark.parametrize("test_case", test_cases + dataset_case)
    def test_compute_fidelity(self, test_case):
        response_states = parse_all_quantum_states(test_case["response"])
        ground_truth_states = parse_all_quantum_states(test_case["ground_truth"])

        assert response_states and ground_truth_states, "Both response and ground truth must contain at least one quantum state"

        fidelity = compute_fidelity(response_states[0], ground_truth_states[0])

        assert isinstance(fidelity, float), "Fidelity should be a float"
        if "expected_fidelity" in test_case:
            assert fidelity == pytest.approx(test_case["expected_fidelity"]), f"Fidelity should be {test_case['expected_fidelity']} but got {fidelity}"
        else:
            assert test_case["expected_fidelity_min"] <= fidelity <= test_case["expected_fidelity_max"], "Fidelity should be within expected range"
    
    @pytest.mark.parametrize("test_case", test_cases + dataset_case)
    def test_parse_parse_probability_distribution(self, test_case):
        response = test_case["response"]
        prob_dist = parse_probability_distribution(response)

        if "expected_prob_dist" in test_case:
            assert prob_dist == test_case["expected_prob_dist"], f"Probability distribution should be {test_case['expected_prob_dist']} but got {prob_dist}"
        else:
            assert isinstance(prob_dist, dict), "Output should be a dictionary"
            assert bool(prob_dist) == test_case["expected_has_prob_dist"], "Probability distribution presence should match expectation"

        if prob_dist:
            for key, value in prob_dist.items():
                assert isinstance(key, str), "Keys should be strings"
                assert isinstance(value, (int, float)), "Values should be numeric"
                assert 0.0 <= value <= 1.0, "Probabilities should be between 0 and 1"

    @pytest.mark.parametrize("test_case", test_cases + dataset_case)
    def test_compute_classical_fidelity(self, test_case):
        response_prob_dist = parse_probability_distribution(test_case["response"])
        ground_truth_prob_dist = parse_probability_distribution(test_case["ground_truth"])

        classical_fidelity = compute_classical_fidelity(response_prob_dist, ground_truth_prob_dist)

        assert isinstance(classical_fidelity, float), "Classical fidelity should be a float"
        if "expected_classical_fidelity" in test_case:
            assert classical_fidelity == pytest.approx(test_case["expected_classical_fidelity"]), f"Classical fidelity should be {test_case['expected_classical_fidelity']} but got {classical_fidelity}"
        else:
            assert test_case["expected_classical_fidelity_min"] <= classical_fidelity <= test_case["expected_classical_fidelity_max"], "Classical fidelity should be within expected range"

class TestStepByStep:
    test_cases = [
        pytest.param({"pred_states": [[0.707, 0, 0, 0.707], [0.5, 0.5, 0.5, 0.5]], 
         "truth_states": [[0.707, 0, 0, 0.707], [0.5, 0.5, 0.5, 0.5]], 
         "num_qubits": 2,
         "expected_step_by_step_fidelity": {0: [1.0], 1: [1.0]},
         "expected_fidelity_score": 1.0,
         "expected_final_fidelity_score": 1.0}, id="perfect_match"),
        
        pytest.param({"pred_states": [[0.707, 0, 0, 0.707], [0, 0.707, 0, 0.707]], 
         "truth_states": [[0, 0.707, 0, 0.707], [0, 0.707, 0, 0.707]], 
         "num_qubits": 2,
         "expected_step_by_step_fidelity": {0: [0.25], 1: [1.0]},
         "expected_fidelity_score": 0.625,
         "expected_final_fidelity_score": 1.0}, id="one_step_wrong"),
        
        pytest.param({"pred_states": [[0.707, 0, 0, 0.707]], 
         "truth_states": [[0.707, 0, 0, 0.707], [0, 0.707, 0, 0.707]], 
         "num_qubits": 2,
         "expected_step_by_step_fidelity": {0: [1.0], 1: [0.0]},
         "expected_fidelity_score": 0.5,
         "expected_final_fidelity_score": 0.0}, id="one_step_missing"),
    ]
    
    dataset_case = [
        {"pred_states": parse_all_quantum_states(data_instance()["responses"]),
         "truth_states": parse_all_quantum_states(data_instance()["completion"]),
         "num_qubits": 5,
         "expected_step_by_step_fidelity": {0:[1.0], 1:[1.0], 2:[1.0], 3:[1.0], 4:[1.0], 5:[1.0], 
                                            6:[0.0], 7:[1.0], 8: [1.0], 9: [1.0], 10: [1.0], 11: [1.0]},  # We will check if it's a list of correct length and values between 0 and 1
         "expected_fidelity_score": 0.917,  # We will check if it's between 0 and 1
         "expected_final_fidelity_score": 1.0,  # We will check if it's between 0 and 1
         "id": "dataset_case"},
    ]

    @pytest.mark.parametrize("test_case", test_cases + dataset_case)
    def test_compute_step_by_step_fidelity(self, test_case):    
        result = compute_step_by_step_fidelity(test_case["pred_states"], test_case["truth_states"], test_case["num_qubits"])

        assert isinstance(result, dict), "Output should be a dictionary"
        assert "step_fidelity_dict" in result and "fidelity_score" in result and "final_fidelity_score" in result, "Output dictionary should contain 'step_fidelity_dict', 'fidelity_score', and 'final_fidelity_score' keys"
        

        step_fidelity_dict = result["step_fidelity_dict"]
        step_fidelity_dict = {step_idx: [round(f, 3) for f in fidelities] for step_idx, fidelities in step_fidelity_dict.items()}

        fidelity_score = round(result["fidelity_score"], 3)
        final_fidelity_score = round(result["final_fidelity_score"], 3)

        assert step_fidelity_dict == test_case["expected_step_by_step_fidelity"]
        assert fidelity_score == test_case["expected_fidelity_score"]
        assert final_fidelity_score == test_case["expected_final_fidelity_score"]


class TestParseNumber:
    test_cases = [
        pytest.param({"input": "√2/2", "expected_output": 0.707}, id="sqrt_fraction"),
        pytest.param({"input": "2/√2", "expected_output": 1.414}, id="fraction_sqrt"),
        pytest.param({"input": "0.5/√2", "expected_output": 0.354}, id="decimal_fraction_sqrt"),
        pytest.param({"input": "2/(4√2)", "expected_output": 0.354}, id="fraction_with_sqrt_in_denominator"),
    ]

    @pytest.mark.parametrize("test_case", test_cases)
    def test_parse_real_value(self, test_case):
        output = parse_real_value(test_case["input"])

        assert output == pytest.approx(test_case["expected_output"], abs=1e-3), f"Output should be approximately {test_case['expected_output']} but got {output}"
    