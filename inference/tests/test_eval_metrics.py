import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_metrics import parse_response, get_top_k_probs
"""
    Dataframe columns:
    df["responses"] : model generated response
    df["reward_model"]["msb_measurement_probabilities"] : ground truth response
"""


@pytest.mark.parametrize("response, expected", [
    ("The predicted probabilities are: {\"00\": 0.1, \"01\": 0.2, \"10\": 0.3, \"11\": 0.4}", {"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}),
    ("The predicted probabilities are: 00: 0.1, 01: 0.2, 10: 0.3, 11: 0.4", {}),
    ("No probabilities provided", {}),
    ("The predicted probabilities are: {\"00\": \"not a number\", \"01\": 0.2}", {}),
    ("The predicted probabilities are: {\"00\": -0.1, \"01\": 0.2}", {}),
    ("The predicted probabilities are: {\"00\": 0.1, \"01\": 1.5}", {}),
    ("The first predicted probabilities are: {\"00\": 0.1, \"01\": 0.2} and the second predicted probabilities are: {\"10\": 0.3, \"11\": 0.4}", {"10": 0.3, "11": 0.4}),
])
def test_parse_response(response, expected):
    assert parse_response(response) == expected

@pytest.mark.parametrize("evaluation_results, k, expected", [
    ({"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}, 2, {"11": 0.4, "10": 0.3}),
    ({"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}, 3, {"11": 0.4, "10": 0.3, "01": 0.2}),
    ({"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}, 5, {"11": 0.4, "10": 0.3, "01": 0.2, "00": 0.1}),
    ({}, 2, {}),
    ({"00": 0.1, "01": 0.2}, 3, {"01": 0.2, "00": 0.1}),
])
def test_get_top_k_probs(evaluation_results, k, expected):
    assert get_top_k_probs(evaluation_results, k) == expected