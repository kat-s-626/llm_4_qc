import pytest
import pandas as pd
import re
import json
import os
from config.paths import RANDOM_SFT_DIR
from config.constants import (
    DATASET_NUM_QUBITS,
    DATASET_CIRCUIT_DEPTH,
    DATASET_CIRCUIT_HASH,
    DATASET_GATES_LIST,
    DATASET_PYTHON_CODE,
    DATASET_LSB_MEASUREMENT_PROBABILITIES,
    DATASET_MSB_MEASUREMENT_PROBABILITIES,
)
home = os.environ.get("HOME")



@pytest.fixture(params=[
    f"{RANDOM_SFT_DIR}/train.parquet",
    f"{RANDOM_SFT_DIR}/test_1.parquet",
    f"{RANDOM_SFT_DIR}/test_2.parquet",
    f"{RANDOM_SFT_DIR}/test_3.parquet",
])
def parquet_path(request):
    """Parametrized fixture to test all parquet datasets."""
    return request.param    



def test_num_entries(parquet_path):
    """Check that the parquet files have the expected number of entries."""
    import pandas as pd
    
    expected_counts = {
        f"{RANDOM_SFT_DIR}/train.parquet": 100000,
        f"{RANDOM_SFT_DIR}/test_1.parquet": 10000,
        f"{RANDOM_SFT_DIR}/test_2.parquet": 4000,
        f"{RANDOM_SFT_DIR}/test_3.parquet": 4000,
    }
    
    df = pd.read_parquet(parquet_path)
    actual_count = len(df)
    
    expected_count = expected_counts.get(parquet_path, None)
    assert expected_count is not None, f"No expected count defined for {parquet_path}"
    
    assert actual_count == expected_count, (
        f"Parquet file {parquet_path} has {actual_count} entries, "
        f"but expected {expected_count}."
    )

    assert actual_count == expected_count, (
        f"Parquet file {parquet_path} has {actual_count} entries, "
        f"but expected {expected_count}."
    )

def test_data_fields(parquet_path):
    """Check that the data fields in the parquet files are correctly formatted."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    
    required_fields = {
        "prompt": str,
        "completion": str,
    }
    
    extra_info_fields = {
        "split": str,
        "index": int,
        DATASET_NUM_QUBITS: int,
        DATASET_CIRCUIT_DEPTH: int,
        "num_gates": int,
        DATASET_CIRCUIT_HASH: str,
        DATASET_GATES_LIST: str,
        DATASET_PYTHON_CODE: str,
        DATASET_LSB_MEASUREMENT_PROBABILITIES: str,
        DATASET_MSB_MEASUREMENT_PROBABILITIES: str,
    }
    
    for i, row in df.iterrows():
        for field, expected_type in required_fields.items():
            assert field in row, f"Entry {i} in {parquet_path} is missing required field '{field}'."
            assert isinstance(row[field], expected_type), (
                f"Entry {i} in {parquet_path} has field '{field}' of type {type(row[field])}, "
                f"but expected type {expected_type}."
            )
        
        for field, expected_type in extra_info_fields.items():
            assert field in row["extra_info"], f"Entry {i} in {parquet_path} is missing extra_info field '{field}'."
            assert isinstance(row["extra_info"][field], expected_type), (
                f"Entry {i} in {parquet_path} has extra_info field '{field}' of type {type(row['extra_info'][field])}, "
                f"but expected type {expected_type}."
            )

def test_prompt_format(parquet_path):
    """Check that the natural language descriptions in the parquet files follow the expected format."""
    import pandas as pd
    
    df = pd.read_parquet(parquet_path)
    
    for i, nl_desc in enumerate(df["prompt"]):
        assert isinstance(nl_desc, str), f"Entry {i} in {parquet_path} has prompt of type {type(nl_desc)}, but expected str."
        assert nl_desc.startswith("Simulate this quantum circuit."), f"Entry {i} in {parquet_path} has prompt that does not start with expected phrase."

def test_completion_format(parquet_path):
    PROB_LINE_RE = re.compile(r"\|([01]+)>.*=\s*([0-9.]+)")
    FLOAT_TOL = 1e-3

    def _parse_prob_block(completion: str) -> dict[str, float]:
        """Extract {state: probability} from the completion string."""
        return {
            m.group(1): float(m.group(2))
            for m in PROB_LINE_RE.finditer(completion)
        }

    """Parsed probabilities in completion match extra_info[msb_measurement_probabilities]."""
    df = pd.read_parquet(parquet_path)

    for row_idx, row in df.iterrows():
        parsed = _parse_prob_block(row["completion"])
        expected = json.loads(row["extra_info"][DATASET_MSB_MEASUREMENT_PROBABILITIES])

        for state, expected_prob in expected.items():
            assert state in parsed, (
                f"Row {row_idx}: state |{state}> not found in completion"
            )
            assert abs(parsed[state] - float(expected_prob)) <= FLOAT_TOL, (
                f"Row {row_idx}, state |{state}>: "
                f"parsed={parsed[state]:.6f}, stored={float(expected_prob):.6f}"
            )

def test_completion_probabilities_format(parquet_path):
    """Check that the probabilities in the completion are correctly formatted."""
    
    # Get after </circuit_reasoning> tag
    for row_idx, row in pd.read_parquet(parquet_path).iterrows():
        completion = row["completion"]
        if "</circuit_reasoning>" not in completion:
            assert False, f"Row {row_idx} in {parquet_path} is missing </circuit_reasoning> tag."
        
        prob_block = completion.split("</circuit_reasoning>")[-1].strip()
        assert prob_block.startswith("{") and prob_block.endswith("}"), (
            f"Row {row_idx} in {parquet_path} has probabilities block that does not start with '{{' and end with '}}'."
        )
        
        # Test each entries no more than 3 significnat places
        prob_dict = json.loads(prob_block)
        for state, prob in prob_dict.items():
            assert isinstance(state, str), f"Row {row_idx} in {parquet_path} has state {state} of type {type(state)}, but expected str."
            assert isinstance(prob, (float, int)), f"Row {row_idx} in {parquet_path} has probability {prob} of type {type(prob)}, but expected float."
            assert 0 <= prob <= 1, f"Row {row_idx} in {parquet_path} has probability {prob} outside of [0, 1]."
            assert abs(prob - round(prob, 3)) < 1e-6, f"Row {row_idx} in {parquet_path} has probability {prob} that is not rounded to 3 decimal places."


        # Also check that the probabilities are sorted in descending order
        probs = list(prob_dict.values())
        assert all(probs[i] >= probs[i+1] for i in range(len(probs)-1)), f"Row {row_idx} in {parquet_path} has probabilities that are not sorted in descending order."