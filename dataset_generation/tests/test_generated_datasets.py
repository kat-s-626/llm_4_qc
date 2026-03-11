import json
import re
import pytest
from pathlib import Path
from collections import Counter
from config.constants import (
    DATASET_REQUIRED_FIELDS,
    DATASET_NUM_QUBITS,
    DATASET_CIRCUIT_DEPTH,
    DATASET_GATES_LIST,
    DATASET_CIRCUIT_HASH,
    DATASET_LSB_MEASUREMENT_PROBABILITIES,
    DATASET_MSB_MEASUREMENT_PROBABILITIES,
    DATASET_PYTHON_CODE,
    DATASET_NL_DESCRIPTION,
    DATASET_EXTRA_INFO,
)


@pytest.fixture
def dataset(request):
    """Load dataset from file path provided by test parameter."""
    file_path = request.param
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


@pytest.fixture(params=[
    "/scratch3/ip004/data/grover_sets/train_updated.jsonl",
    "/scratch3/ip004/data/grover_sets/test_1_updated.jsonl",
    "/scratch3/ip004/data/grover_sets/test_2_updated.jsonl",
    "/scratch3/ip004/data/grover_sets/test_3_updated.jsonl",
    "/scratch3/ip004/data/rotation_sets/train_updated.jsonl",
    "/scratch3/ip004/data/rotation_sets/test_1_updated.jsonl",
    "/scratch3/ip004/data/rotation_sets/test_2_updated.jsonl",
    "/scratch3/ip004/data/rotation_sets/test_3_updated.jsonl",
])
def dataset_path(request):
    """Parametrized fixture to test all datasets."""
    return request.param


def load_dataset(file_path):
    """Helper function to load dataset."""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


class TestDatasetStructure:
    """Test the structure and format of the dataset."""
    
    def test_dataset_not_empty(self, dataset_path):
        """Check that dataset contains at least one entry."""
        data = load_dataset(dataset_path)
        assert len(data) > 0, "Dataset is empty."
    
    def test_all_required_fields_present(self, dataset_path):
        """Check that all required fields are present in each entry."""
        data = load_dataset(dataset_path)
        required_fields = DATASET_REQUIRED_FIELDS
        
        for i, entry in enumerate(data):
            for field in required_fields:
                assert field in entry, f"Entry {i}: Missing '{field}' field."
    
    def test_field_types(self, dataset_path):
        """Check that fields have correct types."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            assert isinstance(entry[DATASET_NUM_QUBITS], int), f"Entry {i}: 'num_qubits' should be int"
            assert isinstance(entry[DATASET_CIRCUIT_DEPTH], int), f"Entry {i}: 'circuit_depth' should be int"
            assert isinstance(entry[DATASET_GATES_LIST], list), f"Entry {i}: 'gates_list' should be list"
            assert isinstance(entry[DATASET_CIRCUIT_HASH], str), f"Entry {i}: 'circuit_hash' should be str"
            assert isinstance(entry[DATASET_LSB_MEASUREMENT_PROBABILITIES], dict), f"Entry {i}: 'lsb_measurement_probabilities' should be dict"
            assert isinstance(entry[DATASET_MSB_MEASUREMENT_PROBABILITIES], dict), f"Entry {i}: 'msb_measurement_probabilities' should be dict"
            assert isinstance(entry[DATASET_PYTHON_CODE], str), f"Entry {i}: 'python_code' should be str"
            assert isinstance(entry[DATASET_NL_DESCRIPTION], str), f"Entry {i}: 'nl_description' should be str"
            assert isinstance(entry[DATASET_EXTRA_INFO], dict), f"Entry {i}: 'extra_info' should be dict"
    
    def test_natural_language_format(self, dataset_path):
        """Check that natural language description follows expected format."""
        data = load_dataset(dataset_path)

        # There should be exactly one pair of <circuit_reasoning> and </circuit_reasoning> tags, 
        # and exactly n=len(gates_list) pair of <quantum_state> and </quantum_state> tags in each nl_description
        for i, entry in enumerate(data):
            nl_desc = entry[DATASET_NL_DESCRIPTION]
            gates_list = entry[DATASET_GATES_LIST]

            reasoning_start = nl_desc.count("<circuit_reasoning>")
            reasoning_end = nl_desc.count("</circuit_reasoning>")
            state_start_tags = nl_desc.count("<quantum_state>")
            start_end_tags = nl_desc.count("</quantum_state>")

            # make sure the format is in <quantum_state>[...]</quantum_state>
            pattern = r"<quantum_state>\[.*?\]</quantum_state>"

            assert reasoning_start == 1, f"Entry {i}: Expected exactly one <circuit_reasoning> tag, found {reasoning_start}"
            assert reasoning_end == 1, f"Entry {i}: Expected exactly one </circuit_reasoning> tag, found {reasoning_end}"
            assert state_start_tags == len(gates_list), f"Entry {i}: Expected {len(gates_list)} <quantum_state> start tags, found {state_start_tags}"
            assert start_end_tags == len(gates_list), f"Entry {i}: Expected {len(gates_list)} <quantum_state> end tags, found {start_end_tags}"
            assert re.findall(pattern, nl_desc), f"Entry {i}: <quantum_state> tags do not match expected format"


class TestDatasetUniqueness:
    """Test uniqueness constraints in the dataset."""
    
    def test_circuit_hashes_unique(self, dataset_path):
        """Check that all circuit_hash values are unique."""
        data = load_dataset(dataset_path)
        circuit_hashes = [entry[DATASET_CIRCUIT_HASH] for entry in data]
        
        duplicates = [hash for hash, count in Counter(circuit_hashes).items() if count > 1]
        assert len(duplicates) == 0, f"Found duplicate circuit_hashes: {duplicates}"


class TestMeasurementProbabilities:
    """Test measurement probability consistency."""
    
    def test_lsb_msb_keys_consistent(self, dataset_path):
        """Check that lsb and msb measurement probabilities have reversed keys."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            lsb_probs = entry[DATASET_LSB_MEASUREMENT_PROBABILITIES]
            msb_probs = entry[DATASET_MSB_MEASUREMENT_PROBABILITIES]
            circuit_hash = entry[DATASET_CIRCUIT_HASH]
            
            lsb_keys = set(lsb_probs.keys())
            msb_keys = set(msb_probs.keys())
            
            msb_reversed_keys = set(k[::-1] for k in msb_keys)
            lsb_reversed_keys = set(k[::-1] for k in lsb_keys)
            
            # Check same set of keys
            assert lsb_keys == msb_reversed_keys, (
                f"Entry {i} (hash: {circuit_hash}): "
                f"LSB keys {lsb_keys} do not match reversed MSB keys {msb_reversed_keys}"
            )
            assert msb_keys == lsb_reversed_keys, (
                f"Entry {i} (hash: {circuit_hash}): "
                f"MSB keys {msb_keys} do not match reversed LSB keys {lsb_reversed_keys}"
            )
    
    def test_lsb_msb_keys_are_reversed(self, dataset_path):
        """Check that lsb keys are bit-reversed versions of msb keys."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            lsb_probs = entry[DATASET_LSB_MEASUREMENT_PROBABILITIES]
            msb_probs = entry[DATASET_MSB_MEASUREMENT_PROBABILITIES]
            circuit_hash = entry[DATASET_CIRCUIT_HASH]
            num_qubits = entry[DATASET_NUM_QUBITS]
            
            for lsb_key in lsb_probs.keys():
                # Reverse the bit string
                msb_key = lsb_key[::-1]
                
                assert msb_key in msb_probs, (
                    f"Entry {i} (hash: {circuit_hash}): "
                    f"LSB key '{lsb_key}' reversed to '{msb_key}' not found in MSB probabilities"
                )
    
    def test_lsb_msb_probabilities_match(self, dataset_path):
        """Check that probability values match between lsb and msb for corresponding keys."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            lsb_probs = entry[DATASET_LSB_MEASUREMENT_PROBABILITIES]
            msb_probs = entry[DATASET_MSB_MEASUREMENT_PROBABILITIES]
            circuit_hash = entry[DATASET_CIRCUIT_HASH]
            
            for lsb_key, lsb_prob in lsb_probs.items():
                msb_key = lsb_key[::-1]
                msb_prob = msb_probs[msb_key]
                
                assert abs(lsb_prob - msb_prob) < 1e-10, (
                    f"Entry {i} (hash: {circuit_hash}): "
                    f"Probability mismatch for key '{lsb_key}': "
                    f"LSB={lsb_prob}, MSB={msb_prob}"
                )
    
    def test_probabilities_sum_to_one(self, dataset_path):
        """Check that measurement probabilities sum to 1."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            lsb_probs = entry[DATASET_LSB_MEASUREMENT_PROBABILITIES]
            msb_probs = entry[DATASET_MSB_MEASUREMENT_PROBABILITIES]
            circuit_hash = entry[DATASET_CIRCUIT_HASH]
            
            lsb_sum = sum(lsb_probs.values())
            msb_sum = sum(msb_probs.values())
            
            assert abs(lsb_sum - 1.0) < 1e-10, (
                f"Entry {i} (hash: {circuit_hash}): "
                f"LSB probabilities sum to {lsb_sum}, expected 1.0"
            )
            assert abs(msb_sum - 1.0) < 1e-10, (
                f"Entry {i} (hash: {circuit_hash}): "
                f"MSB probabilities sum to {msb_sum}, expected 1.0"
            )


class TestQuantumCircuitProperties:
    """Test quantum circuit specific properties."""
    
    def test_num_qubits_positive(self, dataset_path):
        """Check that num_qubits is positive."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            assert entry[DATASET_NUM_QUBITS] > 0, f"Entry {i}: num_qubits must be positive"
    
    def test_circuit_depth_non_negative(self, dataset_path):
        """Check that circuit_depth is non-negative."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            assert entry[DATASET_CIRCUIT_DEPTH] >= 0, f"Entry {i}: circuit_depth must be non-negative"
    
    def test_gates_list_not_empty(self, dataset_path):
        """Check that gates_list is not empty (if circuit_depth > 0)."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            if entry[DATASET_CIRCUIT_DEPTH] > 0:
                assert len(entry[DATASET_GATES_LIST]) > 0, (
                    f"Entry {i}: gates_list is empty but circuit_depth is {entry[DATASET_CIRCUIT_DEPTH]}"
                )

class TestDatasetCounts:
    
    @pytest.mark.parametrize("dataset_path,length", [
        ("/scratch3/ip004/data/grover_sets/train_updated.jsonl", 100000),
        ("/scratch3/ip004/data/grover_sets/test_1_updated.jsonl", 10000),
        ("/scratch3/ip004/data/grover_sets/test_2_updated.jsonl", 4000),
        ("/scratch3/ip004/data/grover_sets/test_3_updated.jsonl", 4000),
        ("/scratch3/ip004/data/rotation_sets/train_updated.jsonl", 100000), 
        ("/scratch3/ip004/data/rotation_sets/test_1_updated.jsonl", 10000),
        ("/scratch3/ip004/data/rotation_sets/test_2_updated.jsonl", 4000),
        ("/scratch3/ip004/data/rotation_sets/test_3_updated.jsonl", 4000),
    ])

    def test_dataset_counts(self, dataset_path, length):
        """Check that dataset contains expected number of entries."""
        data = load_dataset(dataset_path)
        assert len(data) == length, f"Dataset {dataset_path} contains {len(data)} entries, expected {length}"

class TestNLDesc:
    
    def test_valid_intermediate_states_in_nl_description(self, dataset_path):
        """Check that intermediate quantum states are included in nl_description."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            nl_desc = entry[DATASET_NL_DESCRIPTION]
            
            # Check that there are as many <quantum_state> tags as gates
            state_vector = re.findall(r"<quantum_state>\[(.*?)\]</quantum_state>", nl_desc)
            states = [s.strip() for s in state_vector[0].split(",")] if state_vector else []

            assert all(
                c in "0123456789√()/i+-. "
                for state in states
                for c in state
            ), f"Entry {i}: Invalid characters in quantum state vector in {states}"

    def test_non_parametric_gates_nl_label(self, dataset_path):
        """
        Ensure non-parametric gates are rendered like 'Z gate ...', not 'Z(0) gate ...'
        in step lines containing <quantum_state>.
        """
        data = load_dataset(dataset_path)

        non_parametric_gate_types = {
             "x", "y", "z", "h", 
        }

        for i, entry in enumerate(data):
            nl_desc = entry[DATASET_NL_DESCRIPTION]
            gates_list = entry[DATASET_GATES_LIST]

            step_lines = [line.strip() for line in nl_desc.splitlines() if "<quantum_state>" in line]
            assert len(step_lines) == len(gates_list), (
                f"Entry {i}: step lines count {len(step_lines)} != gates_list count {len(gates_list)}"
            )

            for step_idx, (line, gate) in enumerate(zip(step_lines, gates_list), start=1):
                gate_type = str(gate.get("type", "")).lower()
                params = gate.get("params", None)

                is_non_parametric_instance = (params is None) or (isinstance(params, list) and len(params) == 0)

                if gate_type in non_parametric_gate_types and is_non_parametric_instance:
                    # Reject labels like "Z(0) gate", "H(1) gate", etc. in NL segment.
                    invalid_label_pattern = rf"\b{gate_type.upper()}\s*\([^)]*\)\s+gate\b"
                    assert re.search(invalid_label_pattern, line) is None, (
                        f"Entry {i}, step {step_idx}: non-parametric gate '{gate_type}' "
                        f"should not be parenthesized in NL line: {line}"
                    )

    def test_valid_beginning(self, dataset_path):
        """Check that nl_description starts with expected format."""
        data = load_dataset(dataset_path)
        
        for i, entry in enumerate(data):
            nl_desc = entry[DATASET_NL_DESCRIPTION]
            assert nl_desc.startswith("<circuit_reasoning>Initialize"), f"Entry {i}: nl_description should start with <circuit_reasoning>"



