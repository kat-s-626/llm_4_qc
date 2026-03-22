import json
import os
import random

from dataset_generator.src.marked_state_sampler import (
    _get_all_bitstrings,
    _random_bitstring,
    _save_sampled_marked_states,
    sample_marked_states,
)

class TestGetAllBitstrings:
    def test_single_qubit(self):
        result = _get_all_bitstrings(1)
        assert result == ['0', '1']

    def test_two_qubits(self):
        result = _get_all_bitstrings(2)
        assert result == ['00', '01', '10', '11']

    def test_three_qubits_length(self):
        result = _get_all_bitstrings(3)
        assert len(result) == 8

    def test_three_qubits_all_unique(self):
        result = _get_all_bitstrings(3)
        assert len(result) == len(set(result))

    def test_zero_padding(self):
        result = _get_all_bitstrings(4)
        assert result[1] == '0001'

    def test_all_strings_correct_length(self):
        for n in range(1, 6):
            result = _get_all_bitstrings(n)
            assert all(len(s) == n for s in result)

    def test_count_correct(self):
        for n in range(1, 6):
            result = _get_all_bitstrings(n)
            assert len(result) == 2 ** n

class TestRandomBitstring:
    def test_correct_length(self):
        for n in [1, 5, 10, 20]:
            result = _random_bitstring(n)
            assert len(result) == n

    def test_only_binary_chars(self):
        result = _random_bitstring(100)
        assert all(c in '01' for c in result)

    def test_returns_string(self):
        result = _random_bitstring(5)
        assert isinstance(result, str)

    def test_length_zero(self):
        result = _random_bitstring(0)
        assert result == ''


class TestSaveSampledMarkedStates:
    def test_saves_file(self, tmp_path):
        filepath = str(tmp_path / "output.json")
        data = [['00', '01'], ['10', '11']]
        _save_sampled_marked_states(data, filepath)
        assert os.path.exists(filepath)

    def test_file_content_correct(self, tmp_path):
        filepath = str(tmp_path / "output.json")
        data = [['00', '01'], ['10', '11']]
        _save_sampled_marked_states(data, filepath)
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_nested_directory(self, tmp_path):
        filepath = str(tmp_path / "nested" / "deep" / "output.json")
        data = [['000']]
        _save_sampled_marked_states(data, filepath)
        assert os.path.exists(filepath)

    def test_saves_empty_list(self, tmp_path):
        filepath = str(tmp_path / "empty.json")
        _save_sampled_marked_states([], filepath)
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == []

    def test_overwrites_existing_file(self, tmp_path):
        filepath = str(tmp_path / "output.json")
        _save_sampled_marked_states([['00']], filepath)
        _save_sampled_marked_states([['11']], filepath)
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == [['11']]


class TestSampleMarkedStates:
    def test_returns_list(self):
        result = sample_marked_states(2, 3, 1, 1, 5)
        assert isinstance(result, list)

    def test_each_element_is_list(self):
        result = sample_marked_states(2, 3, 1, 1, 5)
        for item in result:
            assert isinstance(item, list)

    def test_marked_states_are_strings(self):
        result = sample_marked_states(2, 3, 1, 1, 5)
        for combo in result:
            for state in combo:
                assert isinstance(state, str)

    def test_marked_states_have_correct_length_for_qubits(self):
        result = sample_marked_states(2, 4, 1, 2, 5)
        for combo in result:
            lengths = {len(s) for s in combo}
            assert len(lengths) == 1
            assert lengths.pop() in range(2, 5)

    def test_no_duplicates_within_combination(self):
        result = sample_marked_states(6, 8, 2, 8, 1000)
        for combo in result:
            assert len(combo) == len(set(combo))

    def test_number_of_marked_states_within_range(self):
        min_m, max_m = 1, 2
        result = sample_marked_states(6, 8, min_m, max_m, 1000)
        for combo in result:
            assert min_m <= len(combo) <= max_m

    def test_single_qubit_skipped_if_constraints_invalid(self):
        # For 2 qubits: max_possible = 2^(2-1) - 1 = 1
        # min_marked_states=2 > max_possible=1, so should skip
        result = sample_marked_states(2, 2, 2, 3, 5)
        assert result == []


    def test_small_scale_combinations_valid(self):
        result = sample_marked_states(2, 3, 1, 2, 10)
        assert len(result) > 0

    def test_determinism_with_seed(self):
        random.seed(42)
        result1 = sample_marked_states(2, 4, 1, 2, 5)
        random.seed(42)
        result2 = sample_marked_states(2, 4, 1, 2, 5)
        assert sorted(map(sorted, result1)) == sorted(map(sorted, result2))

    def test_larger_qubits(self):
        result = sample_marked_states(5, 6, 1, 2, 3)
        for combo in result:
            assert len(combo[0]) in [5, 6]

    def test_marked_states_sorted_within_combination(self):
        result = sample_marked_states(2, 4, 1, 2, 5)
        for combo in result:
            assert combo == sorted(combo)

    def test_min_equals_max_qubits(self):
        result = sample_marked_states(3, 3, 1, 1, 5)
        for combo in result:
            assert all(len(s) == 3 for s in combo)