import pytest
import math
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import HGate, RXGate, RYGate, RZGate, CXGate
from dataset_generator.src.utils.random_rotation_set import (
    generate_random_circuit,
    _select_gate,
    _get_measurement_probabilities,
    _get_circuit_hash,
    _add_gate,
    _reverse_qubit_ordering,
    _get_python_code,
    _get_natural_language_description,
    ROTATION_SET,
    PROBABILITY_DISTRIBUTIONS,
    AMPLITUDE_THRESHOLD,
    PARAMS
)


class TestSelectGate:
    """Test the gate selection logic."""

    def test_single_qubit_only(self):
        """With 1 qubit, only single-qubit gates should be selected."""
        for _ in range(100):
            gate_type = _select_gate(1)
            assert gate_type == 0

    def test_two_qubit_distribution(self):
        """With 2 qubits, should select from single and two-qubit gates."""
        results = [_select_gate(2) for _ in range(1000)]
        assert all(g in [0, 1] for g in results)
        assert 0 in results and 1 in results  # Both types should appear



class TestGenerateRandomRotationSet:
    """Test the main circuit generation function."""

    def test_circuit_creation(self):
        """Circuit should be created with correct number of qubits."""
        num_qubits = 3
        max_gates = 10
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == num_qubits

    def test_max_gates_limit(self):
        """Circuit should not exceed max_num_gates."""
        num_qubits = 3
        max_gates = 15
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        assert len(circuit.data) <= max_gates

    def test_single_qubit_circuit(self):
        """Single qubit circuit should only contain single-qubit gates."""
        num_qubits = 1
        max_gates = 10
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        
        single_qubit_gates = {'rx', 'ry', 'rz', 'h'}
        for instruction in circuit.data:
            assert instruction.operation.name in single_qubit_gates

    def test_gate_types_present(self):
        """Circuit should contain gates from the Rotation set."""
        num_qubits = 4
        max_gates = 50
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        
        valid_gate_names = {
            'rx', 'ry', 'rz', 'h', 'cx'
        }
        
        gate_names = {inst.operation.name for inst in circuit.data}
        assert gate_names.issubset(valid_gate_names)
        assert len(gate_names) > 0  # At least some gates should be present

    def test_circuit_is_executable(self):
        """Generated circuit should be executable and produce valid statevector."""
        num_qubits = 3
        max_gates = 20
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        
        # Should not raise an exception
        statevector = Statevector.from_instruction(circuit)
        assert len(statevector.data) == 2**num_qubits
        # Statevector should be normalized
        assert abs(sum(abs(amp)**2 for amp in statevector.data) - 1.0) < 1e-10

    def test_different_seeds_produce_different_circuits(self):
        """Multiple calls should produce different circuits (probabilistic test)."""
        num_qubits = 3
        max_gates = 10
        circuit1, _, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        circuit2, _, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        
        # With high probability, these should not be identical
        assert qasm3.dumps(circuit1) != qasm3.dumps(circuit2)

    def test_zero_max_gates(self):
        """Circuit with max_gates=0 should be empty."""
        num_qubits = 3
        max_gates = 0
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        assert len(circuit.data) == 0

    def test_large_circuit(self):
        """Should handle larger circuits without issues."""
        num_qubits = 5
        max_gates = 100
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == num_qubits
        assert len(circuit.data) <= max_gates

    def test_two_qubit_gates_structure(self):
        """Two-qubit gates should have distinct control and target."""
        num_qubits = 3
        max_gates = 50
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        
        for instruction in circuit.data:
            if instruction.operation.name in ['cx', 'cz']:
                qubits = [circuit.qubits.index(q) for q in instruction.qubits]
                assert len(qubits) == 2
                assert qubits[0] != qubits[1]  # Control and target should differ


class TestRotationSetConfiguration:
    """Test the configuration constants."""

    def test_rotation_set_contents(self):
        """Rotation set should contain valid gate names."""
        valid_gate_names = {
            'rx', 'ry', 'rz', 'h', 'cx'
        }
        assert set(ROTATION_SET).issubset(valid_gate_names)
        assert len(ROTATION_SET) > 0  # Should not be empty

    def test_probability_distribution(self):
        """Probability distribution should sum to 1.0."""
        assert abs(sum(PROBABILITY_DISTRIBUTIONS) - 1.0) < 1e-10

    def test_parameters_list(self):
        """Parameters list should contain valid rotation angles."""
        for param in PARAMS:
            assert isinstance(param, (int, float))
            assert -2*math.pi <= param <= 2*math.pi

class TestOutputFormat:
    """Test output json format and hash function."""
    
    def test_circuit_hash_consistency(self):
        """Hash function should produce consistent results for the same circuit."""
        num_qubits = 3
        max_gates = 10
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        
        hash1 = _get_circuit_hash(circuit)
        hash2 = _get_circuit_hash(circuit)
        
        assert hash1 == hash2
    
    def test_measurement_probabilities_format(self):
        """Probabilities should be returned in the correct format."""
        num_qubits = 2
        max_gates = 5
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        
        probabilities = _get_measurement_probabilities(circuit)
        assert isinstance(probabilities, dict)
        for key, value in probabilities.items():
            assert isinstance(key, str)  # Keys should be strings
            assert isinstance(value, float)  # Values should be floats
            assert value >= 0.0 and value <= 1.0  # Probabilities should be valid
    
    
    def test_gate_list_structure(self):
        """Gates list should contain correct structure for each gate."""
        num_qubits = 3
        max_gates = 10
        circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, max_gates)
        
        assert isinstance(gates_list, list)
        for gate_info in gates_list:
            assert isinstance(gate_info, dict)
            assert 'type' in gate_info
            assert 'params' in gate_info
            assert 'target_qubits' in gate_info
            assert 'target_gate' in gate_info
            assert 'num_controls' in gate_info
            assert 'num_targets' in gate_info
    
    def test_lsb_first_format(self):
        qc = QuantumCircuit(3)
        qc.x(0)  # This creates |100> state in MSB-first, but should be |001> in LSB-first
        probabilities = _get_measurement_probabilities(qc)
        expected_probabilities = {'001': 1.0}  # LSB-first format
        assert probabilities == expected_probabilities
    
    def test_msb_conversion(self):
       qc = QuantumCircuit(3)
       qc.x(0)  # This creates |100> state in MSB-first
       probabilities = _get_measurement_probabilities(qc)
       reversed_probabilities = _reverse_qubit_ordering(probabilities)
       expected_probabilities = {'100': 1.0}  # MSB-first format
       assert reversed_probabilities == expected_probabilities

class TestPythonCodeGeneration:
    """Test the Python code generation from Qiskit circuits."""
    
    def test_code_generation_consistency(self):
        num_qubits = 2
        gates_list = []
        gates_list.append({"type": "h", "params": None, "target_qubits": [0], "target_gate": None, "num_controls": 0, "num_targets": 1})
        gates_list.append({"type": "cx", "params": None, "target_qubits": [0, 1], "target_gate": None, "num_controls": 1, "num_targets": 1})
        gates_list.append({"type": "rz", "params": [math.pi/4], "target_qubits": [1], "target_gate": None, "num_controls": 0, "num_targets": 1})
        code1 = _get_python_code(num_qubits, gates_list)

        target_code_str = """circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.rz(pi/4, 1)"""
        assert code1.strip() == target_code_str.strip()
    
    def test_natural_language_generation_consistency(self):
        num_qubits = 2
        gates_list = []
        gates_list.append({"type": "h", "params": None, "target_qubits": [0], "target_gate": None, "num_controls": 0, "num_targets": 1})
        gates_list.append({"type": "cx", "params": None, "target_qubits": [0, 1], "target_gate": None, "num_controls": 1, "num_targets": 1})
        gates_list.append({"type": "rz", "params": [math.pi/4], "target_qubits": [1], "target_gate": None, "num_controls": 0, "num_targets": 1})
        nl_description = _get_natural_language_description(num_qubits, gates_list)
        target_nl_str = """Given the initial quantum state |0⟩^⊗2, apply the following quantum gates in sequence:

1. H gate on qubit 0.
2. CX gate controls on qubits [0], targets on qubit 1.
3. RZ(π/4) gate on qubit 1."""
        assert nl_description.strip() == target_nl_str.strip()