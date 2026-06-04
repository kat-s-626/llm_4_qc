import os
from itertools import islice
import torch
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler


def run_circuit_sampled(circuit, n_samples):
    backend = AerSimulator(max_parallel_threads=os.cpu_count())
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    circuit_isa = pm.run(circuit)
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = n_samples
    result = sampler.run([circuit_isa]).result()
    distribution = result[0].data.meas.get_counts()

    return distribution


def run_circuit_statevector(circuit, significant_digits=4, top_n=30):
    circuit_to_run = circuit.copy()
    circuit_to_run.remove_final_measurements()
    circuit_to_run.save_statevector()

    def _make_simulator(device: str) -> AerSimulator:
        return AerSimulator(method="statevector", device=device)

    tqc = None
    result = None
    for device in (["GPU", "CPU"] if torch.cuda.is_available() else ["CPU"]):
        try:
            simulator = _make_simulator(device)
            tqc = transpile(circuit_to_run, simulator)
            result = simulator.run(tqc).result()
            break
        except (RuntimeError, Exception):
            if device == "CPU":
                raise
            # GPU failed – fall through to CPU
            continue
    state = result.get_statevector(circuit_to_run)
    probs = Statevector(state).probabilities_dict()

    formatted_dict = {
        str(key): round(float(value), significant_digits)
        for key, value in probs.items()
        if round(float(value), significant_digits) != 0
    }
    sorted_dict = dict(sorted(formatted_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_dict = dict(islice(sorted_dict.items(), top_n))

    return sorted_dict