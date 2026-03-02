import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import dotenv
import os
import argparse

os.environ["PLOT_DIR"] = os.getenv("PLOT_DIR", "./plots")

def parse_log(log_text):
    """
    Parse the evaluation summary log and extract per-step fidelity by qubit count.
    Returns a dict: { "overall": {step: fidelity}, "1_qubits": {step: fidelity}, ... }
    """
    results = {}

    # ── Overall per-step fidelity ────────────────────────────────────────────
    overall_section = re.search(
        r"PER-STEP FIDELITY:\n(.*?)\n\s*PER-STEP FIDELITY BY NUMBER OF QUBITS",
        log_text, re.DOTALL
    )
    if overall_section:
        results["overall"] = _parse_step_block(overall_section.group(1))

    # ── Per-qubit per-step fidelity ──────────────────────────────────────────
    qubit_section = re.search(
        r"PER-STEP FIDELITY BY NUMBER OF QUBITS:(.*?)QUANTUM STATE PARSE STATISTICS",
        log_text, re.DOTALL
    )
    if qubit_section:
        # Find each "N_qubits:" block
        qubit_blocks = re.findall(
            r"(\d+_qubits):\n(.*?)(?=\n\s+\d+_qubits:|\Z)",
            qubit_section.group(1), re.DOTALL
        )
        for qubit_label, block in qubit_blocks:
            results[qubit_label] = _parse_step_block(block)

    return results


def _parse_step_block(block_text):
    """Parse lines like '    step_1: 1.000000 (n=10000)' into {step_num: fidelity}."""
    pattern = re.compile(r"step_(\d+):\s+([\d.]+)\s+\(n=(\d+)\)")
    step_data = {}
    for match in pattern.finditer(block_text):
        step_num = int(match.group(1))
        fidelity = float(match.group(2))
        n        = int(match.group(3))
        step_data[step_num] = {"fidelity": fidelity, "n": n}
    return step_data


def plot_stepwise_fidelity(results, title="Step-wise Quantum State Fidelity", save_path=None):

    style_map = {
        "1_qubits": {"label": "1 qubit",            "color": "#2ecc71", "lw": 1.8},
        "2_qubits": {"label": "2 qubits",            "color": "#3498db", "lw": 1.8},
        "3_qubits": {"label": "3 qubits",            "color": "#f39c12", "lw": 1.8},
        "4_qubits": {"label": "4 qubits",            "color": "#e74c3c", "lw": 1.8},
        "5_qubits": {"label": "5 qubits",            "color": "#9b59b6", "lw": 1.8},
        "overall":  {"label": "Overall (Aggregated)","color": "#1a1a2e", "lw": 2.8},
    }

    # Preferred draw order (overall on top)
    draw_order = ["1_qubits", "2_qubits", "3_qubits", "4_qubits", "5_qubits", "overall"]

     # styled combined plot
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",   # Computer Modern for math
    })

    fig, ax = plt.subplots(figsize=(7, 6))

    for key in draw_order:
        if key not in results:
            continue
        step_data = results[key]
        steps     = sorted(step_data.keys())
        fidelities = [step_data[s]["fidelity"] for s in steps]
        sty = style_map[key]
        ax.plot(steps, fidelities, label=sty["label"],
                color=sty["color"], linewidth=sty["lw"])

    for threshold, label in [(0.99, "99%"), (0.95, "95%"), (0.90, "90%")]:
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 51,
                threshold, f" {label}", va="center",
                fontsize=9, color="gray")

    ax.set_xlabel("Number of Gates", fontsize=13)
    ax.set_ylabel("Quantum State Fidelity", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which="major", linestyle="-", alpha=0.15)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot step-wise quantum state fidelity from evaluation logs.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the evaluation summary log file.")
    parser.add_argument("--output_file", type=str, help="Path to save the generated plot.")
    args = parser.parse_args()


    with open(args.log_path, "r") as f:
        log_text = f.read()

    save_path = os.path.join(os.getenv("PLOT_DIR"), args.output_file or "stepwise_fidelity_plot.png")

    results = parse_log(log_text)
    plot_stepwise_fidelity(
        results,
        save_path=save_path
    )