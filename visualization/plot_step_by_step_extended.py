"""Plot step-wise fidelity from 1–70 gates by stitching an in-distribution set
(test_1, steps 1–50) with an extrapolation set (test_2, steps 51–70).

A vertical separator is drawn at the boundary step to visually distinguish the
in-distribution and extrapolation regions.

Usage
-----
python -m visualization.plot_step_by_step_extended \
    --test1-summary /path/to/test_1_summary.txt \
    --test2-summary /path/to/test_2_summary.txt \
    [--boundary 50] \
    [--output-dir visualization/figures] \
    [--output-file extended_stepwise_fidelity.png]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config.paths import FIG_DIR
from visualization.constants import apply_plot_style, REFERENCE_LINE_STYLE, STEP_FIDELITY_STYLES

apply_plot_style()

# ── regex helpers (mirrors sft_output_parser) ─────────────────────────────────
_OVERALL_RE = re.compile(
    r"PER-STEP FIDELITY:\s*\n(.*?)\n\s*PER-STEP FIDELITY BY NUMBER OF QUBITS",
    re.DOTALL,
)
_QUBIT_SECTION_RE = re.compile(
    r"PER-STEP FIDELITY BY NUMBER OF QUBITS:(.*?)(?:\n\s*QUANTUM STATE PARSE STATISTICS|\Z)",
    re.DOTALL,
)
_QUBIT_BLOCK_RE = re.compile(r"(\d+_qubits):\s*\n(.*?)(?=\n\s+\d+_qubits:|\Z)", re.DOTALL)
_STEP_RE = re.compile(r"step_(\d+):\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+\(n=(\d+)\)")

DRAW_ORDER = ("1_qubits", "2_qubits", "3_qubits", "4_qubits", "5_qubits", "overall")


# ── parsing ───────────────────────────────────────────────────────────────────

def _parse_block(text: str) -> dict[int, tuple[float, int]]:
    """Return {step: (fidelity, n)} from a step-fidelity text block."""
    result: dict[int, tuple[float, int]] = {}
    for m in _STEP_RE.finditer(text):
        result[int(m.group(1))] = (float(m.group(2)), int(m.group(3)))
    return result


def parse_summary(path: Path) -> dict[str, dict[int, tuple[float, int]]]:
    """Parse a *_summary.txt file into {group: {step: (fidelity, n)}}."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    groups: dict[str, dict[int, tuple[float, int]]] = {}

    m = _OVERALL_RE.search(text)
    if m:
        groups["overall"] = _parse_block(m.group(1))

    qs = _QUBIT_SECTION_RE.search(text)
    if qs:
        for label, block in _QUBIT_BLOCK_RE.findall(qs.group(1)):
            groups[label] = _parse_block(block)

    return groups


# ── stitching ─────────────────────────────────────────────────────────────────

def stitch(
    data1: dict[str, dict[int, tuple[float, int]]],
    data2: dict[str, dict[int, tuple[float, int]]],
    boundary: int,
) -> dict[str, dict[int, tuple[float, int]]]:
    """Merge two parsed summaries.

    For each group, steps <= boundary come from *data1*; steps > boundary come
    from *data2*.  Groups present in only one source are included as-is.
    """
    all_groups = set(data1) | set(data2)
    merged: dict[str, dict[int, tuple[float, int]]] = {}
    for group in all_groups:
        d: dict[int, tuple[float, int]] = {}
        for step, val in (data1.get(group) or {}).items():
            if step <= boundary:
                d[step] = val
        for step, val in (data2.get(group) or {}).items():
            if step > boundary:
                d[step] = val
        if d:
            merged[group] = d
    return merged


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_extended(
    stitched: dict[str, dict[int, tuple[float, int]]],
    boundary: int,
    save_path: Path,
) -> bool:
    draw_order = [*DRAW_ORDER, *sorted(k for k in stitched if k not in DRAW_ORDER)]

    fig, ax = plt.subplots(figsize=(9, 6))
    has_data = False

    for key in draw_order:
        if key not in stitched:
            continue
        step_data = stitched[key]
        if not step_data:
            continue
        steps = sorted(step_data)
        fidelities = [step_data[s][0] for s in steps]
        sty = STEP_FIDELITY_STYLES.get(key)
        if sty:
            ax.plot(steps, fidelities, label=sty["label"],
                    color=sty["color"], linewidth=sty["linewidth"])
        else:
            ax.plot(steps, fidelities, label=key, linewidth=1.8)
        has_data = True

    if not has_data:
        plt.close(fig)
        return False

    # ── extrapolation boundary ────────────────────────────────────────────────
    ax.axvline(
        boundary,
        color="#94a3b8",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        zorder=2,
    )
    y_top = 1.02
    ax.text(
        boundary - 0.8, y_top, "In-distribution",
        ha="right", va="top", fontsize=9, color="#475569",
        style="italic",
    )
    ax.text(
        boundary + 0.8, y_top, "Extrapolation",
        ha="left", va="top", fontsize=9, color="#475569",
        style="italic",
    )

    # ── region shading ────────────────────────────────────────────────────────
    x_max = max(max(d.keys()) for d in stitched.values()) + 1
    ax.axvspan(boundary, x_max, alpha=0.04, color="#f97316", zorder=0)

    # ── reference lines ───────────────────────────────────────────────────────
    for threshold, label in [(0.99, "99%"), (0.95, "95%"), (0.90, "90%")]:
        ax.axhline(threshold, **REFERENCE_LINE_STYLE)
        ax.text(x_max, threshold, f" {label}", va="center",
                fontsize=9, color=REFERENCE_LINE_STYLE["color"])

    ax.set_xlabel("Number of Gates", fontsize=13)
    ax.set_ylabel("Quantum State Fidelity", fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(left=0, right=x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which="major", linestyle="-", alpha=0.15)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot step-wise fidelity (1–N gates) stitching an in-distribution "
                    "and an extrapolation summary file, with a visual boundary separator."
    )
    parser.add_argument("--test1-summary", type=Path, required=True,
                        help="Path to the in-distribution summary .txt (e.g. test_1_summary.txt).")
    parser.add_argument("--test2-summary", type=Path, required=True,
                        help="Path to the extrapolation summary .txt (e.g. test_2_summary.txt).")
    parser.add_argument("--boundary", type=int, default=50,
                        help="Gate step at which in-distribution ends and extrapolation begins (default: 50).")
    parser.add_argument("--output-dir", type=Path, default=Path(FIG_DIR),
                        help="Directory for saved plot.")
    parser.add_argument("--output-file", type=str,
                        default="extended_stepwise_fidelity.png",
                        help="Output filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for p in (args.test1_summary, args.test2_summary):
        if not p.exists():
            raise FileNotFoundError(f"Summary file not found: {p}")

    data1 = parse_summary(args.test1_summary.resolve())
    data2 = parse_summary(args.test2_summary.resolve())
    stitched = stitch(data1, data2, args.boundary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / args.output_file

    saved = plot_extended(stitched, args.boundary, save_path)

    if saved:
        print(f"Extended step-wise fidelity plot saved to: {save_path}")
    else:
        print("Plot skipped: no plottable fidelity data.")
