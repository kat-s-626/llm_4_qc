"""Shared plotting style constants for visualization scripts."""

import matplotlib.pyplot as plt

PLOT_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Libertinus Serif", "Linux Libertine", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": False,
}

PLOT_COLORS = {
    "purple": "#6d28d9",
    "orange": "#f97316",
    "grey": "#64748b",
    "teal": "#14b8a6",
    "baseline": "#3b0764",
    "accent": "#8b5cf6",
    "green": "#2ecc71",
    "blue": "#3498db",
    "red": "#e74c3c",
    "navy": "#1a1a2e",
    "axis": "#333333",
}

STEP_FIDELITY_STYLES = {
    "1_qubits": {"label": "1 qubit", "color": PLOT_COLORS["green"], "linewidth": 1.8},
    "2_qubits": {"label": "2 qubits", "color": PLOT_COLORS["blue"], "linewidth": 1.8},
    "3_qubits": {"label": "3 qubits", "color": PLOT_COLORS["orange"], "linewidth": 1.8},
    "4_qubits": {"label": "4 qubits", "color": PLOT_COLORS["red"], "linewidth": 1.8},
    "5_qubits": {"label": "5 qubits", "color": PLOT_COLORS["purple"], "linewidth": 1.8},
    "overall": {"label": "Overall (Aggregated)", "color": PLOT_COLORS["navy"], "linewidth": 2.8},
}

LOSS_CURVE_STYLES = {
    "train": {
        "label": "Train",
        "color": PLOT_COLORS["purple"],
        "linewidth": 1.8,
        "linestyle": "-",
        "alpha": 0.9,
    },
    "validation": {
        "label": "Validation",
        "color": PLOT_COLORS["baseline"],
        "linewidth": 1.8,
        "linestyle": "--",
        "alpha": 0.95,
    },
}

REFERENCE_LINE_STYLE = {
    "color": PLOT_COLORS["grey"],
    "linestyle": "--",
    "linewidth": 0.9,
    "alpha": 0.7,
}

LOG_METRIC_FIELDS = (
    "step",
    "global_seqlen/min",
    "global_seqlen/max",
    "global_seqlen/minmax_diff",
    "global_seqlen/balanced_min",
    "global_seqlen/balanced_max",
    "global_seqlen/mean",
    "actor/entropy",
    "actor/kl_loss",
    "actor/kl_coef",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/pg_clipfrac_lower",
    "actor/grad_norm",
    "perf/mfu/actor",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
    "actor/lr",
    "training/global_step",
    "training/epoch",
    "critic/score/mean",
    "critic/score/max",
    "critic/score/min",
    "critic/rewards/mean",
    "critic/rewards/max",
    "critic/rewards/min",
    "critic/advantages/mean",
    "critic/advantages/max",
    "critic/advantages/min",
    "critic/returns/mean",
    "critic/returns/max",
    "critic/returns/min",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
    "response_length/clip_ratio",
    "response_length_non_aborted/mean",
    "response_length_non_aborted/max",
    "response_length_non_aborted/min",
    "response_length_non_aborted/clip_ratio",
    "response/aborted_ratio",
    "prompt_length/mean",
    "prompt_length/max",
    "prompt_length/min",
    "prompt_length/clip_ratio",
    "timing_s/start_profile",
    "timing_s/generate_sequences",
    "timing_s/reshard",
    "timing_s/generation_timing/max",
    "timing_s/generation_timing/min",
    "timing_s/generation_timing/topk_ratio",
    "timing_s/gen",
    "timing_s/reward",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/adv",
    "timing_s/update_actor",
    "timing_s/step",
    "timing_s/save_checkpoint",
    "timing_s/stop_profile",
    "timing_per_token_ms/ref",
    "timing_per_token_ms/gen",
    "timing_per_token_ms/adv",
    "timing_per_token_ms/update_actor",
    "perf/total_num_tokens",
    "perf/time_per_step",
    "perf/throughput",
)



def apply_plot_style() -> None:
    """Apply shared Matplotlib rcParams style for plots."""
    plt.rcParams.update(PLOT_RCPARAMS)
