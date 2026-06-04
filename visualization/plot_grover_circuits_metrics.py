"""
Plot F1-score and TVD for each circuit in grover_circuits_{nq}q[_{variant}].parquet.

Computes metrics by comparing the JSON probability distribution extracted from
the SFT completion against the ground-truth msb_measurement_probabilities stored
in extra_info.

Usage
-----
    # all qubit counts (2, 3, 4), optimal variant only
    python -m visualization.plot_grover_circuits_metrics

    # specific qubit count(s)
    python -m visualization.plot_grover_circuits_metrics --nq 2
    python -m visualization.plot_grover_circuits_metrics --nq 2 3 4

    # over/under-rotated variants
    python -m visualization.plot_grover_circuits_metrics --nq 3 4 --variant over
    python -m visualization.plot_grover_circuits_metrics --nq 3 4 --variant under

    # all variants at once
    python -m visualization.plot_grover_circuits_metrics --nq 2 3 4 --variant optimal over under

Outputs (per qubit count × variant)
-------
* A plain-text summary file  (figures/grover_circuits_{tag}/grover_circuits_{tag}_summary.txt)
* A side-by-side bar chart   (figures/grover_circuits_{tag}/grover_circuits_{tag}_metrics.png)
  where tag = {nq}q  or  {nq}q_over  /  {nq}q_under
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.paths import FIG_DIR
from visualization.constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

# ── shared paths ──────────────────────────────────────────────────────────────
_SFT_DIR = Path("/scratch3/ip004/llm_4_qc/data/grover_gpt_replication/sft_datasets")

# Set at runtime via --parquet_dir (overrides _SFT_DIR when provided)
_PARQUET_DIR: Path | None = None


def _tag(nq: int, variant: str) -> str:
    """Return the file-name tag for this (nq, variant) pair."""
    return f"grover_circuits_{nq}q" if variant == "optimal" else f"grover_circuits_{nq}q_{variant}"


def _paths(nq: int, variant: str = "optimal") -> tuple[Path, Path, Path]:
    """Return (parquet_path, summary_path, plot_path) for a given qubit count and variant."""
    tag        = _tag(nq, variant)
    parquet_dir = _PARQUET_DIR if _PARQUET_DIR is not None else _SFT_DIR
    parquet    = parquet_dir / f"{tag}.parquet"
    # Use a sub-folder name that reflects the source so SFT and inference plots don't collide
    source_tag = parquet_dir.name if _PARQUET_DIR is not None else "sft"
    out_dir    = Path(FIG_DIR) / source_tag / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return parquet, out_dir / f"{tag}_summary.txt", out_dir / f"{tag}_metrics.png"

# ── metric helpers ─────────────────────────────────────────────────────────────

_CLIP = 600  # chars from end of completion to search for JSON


def _extract_json_dist(text: str) -> Optional[Dict[str, float]]:
    """Extract the last JSON probability distribution from a completion string."""
    snippet = text[-_CLIP:]
    m = re.search(r"</circuit_reasoning>\s*", snippet)
    if m:
        snippet = snippet[m.end():]
    snippet = re.sub(r"```(?:json)?\s*", "", snippet).strip()
    matches = re.findall(r"\{[^{}]+\}", snippet, re.DOTALL)
    for raw in reversed(matches):
        try:
            d = json.loads(raw)
            if isinstance(d, dict) and all(
                isinstance(v, (int, float)) and v >= 0 for v in d.values()
            ):
                return d
        except json.JSONDecodeError:
            continue
    return None


def _tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Total variation distance (½ ‖p − q‖₁)."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _f1(
    pred: Dict[str, float],
    gt: Dict[str, float],
    tol: float = 0.05,
) -> float:
    """
    Token-level F1 over probability states within *tol* absolute tolerance.

    A state is a true-positive if |pred[s] - gt[s]| <= tol for both dicts.
    """
    tp = sum(1 for k, v in gt.items() if abs(pred.get(k, 0.0) - v) <= tol)
    fp = sum(1 for k, v in pred.items() if abs(gt.get(k, 0.0) - v) > tol)
    fn = sum(1 for k, v in gt.items() if abs(pred.get(k, 0.0) - v) > tol)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _search_acc(pred: Dict[str, float], marked_states: list[str]) -> float:
    """
    Mirrors eval_gates.py search_acc:
    Flip each marked state (LSB→MSB), take top-k predictions (k = len(marked_states)),
    return |intersection| / k.
    """
    if not pred or not marked_states:
        return float("nan")
    k = len(marked_states)
    flipped = {s[::-1] for s in marked_states}
    top_k = set(
        sorted(pred.keys(), key=lambda x: pred[x], reverse=True)[:k]
    )
    return len(top_k & flipped) / k


def _load_marked_states(nq: int, variant: str = "optimal") -> Dict[str, list[str]]:
    """Return {circuit_hash: [marked_states]} from the circuits JSONL."""
    suffix = "" if variant == "optimal" else f"_{variant}"
    jsonl = (
        Path("/scratch3/ip004/llm_4_qc/data/grover_gpt_replication")
        / f"sampled_grover_circuits_{nq}q{suffix}.jsonl"
    )
    mapping: Dict[str, list[str]] = {}
    if not jsonl.exists():
        return mapping
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            h  = obj.get("circuit_hash", "")
            ms = obj.get("extra_info", {}).get("marked_states", [])
            if h:
                mapping[h] = ms
    return mapping


# ── data loading ──────────────────────────────────────────────────────────────

def _load_metrics(parquet: Path, nq: int, variant: str = "optimal") -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    ms_map = _load_marked_states(nq, variant)

    rows = []
    for _, row in df.iterrows():
        ei = row["extra_info"] if isinstance(row["extra_info"], dict) else {}
        idx        = ei.get("index", "?")
        depth      = ei.get("circuit_depth", "?")
        num_gates  = ei.get("num_gates", "?")
        gt_str     = ei.get("msb_measurement_probabilities")
        h          = ei.get("circuit_hash", "")
        marked     = ms_map.get(h, [])

        pred = _extract_json_dist(row["completion"])
        gt   = json.loads(gt_str) if gt_str else None

        rows.append(
            {
                "index":           idx,
                "depth":           depth,
                "num_gates":       num_gates,
                "marked_states":   marked,
                "num_marked":      len(marked),
                "parse_ok":        pred is not None,
                "tvd":           _tvd(pred, gt)          if (pred and gt)     else float("nan"),
                "search_acc":    _search_acc(pred, marked) if (pred and marked) else float("nan"),
                "pred_dist":     pred,
                "gt_dist":       gt,
            }
        )

    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


# ── text summary ──────────────────────────────────────────────────────────────

def _write_summary(data: pd.DataFrame, path: Path, nq: int, variant: str = "optimal") -> None:
    tag = _tag(nq, variant).replace("grover_circuits_", "")
    show_search = (variant == "optimal")
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"Grover Circuits {tag} — Per-Circuit TVD Summary")
    lines.append(f"  Source : grover_circuits_{tag}.parquet")
    lines.append(f"  Variant: {variant}")
    lines.append("  Metric : predicted distribution (from SFT completion)")
    lines.append("           vs. msb_measurement_probabilities (ground truth)")
    if show_search:
        lines.append("  Search : top-k predicted states ∩ marked states / k  (k = #marked)")
    lines.append("=" * 70)
    lines.append("")

    for _, r in data.iterrows():
        lines.append(f"Circuit index : {r['index']}")
        lines.append(f"  Depth       : {r['depth']}")
        lines.append(f"  Num gates   : {r['num_gates']}")
        ms = r["marked_states"]
        lines.append(f"  Marked states: {ms if ms else 'N/A (not found)'}")
        lines.append(f"  Parse ok    : {r['parse_ok']}")
        if r["parse_ok"]:
            lines.append(f"  TVD         : {r['tvd']:.4f}")
            if show_search:
                sa = r["search_acc"]
                lines.append(f"  Search acc  : {sa:.4f}" if not (isinstance(sa, float) and sa != sa) else "  Search acc  : N/A  (no marked states)")
            gt = r["gt_dist"] or {}
            pred = r["pred_dist"] or {}
            lines.append("  Ground truth probabilities:")
            for k, v in sorted(gt.items(), key=lambda kv: -kv[1]):
                lines.append(f"    |{k}⟩  gt={v:.4f}  pred={pred.get(k, 0):.4f}")
        else:
            lines.append("  TVD         : N/A  (parse failed)")
            if show_search:
                lines.append("  Search acc  : N/A  (parse failed)")
        lines.append("")

    # Aggregate
    valid = data.dropna(subset=["tvd"])
    valid_sa = data.dropna(subset=["search_acc"])
    lines.append("-" * 70)
    lines.append(f"Mean TVD       : {valid['tvd'].mean():.4f}")
    if show_search:
        lines.append(f"Mean Search acc: {valid_sa['search_acc'].mean():.4f}" if len(valid_sa) else "Mean Search acc: N/A")
    lines.append(f"Circuits       : {len(data)}  (parsed: {data['parse_ok'].sum()})")
    lines.append("=" * 70)

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written → {path}")


# ── plot ──────────────────────────────────────────────────────────────────────

def _plot_impl(data: pd.DataFrame, path: Path, nq: int, variant: str = "optimal") -> None:
    """
    Bar chart aggregated by number of marked states (k).
    Each bar = mean metric for all circuits with that k; error bars = ±1 std.
    optimal  → 2 panels: TVD, Search Accuracy
    over/under → 1 panel: TVD only
    """
    show_search = (variant == "optimal")
    ks = sorted(data["num_marked"].dropna().unique())
    labels = [f"k={int(k)}\n(N={int((data['num_marked']==k).sum())})" for k in ks]
    x = np.arange(len(ks))

    def _agg(metric: str) -> tuple[list[float], list[float]]:
        means, stds = [], []
        for k in ks:
            vals = data.loc[data["num_marked"] == k, metric].dropna()
            means.append(float(vals.mean()) if len(vals) else float("nan"))
            stds.append(float(vals.std(ddof=0)) if len(vals) > 1 else 0.0)
        return means, stds

    tvd_mean, tvd_std = _agg("tvd")
    sa_mean,  sa_std  = _agg("search_acc") if show_search else ([], [])

    bar_kw  = dict(edgecolor="white", linewidth=0.6, width=0.55)
    err_kw  = dict(fmt="none", capsize=4, capthick=1.2, elinewidth=1.2, color="#444")

    n_panels = 2 if show_search else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    variant_label = {"optimal": "optimal", "over": "over-rotated (+1 iter)", "under": "under-rotated (−1 iter)"}.get(variant, variant)
    fig.suptitle(
        f"Grover Circuits {nq}q [{variant_label}] — Metrics Aggregated by # Marked States\n"
        "(mean ± std  |  SFT completion vs. ground-truth distribution)",
        fontsize=12,
        fontweight="bold",
    )

    # ── TVD ───────────────────────────────────────────────────────────────────
    ax = axes[0]
    valid_tvd = [v for v in tvd_mean if not np.isnan(v)]
    y_max_tvd = max(max(valid_tvd, default=0.01), 0.01) * 1.5
    bars = ax.bar(x, tvd_mean, color=PLOT_COLORS["teal"], **bar_kw)
    ax.errorbar(x, tvd_mean, yerr=tvd_std, **err_kw)
    ax.set_ylim(0, y_max_tvd)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("TVD", fontsize=11)
    ax.set_title("Total Variation Distance", fontsize=11)
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    for bar, v in zip(bars, tvd_mean):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + y_max_tvd * 0.02, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── Search accuracy (optimal only) ────────────────────────────────────────
    if show_search:
        ax = axes[1]
        bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in sa_mean],
                      color=PLOT_COLORS["purple"], **bar_kw)
        ax.errorbar(x, [v if not np.isnan(v) else 0 for v in sa_mean],
                    yerr=sa_std, **err_kw)
        ax.set_ylim(0, 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Search Accuracy", fontsize=11)
        ax.set_title("Search Accuracy (top-k vs. marked states)", fontsize=11)
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
        for bar, v in zip(bars, sa_mean):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved      → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def _run_for_nq(nq: int, variant: str = "optimal") -> None:
    parquet, summary_path, plot_path = _paths(nq, variant)
    if not parquet.exists():
        print(f"WARNING: parquet not found, skipping {nq}q/{variant}: {parquet}")
        return
    tag = _tag(nq, variant)
    print(f"\n── {tag} ─────────────────────────────────────────────────────────")
    data = _load_metrics(parquet, nq, variant)
    _write_summary(data, summary_path, nq, variant)
    _plot_impl(data, plot_path, nq, variant)


def main() -> None:
    global _PARQUET_DIR
    parser = argparse.ArgumentParser(
        description="Plot F1-score and TVD for Grover circuit SFT parquets."
    )
    parser.add_argument(
        "--nq",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5],
        metavar="N",
        help="Qubit counts to process (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        default=["optimal"],
        choices=["optimal", "over", "under"],
        help="Circuit variants to process (default: optimal)",
    )
    parser.add_argument(
        "--parquet_dir",
        type=Path,
        default=None,
        help="Directory containing inference parquets (overrides default SFT dir)",
    )
    args = parser.parse_args()
    if args.parquet_dir is not None:
        _PARQUET_DIR = args.parquet_dir
    for variant in args.variant:
        for nq in args.nq:
            _run_for_nq(nq, variant)


if __name__ == "__main__":
    main()
