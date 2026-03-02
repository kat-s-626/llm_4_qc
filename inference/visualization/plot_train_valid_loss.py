# ...existing code...
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import numpy as np
import dotenv
import argparse

dotenv.load_dotenv()

def read_lines(path):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    # final fallback: replace invalid bytes
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()



def parse_log_file(log_file, train_map, val_map):
    """Read a single log file and update `train_map` and `val_map` dicts keyed by step.
    Later entries (from later files or later lines) overwrite earlier values for the same step.
    """
    lines = read_lines(log_file)

    current_step = None
    for line in lines:
        # try to extract an explicit step number from the line
        m_step = re.search(r'[Ss]tep[:=\s]+(\d+)', line)
        if m_step:
            current_step = int(m_step.group(1))

        if 'train/loss' in line:
            # Extract the training loss value
            try:
                loss = float(line.split('train/loss:')[1].split(' - ')[0])
            except Exception:
                m = re.search(r'train/loss:\s*([0-9.eE+-]+)', line)
                loss = float(m.group(1)) if m else None

            step = current_step
            # fallback: try to find a number near the loss if no explicit step found
            if step is None:
                m2 = re.search(r'(\d+)\s+.*train/loss', line)
                step = int(m2.group(1)) if m2 else (max(train_map.keys()) + 1 if train_map else 1)
            
            train_map[int(step)] = loss

        elif 'val/loss' in line:
            # Extract the validation loss value
            try:
                loss = float(line.split('val/loss:')[1].split(' - ')[0])
            except Exception:
                m = re.search(r'val/loss:\s*([0-9.eE+-]+)', line)
                loss = float(m.group(1)) if m else None

            step = current_step
            if step is None:
                m2 = re.search(r'(\d+)\s+.*val/loss', line)
                step = int(m2.group(1)) if m2 else (max(val_map.keys()) + 1 if val_map else 1)

            val_map[int(step)] = loss

    # function intentionally does not return lists; maps are updated in-place
    return

def plot_loss(log_path, output_file):
    log_files = os.listdir(log_path)
    # accumulate across all files into these maps (step -> loss)
    train_map = {}
    val_map = {}

    for log_file in log_files:
        log_file_path = os.path.join(log_path, log_file)
        # update train_map and val_map in-place; later files overwrite earlier steps
        try:
            parse_log_file(log_file_path, train_map, val_map)
        except Exception:
            # skip unreadable files but continue
            continue

    # convert accumulated maps into sorted lists for plotting
    train_steps = sorted(train_map.keys())
    training_loss = [train_map[s] for s in train_steps]

    val_steps = sorted(val_map.keys())
    validation_loss = [val_map[s] for s in val_steps]

    # styled combined plot
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",   # Computer Modern for math
    })
        
    fig, ax = plt.subplots(figsize=(7, 6))

    PURPLE_LIGHT = "#4D2CB9"
    PURPLE_DARK = "#1a0a3d"

    # Determine primary x steps to use for axis scaling
    steps = train_steps if train_steps else sorted(set(val_steps))

    if train_steps:
        ax.plot(train_steps, training_loss, color=PURPLE_LIGHT, linewidth=1.8,
                linestyle='-', label='Train', alpha=0.9)
    else:
        ax.plot(steps, training_loss, color=PURPLE_LIGHT, linewidth=1.8,
                linestyle='-', label='Train', alpha=0.9)

    if val_steps:
        ax.plot(val_steps, validation_loss, color=PURPLE_DARK, linewidth=1.8,
                linestyle='--', label='Validation', alpha=0.95)

    # Axes labels & ticks
    ax.set_xlabel("Training Step", fontsize=13, labelpad=8)
    ax.set_ylabel("Loss", fontsize=13, labelpad=8)
    ax.tick_params(axis="both", labelsize=11)

    # Fix y-axis range to [0, 1]
    ax.set_ylim(0.0, 1.0)

    # Remove top & right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")

    # Legend
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    plt.tight_layout()

    plot_dir = os.getenv("PLOT_DIR", log_path)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    out_file = os.path.join(plot_dir, output_file)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training and validation loss from log files')
    parser.add_argument('--log_path', type=str, required=True, help='Directory containing log files')
    parser.add_argument('--output_file', type=str, default='train_valid_loss_plot.png', help='Filename for the output plot image')
    args = parser.parse_args() 


    plot_loss(args.log_path, args.output_file)
