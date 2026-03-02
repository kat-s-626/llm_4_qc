import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
from plot_style_constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

def clean_spines(ax):
    """Remove top and right spines"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def smooth_curve(data, window_size=10):
    """Apply moving average smoothing"""
    if len(data) < window_size:
        return data
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().values

# Load Grover SFT-GRPO data
print("Loading Grover SFT-GRPO training data...")
grover_csv_path = 'grover_sft_grpo_metrics_aggregated.csv'

if not os.path.exists(grover_csv_path):
    raise FileNotFoundError(f"Could not find {grover_csv_path}")

grover_df = pd.read_csv(grover_csv_path)
print(f"Loaded {len(grover_df)} Grover training steps")

# Load GRPO data
print("Loading GRPO training data...")
grpo_csv_path = 'grpo_all_metrics_aggregated.csv'

if not os.path.exists(grpo_csv_path):
    raise FileNotFoundError(f"Could not find {grpo_csv_path}")

grpo_df = pd.read_csv(grpo_csv_path)
print(f"Loaded {len(grpo_df)} GRPO training steps")

# Extract Grover data
grover_steps = grover_df['step'].values
grover_rewards = grover_df['critic/rewards/mean'].values
grover_response_length = grover_df['response_length/mean'].values
grover_entropy = grover_df['actor/entropy'].values

# Extract GRPO rewards (we'll use these to replace steps 250-500)
grpo_rewards = grpo_df['critic/rewards/mean'].values

# Handle NaN values in Grover data
grover_mask = ~(np.isnan(grover_rewards) | np.isnan(grover_response_length) | np.isnan(grover_entropy))
grover_steps = grover_steps[grover_mask]
grover_rewards = grover_rewards[grover_mask]
grover_response_length = grover_response_length[grover_mask]
grover_entropy = grover_entropy[grover_mask]

print(f"\nOriginal Grover data:")
print(f"  Valid steps: {len(grover_steps)}")
print(f"  Step range: {grover_steps[0]} to {grover_steps[-1]}")
print(f"  Reward range: [{np.min(grover_rewards):.4f}, {np.max(grover_rewards):.4f}]")

# Create modified rewards by replacing steps 250-500 with GRPO data
modified_rewards = grover_rewards.copy()

# Find indices for steps 250-500 in Grover data
replace_start = 250
replace_end = 500

grover_replace_mask = (grover_steps >= replace_start) & (grover_steps <= replace_end)
grover_replace_indices = np.where(grover_replace_mask)[0]
num_steps_to_replace = len(grover_replace_indices)

print(f"\nReplacing rewards from step {replace_start} to {replace_end}:")
print(f"  Number of steps to replace: {num_steps_to_replace}")

# Get corresponding GRPO rewards (take the first num_steps_to_replace from GRPO data)
# Remove NaN values from GRPO rewards first
grpo_rewards_clean = grpo_rewards[~np.isnan(grpo_rewards)]

if len(grpo_rewards_clean) >= num_steps_to_replace:
    replacement_rewards = grpo_rewards_clean[:num_steps_to_replace]
    modified_rewards[grover_replace_indices] = replacement_rewards
    print(f"  Replacement reward range: [{np.min(replacement_rewards):.4f}, {np.max(replacement_rewards):.4f}]")
else:
    print(f"  Warning: Not enough GRPO data ({len(grpo_rewards_clean)} steps). Using available data.")
    replacement_rewards = grpo_rewards_clean
    modified_rewards[grover_replace_indices[:len(replacement_rewards)]] = replacement_rewards

print(f"\nModified Grover data:")
print(f"  Reward range: [{np.min(modified_rewards):.4f}, {np.max(modified_rewards):.4f}]")

# Create output directory
os.makedirs('figures/grover_sft_grpo_modified', exist_ok=True)

# Define training stages
training_stages = [
    (0, 250, 'Original Data'),
    (250, 500, 'Replaced with GRPO Data'),
    (500, grover_steps[-1], 'Original Data'),
]

# Stage colors
stage_colors = [PLOT_COLORS['purple'], PLOT_COLORS['orange'], PLOT_COLORS['purple']]

# ============================================================
# Plot 1: Original vs Modified Rewards Comparison
# ============================================================
print("\nCreating comparison plot (Original vs Modified)...")

fig, ax = plt.subplots(figsize=(10, 5))

# Add stage background shading
for idx, (start, end, stage_label) in enumerate(training_stages):
    alpha = 0.08
    if start < grover_steps[-1]:
        end_plot = min(end, grover_steps[-1] + 1)
        if idx == 1:  # Highlight the replaced section
            alpha = 0.12
        ax.axvspan(start, end_plot, alpha=alpha, color=stage_colors[idx], 
                  label=stage_label if idx < 2 or idx == 2 and start < grover_steps[-1] else '', zorder=0)

# Smooth the data
smoothed_original = smooth_curve(grover_rewards, window_size=20)
smoothed_modified = smooth_curve(modified_rewards, window_size=20)

# Plot original data (lighter)
ax.plot(grover_steps, smoothed_original, '-', color=PLOT_COLORS['grey'], 
        linewidth=2.0, alpha=0.5, label='Original Rewards', zorder=2)

# Plot modified data (prominent)
ax.plot(grover_steps, modified_rewards, '-', color=PLOT_COLORS['purple'], 
        linewidth=0.5, alpha=0.3, zorder=1)
ax.plot(grover_steps, smoothed_modified, '-', color=PLOT_COLORS['purple'], 
        linewidth=2.5, label='Modified Rewards', zorder=3)

# Styling
ax.set_xlabel('Training Step')
ax.set_ylabel('Mean Reward')
ax.set_xlim([0, grover_steps[-1] + 10])
ax.set_ylim([max(0, min(np.min(grover_rewards), np.min(modified_rewards)) - 0.05), 
             max(np.max(grover_rewards), np.max(modified_rewards)) + 0.05])
ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
clean_spines(ax)

plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.savefig('figures/grover_sft_grpo_modified/grover_modified_comparison.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('figures/grover_sft_grpo_modified/grover_modified_comparison.png', dpi=1000, bbox_inches='tight')
plt.close()

print(f"Saved: figures/grover_sft_grpo_modified/grover_modified_comparison.pdf")

# ============================================================
# Plot 2: Dual Y-Axis (Modified Reward + Response Length)
# ============================================================
print("Creating dual y-axis plot with modified rewards...")

fig, ax1 = plt.subplots(figsize=(10, 5))

# Add stage background shading
for idx, (start, end, stage_label) in enumerate(training_stages):
    alpha = 0.08
    if start < grover_steps[-1]:
        end_plot = min(end, grover_steps[-1] + 1)
        if idx == 1:  # Highlight the replaced section
            alpha = 0.12
        ax1.axvspan(start, end_plot, alpha=alpha, color=stage_colors[idx], zorder=0)

# Smooth both curves
smoothed_modified = smooth_curve(modified_rewards, window_size=20)
smoothed_length = smooth_curve(grover_response_length, window_size=20)

# Plot rewards on left y-axis
color_rewards = PLOT_COLORS['purple']
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Mean Reward', color=color_rewards)
line1 = ax1.plot(grover_steps, smoothed_modified, '-', color=color_rewards, 
        linewidth=2.5, label='Mean Reward (Modified)', zorder=3)
ax1.tick_params(axis='y', labelcolor=color_rewards)
ax1.set_xlim([0, grover_steps[-1] + 10])
ax1.set_ylim([0, max(0.6, np.max(modified_rewards) + 0.05)])

# Create second y-axis for response length
ax2 = ax1.twinx()
color_length = PLOT_COLORS['orange']
ax2.set_ylabel('Mean Response Length', color=color_length)
line2 = ax2.plot(grover_steps, smoothed_length, '-', color=color_length, 
        linewidth=2.5, label='Mean Response Length', zorder=3)
ax2.tick_params(axis='y', labelcolor=color_length)
ax2.set_ylim([max(0, np.min(grover_response_length) - 200), np.max(grover_response_length) + 200])

# Add combined legend with all elements
lines = line1 + line2
labels = [l.get_label() for l in lines]
# Add spacing and stage labels
labels.append('')  # Add empty label for spacing
stage_handles = [plt.Rectangle((0,0),1,1, fc='white', alpha=0)]  # spacing
stage_labels = ['']
for idx, (start, end, label) in enumerate(training_stages):
    if start < grover_steps[-1]:
        stage_handles.append(plt.Rectangle((0,0),1,1, fc=stage_colors[idx], alpha=0.3))
        stage_labels.append(label)

ax1.legend(lines + stage_handles, labels + stage_labels, frameon=False, 
          loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=9)

# Styling
clean_spines(ax1)
clean_spines(ax2)
ax1.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.subplots_adjust(right=0.80)
plt.savefig('figures/grover_sft_grpo_modified/grover_modified_dual_axis.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('figures/grover_sft_grpo_modified/grover_modified_dual_axis.png', dpi=1000, bbox_inches='tight')
plt.close()

print(f"Saved: figures/grover_sft_grpo_modified/grover_modified_dual_axis.pdf")

# ============================================================
# Plot 3: Difference Plot (showing what changed)
# ============================================================
print("Creating difference plot...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top plot: Both curves
for idx, (start, end, stage_label) in enumerate(training_stages):
    alpha = 0.08
    if start < grover_steps[-1]:
        end_plot = min(end, grover_steps[-1] + 1)
        if idx == 1:
            alpha = 0.12
        ax1.axvspan(start, end_plot, alpha=alpha, color=stage_colors[idx], 
                   label=stage_label if idx < 2 else '', zorder=0)

ax1.plot(grover_steps, smoothed_original, '-', color=PLOT_COLORS['grey'], 
        linewidth=2.0, label='Original', zorder=2)
ax1.plot(grover_steps, smoothed_modified, '-', color=PLOT_COLORS['purple'], 
        linewidth=2.0, label='Modified', zorder=3)
ax1.set_ylabel('Mean Reward')
ax1.set_xlim([0, grover_steps[-1] + 10])
ax1.set_title('Reward Curves: Original vs Modified')
ax1.legend(frameon=False, loc='best', fontsize=9)
clean_spines(ax1)
ax1.grid(True, alpha=0.3, linestyle='--')

# Bottom plot: Difference (Modified - Original)
difference = modified_rewards - grover_rewards

for idx, (start, end, stage_label) in enumerate(training_stages):
    alpha = 0.08
    if start < grover_steps[-1]:
        end_plot = min(end, grover_steps[-1] + 1)
        if idx == 1:
            alpha = 0.12
        ax2.axvspan(start, end_plot, alpha=alpha, color=stage_colors[idx], zorder=0)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax2.plot(grover_steps, difference, '-', color=PLOT_COLORS['orange'], 
        linewidth=0.8, alpha=0.6, zorder=2)
smoothed_diff = smooth_curve(difference, window_size=20)
ax2.plot(grover_steps, smoothed_diff, '-', color=PLOT_COLORS['orange'], 
        linewidth=2.0, label='Smoothed Difference', zorder=3)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Difference')
ax2.set_xlim([0, grover_steps[-1] + 10])
ax2.set_title('Reward Difference (Modified - Original)')
ax2.legend(frameon=False, loc='best', fontsize=9)
clean_spines(ax2)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/grover_sft_grpo_modified/grover_difference_plot.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('figures/grover_sft_grpo_modified/grover_difference_plot.png', dpi=1000, bbox_inches='tight')
plt.close()

print(f"Saved: figures/grover_sft_grpo_modified/grover_difference_plot.pdf")

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "="*60)
print("Grover SFT-GRPO Modified Training Summary")
print("="*60)

print(f"\nOriginal Rewards (steps 250-500):")
replace_original = grover_rewards[grover_replace_indices]
print(f"  Mean: {np.mean(replace_original):.4f}")
print(f"  Std: {np.std(replace_original):.4f}")
print(f"  Min: {np.min(replace_original):.4f}")
print(f"  Max: {np.max(replace_original):.4f}")

print(f"\nReplacement Rewards (from GRPO, steps 250-500):")
print(f"  Mean: {np.mean(replacement_rewards):.4f}")
print(f"  Std: {np.std(replacement_rewards):.4f}")
print(f"  Min: {np.min(replacement_rewards):.4f}")
print(f"  Max: {np.max(replacement_rewards):.4f}")

print(f"\nOverall Modified Data:")
print(f"  Mean reward: {np.mean(modified_rewards):.4f}")
print(f"  Std reward: {np.std(modified_rewards):.4f}")
print(f"  Min reward: {np.min(modified_rewards):.4f}")
print(f"  Max reward: {np.max(modified_rewards):.4f}")

print(f"\nChange in Replaced Region:")
replace_diff = replacement_rewards - replace_original[:len(replacement_rewards)]
print(f"  Mean difference: {np.mean(replace_diff):.4f}")
print(f"  Std difference: {np.std(replace_diff):.4f}")
print(f"  Min difference: {np.min(replace_diff):.4f}")
print(f"  Max difference: {np.max(replace_diff):.4f}")

print("\n" + "="*60)
print("All plots saved to figures/grover_sft_grpo_modified/")
print("="*60)

# Save the modified data to CSV
output_df = pd.DataFrame({
    'step': grover_steps,
    'original_reward': grover_rewards,
    'modified_reward': modified_rewards,
    'response_length': grover_response_length,
    'actor_entropy': grover_entropy,
    'difference': difference
})

output_csv = 'grover_sft_grpo_modified_rewards.csv'
output_df.to_csv(output_csv, index=False)
print(f"\nSaved modified data to: {output_csv}")
