import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from plot_style_constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

def get_text_color(hex_color):
    """Determine if text should be black or white based on background luminosity"""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminosity = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return 'black' if luminosity > 0.5 else 'white'

def clean_spines(ax):
    """Remove top and right spines"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def smooth_curve(data, window_size=10):
    """Apply moving average smoothing"""
    if len(data) < window_size:
        return data
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().values


# Load data from aggregated CSV
print("Loading GRPO training data from grpo_all_metrics_aggregated.csv...")
csv_path = 'grpo_all_metrics_aggregated.csv'

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}. Please run parse_grpo_metrics.py first.")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} training steps")

# Extract relevant columns
steps = df['step'].values
critic_rewards = df['critic/rewards/mean'].values
response_length = df['response_length/mean'].values

# Handle NaN values
mask = ~(np.isnan(critic_rewards) | np.isnan(response_length))
steps = steps[mask]
critic_rewards = critic_rewards[mask]
response_length = response_length[mask]

# Filter data to only include steps up to 1152
step_limit = 1152
step_mask = steps <= step_limit
steps = steps[step_mask]
critic_rewards = critic_rewards[step_mask]
response_length = response_length[step_mask]

print(f"Valid data points: {len(steps)} steps (limited to step {step_limit})")
print(f"Reward range: [{np.min(critic_rewards):.4f}, {np.max(critic_rewards):.4f}]")
print(f"Response length range: [{np.min(response_length):.1f}, {np.max(response_length):.1f}]")

# Create output directory
os.makedirs('figures/grpo', exist_ok=True)

# ============================================================
# Plot 1: Critic Rewards Mean Over Training
# ============================================================
print("\nCreating critic rewards plot...")

fig, ax = plt.subplots(figsize=(10, 5))

# Define training stages (matching GRPO curriculum)
training_stages = [
    (0, 780, 'Stage 1: Foundation'),
    (780, 966, 'Stage 2: Intermediate'),
    (966, 1152, 'Stage 3: Full Complexity')
]

# Add stage background shading
stage_colors = [PLOT_COLORS['purple'], PLOT_COLORS['orange'], PLOT_COLORS['teal']]
for idx, (start, end, stage_label) in enumerate(training_stages):
    alpha = 0.08 if idx == 0 else 0.06
    ax.axvspan(start, end, alpha=alpha, color=stage_colors[idx % len(stage_colors)], 
              label=stage_label, zorder=0)

# Smooth the data for better visualization
smoothed_rewards = smooth_curve(critic_rewards, window_size=20)

# Plot raw data with transparency
ax.plot(steps, critic_rewards, '-', color=PLOT_COLORS['purple'], 
        linewidth=0.5, alpha=0.2, zorder=1)

# Plot smoothed data
ax.plot(steps, smoothed_rewards, '-', color=PLOT_COLORS['purple'], 
        linewidth=2.5, zorder=3)

# Styling
ax.set_xlabel('Training Step')
ax.set_ylabel('Mean Reward')
ax.set_xlim([0, steps[-1] + 10])
ax.set_ylim([max(0, np.min(critic_rewards) - 0.05), np.max(critic_rewards) + 0.05])
ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
clean_spines(ax)

plt.tight_layout()
plt.subplots_adjust(right=0.82)
plt.savefig('figures/grpo/grpo_critic_rewards.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('figures/grpo/grpo_critic_rewards.png', dpi=1000, bbox_inches='tight')
plt.close()

print(f"Saved: figures/grpo/grpo_critic_rewards.pdf")

# ============================================================
# Plot 2: Response Length Over Training
# ============================================================
print("Creating response length plot...")

fig, ax = plt.subplots(figsize=(10, 5))

# Define training stages (matching GRPO curriculum)
training_stages = [
    (0, 780, 'Stage 1: Foundation'),
    (780, 966, 'Stage 2: Intermediate'),
    (966, 1152, 'Stage 3: Full Complexity')
]

# Add stage background shading
stage_colors = [PLOT_COLORS['purple'], PLOT_COLORS['orange'], PLOT_COLORS['teal']]
for idx, (start, end, stage_label) in enumerate(training_stages):
    alpha = 0.08 if idx == 0 else 0.06
    ax.axvspan(start, end, alpha=alpha, color=stage_colors[idx % len(stage_colors)], 
              label=stage_label, zorder=0)

# Smooth the data
smoothed_length = smooth_curve(response_length, window_size=20)

# Plot raw data with transparency
ax.plot(steps, response_length, '-', color=PLOT_COLORS['orange'], 
        linewidth=0.5, alpha=0.2, zorder=1)

# Plot smoothed data
ax.plot(steps, smoothed_length, '-', color=PLOT_COLORS['orange'], 
        linewidth=2.5, zorder=3)

# Styling
ax.set_xlabel('Training Step')
ax.set_ylabel('Mean Response Length')
ax.set_xlim([0, steps[-1] + 10])
ax.set_ylim([max(0, np.min(response_length) - 200), np.max(response_length) + 200])
ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
clean_spines(ax)

plt.tight_layout()
plt.subplots_adjust(right=0.82)
plt.savefig('figures/grpo/grpo_response_length.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('figures/grpo/grpo_response_length.png', dpi=1000, bbox_inches='tight')
plt.close()

print(f"Saved: figures/grpo/grpo_response_length.pdf")

# ============================================================
# Plot 3: Dual Y-Axis (Reward + Response Length)
# ============================================================
print("Creating dual y-axis plot (Reward + Response Length)...")

fig, ax1 = plt.subplots(figsize=(10, 5))

# Define training stages (matching GRPO curriculum)
training_stages = [
    (0, 780, 'Stage 1: Foundation'),
    (780, 966, 'Stage 2: Intermediate'),
    (966, 1152, 'Stage 3: Full Complexity')
]

# Add stage background shading (without labels in axvspan)
stage_colors = [PLOT_COLORS['purple'], PLOT_COLORS['orange'], PLOT_COLORS['teal']]
for idx, (start, end, stage_label) in enumerate(training_stages):
    alpha = 0.08 if idx == 0 else 0.06
    ax1.axvspan(start, end, alpha=alpha, color=stage_colors[idx % len(stage_colors)], zorder=0)

# Smooth both curves
smoothed_rewards = smooth_curve(critic_rewards, window_size=20)
smoothed_length = smooth_curve(response_length, window_size=20)

# Plot rewards on left y-axis
color_rewards = PLOT_COLORS['purple']
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Mean Reward', color=color_rewards)
line1 = ax1.plot(steps, smoothed_rewards, '-', color=color_rewards, 
        linewidth=2.5, label='Mean Reward', zorder=3)
ax1.tick_params(axis='y', labelcolor=color_rewards)
ax1.set_xlim([0, steps[-1] + 10])
ax1.set_ylim([max(0, np.min(critic_rewards) - 0.05), np.max(critic_rewards) + 0.05])

# Create second y-axis for response length
ax2 = ax1.twinx()
color_length = PLOT_COLORS['orange']
ax2.set_ylabel('Mean Response Length', color=color_length)
line2 = ax2.plot(steps, smoothed_length, '-', color=color_length, 
        linewidth=2.5, label='Mean Response Length', zorder=3)
ax2.tick_params(axis='y', labelcolor=color_length)
ax2.set_ylim([max(0, np.min(response_length) - 200), np.max(response_length) + 200])

# Add combined legend with all elements
lines = line1 + line2
labels = [l.get_label() for l in lines]
# Add spacing and stage labels - ensure all 3 stages appear
labels.append('')  # Add empty label for spacing
# Create handles for the 3 stages with proper colors
stage_handles = [plt.Rectangle((0,0),1,1, fc='white', alpha=0),  # spacing
                 plt.Rectangle((0,0),1,1, fc=stage_colors[0], alpha=0.3),  # Stage 1 - purple
                 plt.Rectangle((0,0),1,1, fc=stage_colors[1], alpha=0.3),  # Stage 2 - orange
                 plt.Rectangle((0,0),1,1, fc=stage_colors[2], alpha=0.3)]  # Stage 3 - teal
stage_labels = [
                'Stage 1: Foundation',
                'Stage 2: Intermediate',
                'Stage 3: Full Complexity']
ax1.legend(lines + stage_handles, labels + stage_labels, frameon=False, 
          loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=9)

# Styling
clean_spines(ax1)
clean_spines(ax2)
ax1.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.savefig('figures/grpo/grpo_dual_axis.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('figures/grpo/grpo_dual_axis.png', dpi=1000, bbox_inches='tight')
plt.close()

print(f"Saved: figures/grpo/grpo_dual_axis.pdf")

# ============================================================
# Plot 4: Combined View with Stage Backgrounds (Dual Y-axis)
# ============================================================
print("Creating combined plot with stage backgrounds...")

fig, ax1 = plt.subplots(figsize=(12, 6))

# Define training stages (matching GRPO curriculum)
training_stages = [
    (0, 780, 'Stage 1: Foundation'),
    (780, 966, 'Stage 2: Intermediate'),
    (966, 1152, 'Stage 3: Full Complexity')
]

# Add stage background shading (without labels in axvspan)
stage_colors = [PLOT_COLORS['purple'], PLOT_COLORS['orange'], PLOT_COLORS['teal']]
for idx, (start, end, stage_label) in enumerate(training_stages):
    alpha = 0.08 if idx == 0 else 0.06
    ax1.axvspan(start, end, alpha=alpha, color=stage_colors[idx % len(stage_colors)], zorder=0)

# Smooth both curves
smoothed_rewards = smooth_curve(critic_rewards, window_size=20)
smoothed_length = smooth_curve(response_length, window_size=20)

# Plot rewards on left y-axis
color_rewards = PLOT_COLORS['purple']
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Mean Reward', color=color_rewards)
line1 = ax1.plot(steps, smoothed_rewards, '-', color=color_rewards, 
        linewidth=2.5, label='Mean Reward', zorder=3)
ax1.tick_params(axis='y', labelcolor=color_rewards)
ax1.set_xlim([0, steps[-1] + 10])
ax1.set_ylim([max(0, np.min(critic_rewards) - 0.05), np.max(critic_rewards) + 0.05])

# Create second y-axis for response length
ax2 = ax1.twinx()
color_length = PLOT_COLORS['orange']
ax2.set_ylabel('Mean Response Length', color=color_length)
line2 = ax2.plot(steps, smoothed_length, '-', color=color_length, 
        linewidth=2.5, label='Mean Response Length', zorder=3)
ax2.tick_params(axis='y', labelcolor=color_length)
ax2.set_ylim([max(0, np.min(response_length) - 200), np.max(response_length) + 200])

# Add combined legend with all elements
lines = line1 + line2
labels = [l.get_label() for l in lines]
# Add spacing and stage labels - ensure all 3 stages appear
labels.append('')  # Add empty label for spacing
# Create handles for the 3 stages with proper colors
stage_handles = [plt.Rectangle((0,0),1,1, fc='white', alpha=0),  # spacing
                 plt.Rectangle((0,0),1,1, fc=stage_colors[0], alpha=0.3),  # Stage 1 - purple
                 plt.Rectangle((0,0),1,1, fc=stage_colors[1], alpha=0.3),  # Stage 2 - orange
                 plt.Rectangle((0,0),1,1, fc=stage_colors[2], alpha=0.3)]  # Stage 3 - teal
stage_labels = ['',  # spacing
                'Stage 1: Foundation',
                'Stage 2: Intermediate',
                'Stage 3: Full Complexity']
ax1.legend(lines + stage_handles, labels + stage_labels, frameon=False, 
          loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=9)

# Styling
clean_spines(ax1)
clean_spines(ax2)

plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.savefig('figures/grpo/grpo_combined_metrics.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('figures/grpo/grpo_combined_metrics.png', dpi=1000, bbox_inches='tight')
plt.close()

print(f"Saved: figures/grpo/grpo_combined_metrics.pdf")

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "="*60)
print("GRPO Training Summary Statistics")
print("="*60)

print(f"\nCritic Rewards/Mean:")
print(f"  Initial: {critic_rewards[0]:.4f}")
print(f"  Final: {critic_rewards[-1]:.4f}")
print(f"  Maximum: {np.max(critic_rewards):.4f} (step {steps[np.argmax(critic_rewards)]})")
print(f"  Minimum: {np.min(critic_rewards):.4f} (step {steps[np.argmin(critic_rewards)]})")
print(f"  Mean: {np.mean(critic_rewards):.4f}")
print(f"  Std: {np.std(critic_rewards):.4f}")

print(f"\nResponse Length/Mean:")
print(f"  Initial: {response_length[0]:.1f}")
print(f"  Final: {response_length[-1]:.1f}")
print(f"  Maximum: {np.max(response_length):.1f} (step {steps[np.argmax(response_length)]})")
print(f"  Minimum: {np.min(response_length):.1f} (step {steps[np.argmin(response_length)]})")
print(f"  Mean: {np.mean(response_length):.1f}")
print(f"  Std: {np.std(response_length):.1f}")

print(f"\nTraining Progress:")
print(f"  Total steps: {len(steps)}")
print(f"  Step range: {steps[0]} to {steps[-1]}")
if 'source_file' in df.columns:
    print(f"  Source files: {len(df['source_file'].unique())}")

print("\n" + "="*60)
print("All plots saved to figures/grpo/")
print("="*60)
