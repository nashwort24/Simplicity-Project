import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 5)

# Load the data
csv_path = Path(__file__).parent / 'sensor_data_merged.csv'
df = pd.read_csv(csv_path)

# Clean the Value column - remove " In." and convert to float
df['Value_numeric'] = df['Value'].str.replace(' In.', '').astype(float)

# Get unique locations
locations = sorted(df['Location'].unique())
print("Locations found:", locations)

# Calculate median for each location for baseline definition (median ± 1)
baseline_ranges = {}
for location in locations:
    location_data = df[df['Location'] == location]['Value_numeric']
    median = location_data.median()
    baseline_ranges[location] = {
        'median': median,
        'lower': median - 1,
        'upper': median + 1
    }

print("\n" + "="*70)
print("DISTRIBUTION ANALYSIS WITH REFERENCE LINES")
print("(Baseline = Median ± 1)")
print("="*70 + "\n")

# Create a dictionary to store statistics
stats = {}

# Calculate statistics for each location
for location in locations:
    location_all = df[df['Location'] == location]['Value_numeric']
    baseline_lower = baseline_ranges[location]['lower']
    baseline_upper = baseline_ranges[location]['upper']
    location_no_baseline = df[(df['Location'] == location) & 
                             ((df['Value_numeric'] < baseline_lower) | 
                              (df['Value_numeric'] > baseline_upper))]['Value_numeric']
    
    mean_all = location_all.mean()
    mean_no_baseline = location_no_baseline.mean()
    std_no_baseline = location_no_baseline.std()
    three_sigma = mean_no_baseline + (3 * std_no_baseline)
    
    stats[location] = {
        'median': baseline_ranges[location]['median'],
        'baseline_lower': baseline_lower,
        'baseline_upper': baseline_upper,
        'mean_all': mean_all,
        'mean_no_baseline': mean_no_baseline,
        'std_no_baseline': std_no_baseline,
        'three_sigma': three_sigma,
        'data_all': location_all,
        'count_all': len(location_all),
        'count_no_baseline': len(location_no_baseline)
    }
    
    print(f"Location: {location}")
    print(f"  Median: {baseline_ranges[location]['median']:.4f}")
    print(f"  Baseline range (median ± 1): [{baseline_lower:.4f}, {baseline_upper:.4f}]")
    print(f"  Total data points: {len(location_all)}")
    print(f"  Non-baseline points: {len(location_no_baseline)}")
    print(f"  Mean (all values): {mean_all:.4f}")
    print(f"  Mean (excluding baseline): {mean_no_baseline:.4f}")
    print(f"  Std Dev (excluding baseline): {std_no_baseline:.4f}")
    print(f"  Mean + 3σ (excluding baseline): {three_sigma:.4f}")
    print()

# Create visualizations
print("\n" + "="*70)
print("Generating combined distribution chart...")
print("="*70 + "\n")

# Create a single large figure with subplots for all locations (3 rows x 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

colors = sns.color_palette("husl", len(locations))

for idx, location in enumerate(locations):
    ax = axes[idx]
    data = stats[location]['data_all']
    
    # Create histogram
    ax.hist(data, bins=80, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add reference lines
    median = stats[location]['median']
    baseline_lower = stats[location]['baseline_lower']
    baseline_upper = stats[location]['baseline_upper']
    mean_all = stats[location]['mean_all']
    mean_no_baseline = stats[location]['mean_no_baseline']
    three_sigma = stats[location]['three_sigma']
    
    # Shade the baseline region
    ax.axvspan(baseline_lower, baseline_upper, alpha=0.2, color='green', label=f'Baseline (median ± 1)')
    ax.axvline(median, color='green', linestyle='-', linewidth=3, label=f'Median: {median:.4f}')
    ax.axvline(mean_all, color='red', linestyle='-', linewidth=3, label=f'Mean (all values): {mean_all:.4f}')
    ax.axvline(mean_no_baseline, color='orange', linestyle='--', linewidth=3, label=f'Mean (excluding baseline): {mean_no_baseline:.4f}')
    ax.axvline(three_sigma, color='darkred', linestyle=':', linewidth=3, label=f'Mean + 3σ: {three_sigma:.4f}')
    
    ax.set_xlabel('Value (In.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'{location}\n(Total: {stats[location]["count_all"]}, Non-baseline: {stats[location]["count_no_baseline"]})', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

# Hide the last empty subplot
axes[-1].set_visible(False)

plt.suptitle('Value Distribution by Location with Reference Lines\n(Baseline Defined as Median ± 1)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'all_distributions_combined.png', dpi=300, bbox_inches='tight')
print("✓ Saved: all_distributions_combined.png")
plt.close()

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
print("\nGenerated file:")
print("  • all_distributions_combined.png - All locations in one comprehensive chart")
