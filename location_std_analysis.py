import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
csv_path = Path(__file__).parent / 'sensor_data_merged.csv'
df = pd.read_csv(csv_path)

# Clean the Value column - remove " In." and convert to float
df['Value_numeric'] = df['Value'].str.replace(' In.', '').astype(float)

# Exclude baseline and near-baseline values - keep only peaks (positive values > 0.5)
df = df[df['Value_numeric'] > 0.5]
print("Baseline and near-baseline values excluded. Only peaks retained.\n")

# Get unique locations
locations = df['Location'].unique()
print("Locations found:", locations)
print("\n" + "="*50)
print("STANDARD DEVIATION ANALYSIS BY LOCATION")
print("="*50 + "\n")

# Create a dictionary to store std dev results
std_results = {}
three_sigma_thresholds = {}

# Calculate standard deviation for each location
for location in locations:
    location_data = df[df['Location'] == location]['Value_numeric']
    std_dev = location_data.std()
    std_results[location] = std_dev
    count = len(location_data)
    mean = location_data.mean()
    three_sigma = mean + (3 * std_dev)
    three_sigma_thresholds[location] = three_sigma
    
    print(f"Location: {location}")
    print(f"  Count: {count}")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std Dev: {std_dev:.4f}")
    print(f"  Mean + 3σ (3-Sigma Threshold): {three_sigma:.4f}")
    print()

# 1. Bar plot of standard deviations for all locations
fig, ax = plt.subplots(figsize=(10, 6))
locations_sorted = sorted(std_results.keys())

print("\n" + "="*50)
print("VALUES EXCEEDING 3-SIGMA THRESHOLD")
print("="*50 + "\n")

# Count values above 3-sigma threshold
for location in locations_sorted:
    location_data = df[df['Location'] == location]['Value_numeric']
    three_sigma = three_sigma_thresholds[location]
    count_above_3sigma = (location_data > three_sigma).sum()
    percent_above = (count_above_3sigma / len(location_data)) * 100
    
    print(f"Location: {location}")
    print(f"  3-Sigma Threshold: {three_sigma:.4f} In.")
    print(f"  Count above 3σ: {count_above_3sigma} out of {len(location_data)}")
    print(f"  Percentage: {percent_above:.2f}%")
    print()
stds = [std_results[loc] for loc in locations_sorted]
colors = sns.color_palette("husl", len(locations_sorted))
ax.bar(locations_sorted, stds, color=colors)
ax.set_xlabel('Location', fontsize=12, fontweight='bold')
ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
ax.set_title('Standard Deviation of Sensor Values by Location', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'std_dev_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: std_dev_comparison.png")
plt.close()

# 2. Individual plots for each location showing the distribution of values
fig, axes = plt.subplots(len(locations_sorted), 2, figsize=(14, 4*len(locations_sorted)))
if len(locations_sorted) == 1:
    axes = axes.reshape(1, -1)

for idx, location in enumerate(locations_sorted):
    location_data = df[df['Location'] == location]['Value_numeric']
    
    # Histogram
    axes[idx, 0].hist(location_data, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
    axes[idx, 0].set_xlabel('Value', fontsize=10)
    axes[idx, 0].set_ylabel('Frequency', fontsize=10)
    axes[idx, 0].set_title(f'{location} - Distribution', fontsize=11, fontweight='bold')
    axes[idx, 0].axvline(location_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {location_data.mean():.2f}')
    axes[idx, 0].legend()
    
    # Time series plot
    location_time_data = df[df['Location'] == location].copy()
    location_time_data['Time'] = pd.to_datetime(location_time_data['Time'])
    location_time_data = location_time_data.sort_values('Time')
    
    axes[idx, 1].plot(location_time_data['Time'], location_time_data['Value_numeric'], 
                      color=colors[idx], linewidth=1.5, marker='o', markersize=3, alpha=0.7)
    axes[idx, 1].set_xlabel('Time', fontsize=10)
    axes[idx, 1].set_ylabel('Value', fontsize=10)
    axes[idx, 1].set_title(f'{location} - Time Series', fontsize=11, fontweight='bold')
    axes[idx, 1].tick_params(axis='x', rotation=45)
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'location_analysis_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: location_analysis_detailed.png")
plt.close()

# 3. Box plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
data_by_location = [df[df['Location'] == loc]['Value_numeric'].values for loc in locations_sorted]
bp = ax.boxplot(data_by_location, labels=locations_sorted, patch_artist=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Location', fontsize=12, fontweight='bold')
ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('Box Plot of Sensor Values by Location', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'location_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: location_boxplot.png")
plt.close()

print("\n" + "="*50)
print("Analysis complete!")
print("="*50)
print("\nGenerated files:")
print("  1. std_dev_comparison.png - Bar chart comparing std deviations")
print("  2. location_analysis_detailed.png - Detailed distributions and time series")
print("  3. location_boxplot.png - Box plots for each location")
