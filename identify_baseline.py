import pandas as pd
import numpy as np

# Load the data
csv_path = r'c:\Users\USER\Downloads\Simplicity Project\sensor_data_merged.csv'
df = pd.read_csv(csv_path)

# Clean the Value column - remove " In." and convert to float
df['Value_numeric'] = df['Value'].str.replace(' In.', '').astype(float)

# Get unique locations
locations = sorted(df['Location'].unique())

print("="*70)
print("BASELINE DEFINITION: ±1 OF MEDIAN VALUE")
print("="*70 + "\n")

for location in locations:
    location_data = df[df['Location'] == location]['Value_numeric']
    median = location_data.median()
    baseline_lower = median - 1
    baseline_upper = median + 1
    
    baseline_count = ((location_data >= baseline_lower) & (location_data <= baseline_upper)).sum()
    baseline_percent = (baseline_count / len(location_data)) * 100
    
    print(f"Location: {location}")
    print(f"  Median: {median:.4f}")
    print(f"  Baseline range (median ± 1): [{baseline_lower:.4f}, {baseline_upper:.4f}]")
    print(f"  Baseline values: {baseline_count} ({baseline_percent:.1f}%)")
    print(f"  Non-baseline values: {len(location_data) - baseline_count} ({100-baseline_percent:.1f}%)")
    print()

