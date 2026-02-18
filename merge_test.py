import pandas as pd

# Read both CSV files
df_avalon = pd.read_csv(r'c:\Users\USER\Downloads\Simplicity Project\sesnsor_data\84_avalon.csv')
df_houston = pd.read_csv(r'c:\Users\USER\Downloads\Simplicity Project\east_houston_weather.csv')

# Convert Time columns to datetime
df_avalon['Time'] = pd.to_datetime(df_avalon['Time'])
df_houston['Date'] = pd.to_datetime(df_houston['Date'])

# Extract date and hour for matching
df_avalon['DateHour'] = df_avalon['Time'].dt.floor('H')
df_houston['DateHour'] = df_houston['Date'].dt.floor('H')

# Merge on DateHour (left join to keep all Avalon rows)
merged = pd.merge(df_avalon, df_houston, on='DateHour', how='left')

print(f"Avalon rows: {len(df_avalon)}")
print(f"Houston rows: {len(df_houston)}")
print(f"Merged rows: {len(merged)} (kept all Avalon rows)")
print(f"\nRows with Houston data: {merged['Temperature (C)'].notna().sum()}")
print(f"Rows without Houston data: {merged['Temperature (C)'].isna().sum()}")

print(f"\nFirst few rows of merge:")
print(merged[['Time', 'Value', 'Temperature (C)', 'Humidity (%)', 'Wind Speed (km/h)']].head(10))

# Save to test file (drop DateHour column from output)
merged_output = merged.drop('DateHour', axis=1)
merged_output.to_csv(r'c:\Users\USER\Downloads\Simplicity Project\test_merged_avalon_houston.csv', index=False)
print(f"\n✓ Test merge saved to: test_merged_avalon_houston.csv")
