import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('sensor_data_merged.csv')
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values(['Location', 'Time']).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Time'].min()} to {df['Time'].max()}")

# Create target
print("\nCreating target variable...")
df['is_flood'] = (df['State'] == 'High High').astype(int)

# Create lagged water level features
print("Creating lagged features...")
for location in df['Location'].unique():
    mask = df['Location'] == location
    location_df = df.loc[mask].copy()
    
    # Extract water level from Value column
    try:
        water_levels = location_df['Value'].str.replace(' In.', '').str.strip().astype(float)
    except:
        water_levels = location_df['Value']
    
    # Create lag 1 hour
    df.loc[mask, 'water_level_lag_1h'] = water_levels.shift(1).values
    
    # Create lag 1 hour for precipitation
    df.loc[mask, 'precipitation_lag_1h'] = location_df['precipitation_mm_hourly'].shift(1).values

# Drop NaN values created by lagging
df = df.dropna(subset=['water_level_lag_1h', 'precipitation_lag_1h'])

print(f"Rows after lagging: {len(df)}")
print(f"Flood samples: {df['is_flood'].sum()}")
print(f"Non-flood samples: {(df['is_flood'] == 0).sum()}")

# Select features - NO location dummies, just weather and water patterns
features = [
    'water_level_lag_1h',  # Previous hour water level
    'precipitation_lag_1h',  # Previous hour precipitation
    'precipitation_mm_hourly',
    'temperature_c_hourly',
    'humidity_percent_hourly',
    'wind_speed_kmh_hourly',
    'pressure_hpa_hourly',
    'cloud_cover_percent_hourly',
    'forecasted_precipitation_hourly',
    'San_Jacinto_rain_5min',
    'Caney_Creek_rain_5min',
    'Month'
]

# Prepare data
X = df[features].fillna(df[features].mean())
y = df['is_flood']

print(f"\nFeatures: {features}")
print(f"Total samples: {len(df)}")
print(f"High High samples: {(y == 1).sum()}")
print(f"Other samples: {(y == 0).sum()}")
print(f"Class distribution: {(y == 1).sum() / len(y) * 100:.2f}% High High")

# Split data - RANDOM SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Results
print("\n" + "="*60)
print("RANDOM FOREST RESULTS")
print("="*60)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Flood', 'High High']))

# Feature importance
print(f"\nFeature Importance (Top 10):")
feature_imp = sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in feature_imp[:10]:
    print(f"  {feat:40s}: {imp:.4f}")

print(f"\n✓ Model trained successfully!")
print(f"✓ Ready to make predictions")


