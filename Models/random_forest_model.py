import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r'c:\Users\USER\Downloads\Simplicity Project\test_merged_avalon_houston.csv')

# Create binary target: 1 if "High High", 0 otherwise
df['is_high_high'] = (df['State'] == 'High High').astype(int)

# Select weather features (drop NaN values)
features = ['Temperature (C)', 'Humidity (%)', 'Precipitation (mm)', 'Wind Speed (km/h)', 
            'Wind Gust (km/h)', 'Pressure (hPa)', 'Cloud Cover (%)', 'Weather Code']

# Prepare data
X = df[features].fillna(df[features].mean())
y = df['is_high_high']

print(f"Total samples: {len(df)}")
print(f"High High samples: {(y == 1).sum()}")
print(f"Other samples: {(y == 0).sum()}")
print(f"Class distribution: {(y == 1).sum() / len(y) * 100:.1f}% High High")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Results
print("\n" + "="*60)
print("RANDOM FOREST RESULTS")
print("="*60)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not High High', 'High High']))

# Feature importance
print(f"\nFeature Importance:")
for feat, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.4f}")

print(f"\n✓ Model trained successfully!")
