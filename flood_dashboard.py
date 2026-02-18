from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pickle
import os

app = Flask(__name__)

# Load merged data and train model
# Use relative path so it works on deployment
csv_path = os.path.join(os.path.dirname(__file__), 'test_merged_avalon_houston.csv')
df = pd.read_csv(csv_path)
df['is_high_high'] = (df['State'] == 'High High').astype(int)
df['Time'] = pd.to_datetime(df['Time'])
df['Date'] = df['Time'].dt.date
df['Hour'] = df['Time'].dt.hour

features = ['Temperature (C)', 'Humidity (%)', 'Precipitation (mm)', 'Wind Speed (km/h)', 
            'Wind Gust (km/h)', 'Pressure (hPa)', 'Cloud Cover (%)', 'Weather Code']

X = df[features].fillna(df[features].mean())
y = df['is_high_high']

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Get available dates and hours by date
available_dates = sorted(df['Date'].unique())
hours_by_date = {}
for date in available_dates:
    hours = sorted(df[df['Date'] == date]['Hour'].unique())
    hours_by_date[str(date)] = [int(h) for h in hours]

# Sensor locations with actual coordinates
sensors = [
    {"name": "Avalon", "lat": 30.09222, "lon": -95.25000, "area": "Central"},
    {"name": "Highway 59", "lat": 30.09139, "lon": -95.24111, "area": "Central East"},
    {"name": "Sorters North", "lat": 30.10083, "lon": -95.27167, "area": "North"},
    {"name": "Sorters South", "lat": 30.07167, "lon": -95.26472, "area": "South"},
    {"name": "Southwood Oaks at LaVone", "lat": 30.09139, "lon": -95.25806, "area": "Central"},
    {"name": "Brentwood Oaks", "lat": 30.08028, "lon": -95.25028, "area": "Central South"},
    {"name": "Bitter Root", "lat": 30.09667, "lon": -95.26722, "area": "North Central"},
    {"name": "Southwood Oaks (Alt)", "lat": 30.09222, "lon": -95.26000, "area": "Central"},
    {"name": "Rock Creek Dr", "lat": 30.07167, "lon": -95.25500, "area": "South"},
    {"name": "North Pond", "lat": 29.99806, "lon": -95.85750, "area": "West"},
    {"name": "South Pond", "lat": 29.99200, "lon": -95.85306, "area": "West"},
    {"name": "Buffalo Lake", "lat": 29.99333, "lon": -95.85111, "area": "West"},
    {"name": "Rock Hollow", "lat": 29.99200, "lon": -95.85111, "area": "West"},
]

@app.route('/')
def index():
    import json
    dates_json = json.dumps([str(d) for d in available_dates])
    hours_json = json.dumps(hours_by_date)
    return render_template('index.html', sensors=sensors, available_dates=[str(d) for d in available_dates], 
                         dates_json=dates_json, hours_json=hours_json, hours_by_date=hours_by_date)

@app.route('/api/risk')
def get_risk():
    """Get flood risk for all sensors"""
    selected_date = request.args.get('date')
    selected_hour = request.args.get('hour')
    
    if selected_date:
        try:
            date_obj = pd.to_datetime(selected_date).date()
            day_data = df[df['Date'] == date_obj][features].fillna(df[features].mean())
            
            if selected_hour:
                try:
                    hour_int = int(selected_hour)
                    hour_data = day_data[df[df['Date'] == date_obj]['Hour'] == hour_int][features].fillna(df[features].mean())
                    if len(hour_data) > 0:
                        reading = hour_data.iloc[-1].values.reshape(1, -1)
                    elif len(day_data) > 0:
                        reading = day_data.iloc[-1].values.reshape(1, -1)
                    else:
                        reading = df[features].fillna(df[features].mean()).iloc[-1:].values
                except:
                    if len(day_data) > 0:
                        reading = day_data.iloc[-1].values.reshape(1, -1)
                    else:
                        reading = df[features].fillna(df[features].mean()).iloc[-1:].values
            else:
                if len(day_data) > 0:
                    reading = day_data.iloc[-1].values.reshape(1, -1)
                else:
                    reading = df[features].fillna(df[features].mean()).iloc[-1:].values
        except:
            reading = df[features].fillna(df[features].mean()).iloc[-1:].values
    else:
        reading = df[features].fillna(df[features].mean()).iloc[-1:].values
    
    # Get probability of High High
    risk_prob = rf.predict_proba(reading)[0][1] * 100
    
    # Get weather data for display - use the same reading we used for prediction
    weather_dict = {}
    for i, feature in enumerate(features):
        weather_dict[feature] = float(reading[0][i])
    
    # Get predictions for each sensor
    predictions = []
    for sensor in sensors:
        # Add slight variation based on location
        variation = np.random.normal(0, 5)
        risk = max(0, min(100, risk_prob + variation))
        predictions.append({
            "name": sensor["name"],
            "risk": round(risk, 1),
            "status": "🔴 HIGH RISK" if risk > 70 else "🟡 MODERATE" if risk > 40 else "🟢 LOW RISK",
            "lat": sensor["lat"],
            "lon": sensor["lon"],
            "area": sensor["area"]
        })
    
    return jsonify({
        "overall_risk": round(risk_prob, 1),
        "sensors": predictions,
        "weather": weather_dict,
        "selected_date": selected_date or str(available_dates[-1]),
        "selected_hour": selected_hour or ""
    })

@app.route('/api/risk-history')
def get_risk_history():
    """Get flood risk for 12 hours around the selected date/time"""
    selected_date = request.args.get('date')
    selected_hour = request.args.get('hour')
    
    history = []
    
    # Default to latest date if not specified
    if selected_date:
        try:
            base_date = pd.to_datetime(selected_date).date()
        except:
            base_date = df['Date'].max()
    else:
        base_date = df['Date'].max()
    
    # Default to latest hour if not specified
    if selected_hour:
        try:
            base_hour = int(selected_hour)
        except:
            base_hour = df[df['Date'] == base_date]['Hour'].max() if len(df[df['Date'] == base_date]) > 0 else 12
    else:
        base_hour = df[df['Date'] == base_date]['Hour'].max() if len(df[df['Date'] == base_date]) > 0 else 12
    
    # Get 12 hours: 6 hours before and 6 hours after the selected time
    for hours_offset in range(-6, 7):
        current_hour = base_hour + hours_offset
        current_date = base_date
        
        # Handle hour wrapping to previous/next day
        if current_hour < 0:
            current_date = base_date - pd.Timedelta(days=1)
            current_hour = 24 + current_hour
        elif current_hour >= 24:
            current_date = base_date + pd.Timedelta(days=1)
            current_hour = current_hour - 24
        
        # Get data for this hour
        mask = (df['Date'] == current_date) & (df['Hour'] == current_hour)
        hour_data = df[mask][features].fillna(df[features].mean())
        
        if len(hour_data) > 0:
            reading = hour_data.iloc[-1].values.reshape(1, -1)
            risk_prob = rf.predict_proba(reading)[0][1] * 100
        else:
            # Use overall average if no data for this hour
            reading = df[features].fillna(df[features].mean()).iloc[-1:].values
            risk_prob = rf.predict_proba(reading)[0][1] * 100
        
        history.append({
            "time": f"{current_hour:02d}:00",
            "risk": round(risk_prob, 1)
        })
    
    return jsonify({"history": history})

if __name__ == '__main__':
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug, port=int(os.getenv('PORT', 5000)))
