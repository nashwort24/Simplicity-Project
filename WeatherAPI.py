import requests
import pandas as pd

# ---- SET YOUR LOCATION ----
lat = 29.7604
lon = -95.3698

headers = {
    "User-Agent": "your-email@example.com"
}

# ---- STEP 1: GET GRID INFO ----
points_url = f"https://api.weather.gov/points/{lat},{lon}"
points_response = requests.get(points_url, headers=headers)
points_data = points_response.json()

grid_id = points_data['properties']['gridId']
grid_x = points_data['properties']['gridX']
grid_y = points_data['properties']['gridY']

# ---- STEP 2: GET HOURLY FORECAST ----
forecast_url = f"https://api.weather.gov/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast/hourly"
forecast_response = requests.get(forecast_url, headers=headers)
forecast_data = forecast_response.json()

# ---- STEP 3: CONVERT TO DATAFRAME ----
periods = forecast_data['properties']['periods']
weather_df = pd.DataFrame(periods)

weather_df['timestamp'] = pd.to_datetime(weather_df['startTime'])

weather_df = weather_df[['timestamp', 'temperature', 'windSpeed', 'shortForecast']]

print(weather_df.head())

weather_df.to_csv("weather_data.csv", index=False)

