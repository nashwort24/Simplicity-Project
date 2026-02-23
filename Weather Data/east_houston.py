from datetime import datetime, timedelta
import meteostat
import sys

station = '72243'

def parse_date(date_str):
    if ',' in date_str:
        parts = date_str.split(',')
        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
        year = 2000 + year if year < 100 else year
        return datetime(year, month, day)
    else:
        return datetime.strptime(date_str, "%Y-%m-%d")

if len(sys.argv) == 3:
    start = parse_date(sys.argv[1])
    end = parse_date(sys.argv[2])
else:
    end = datetime.now()
    start = end - timedelta(days=7)

data = meteostat.hourly(station, start, end).fetch()

# Print results
print(f"East Houston (Intercontinental) - Hourly weather data ({len(data)} records):\n")
print(data.head(20))

# Prepare data for CSV with readable columns
data_csv = data.reset_index()
data_csv.columns = ['Date', 'Temperature (C)', 'Humidity (%)', 'Precipitation (mm)', 'Snow Depth (cm)', 
                     'Wind Direction (deg)', 'Wind Speed (km/h)', 'Wind Gust (km/h)', 'Pressure (hPa)', 'Sunshine (min)', 
                     'Cloud Cover (%)', 'Weather Code']

data_csv.to_csv("east_houston_weather.csv", index=False)