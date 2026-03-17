# Exploratory & Descriptive Analysis: sensor_data_merged.csv

## Dataset Overview
- **Total Records**: 138,802
- **Total Columns**: 21
- **Date Range**: April 15, 2025 - December 20, 2025
- **Data Coverage**: 5 sensor locations across Houston area

---

## 1. Numeric Variable Summary Statistics

All numeric variables reported with: **Mean, Median, Std Dev, Min, Max**

| Variable | Mean | Median | Std Dev | Min | Max |
|----------|------|--------|---------|-----|-----|
| San_Jacinto_rain_5min | 0.0004 | 0.0 | 0.0080 | 0.0 | 0.40 |
| Caney_Creek_rain_5min | 0.0003 | 0.0 | 0.0066 | 0.0 | 0.48 |
| temperature_c_hourly | 27.3437 | 28.0 | 5.9062 | 0.0 | 38.30 |
| humidity_percent_hourly | 69.4946 | 72.0 | 19.1566 | 21.0 | 100.00 |
| precipitation_mm_hourly | 0.0956 | 0.0 | 0.9410 | 0.0 | 27.20 |
| wind_direction_deg_hourly | 141.5387 | 160.0 | 97.1497 | 0.0 | 360.00 |
| wind_speed_kmh_hourly | 9.7880 | 9.4 | 6.7139 | 0.0 | 46.00 |
| wind_gust_kmh_hourly | 20.3097 | 19.0 | 4.5860 | 14.0 | 38.00 |
| pressure_hpa_hourly | 1016.5433 | 1016.4 | 3.6363 | 1004.5 | 1036.00 |
| cloud_cover_percent_hourly | 3.4074 | 2.0 | 1.9810 | 0.0 | 8.00 |
| weather_code_hourly | 3.4004 | 3.0 | 3.3172 | 1.0 | 25.00 |
| forecasted_precipitation_hourly | 0.1873 | 0.0 | 0.9512 | 0.0 | 23.20 |
| Month | 7.9031 | 7.0 | 1.7176 | 4.0 | 12.00 |

**Key Observations**:
- Rain measurements are sparse (most values = 0), skewed distributions
- Temperature ranges from 0-38.3°C with mean of 27.3°C (hot climate)
- Humidity typically high (mean 69.5%, median 72%)
- Wind predominantly from southwest (mean direction 141.5°, median 160°)
- Pressure stable (mean 1016.5 hPa ± 3.6)

---

## 2. Categorical Variable Breakdown

### State (Water Level Classification)
| Category | Count | Percentage |
|----------|-------|------------|
| Normal | 59,695 | 43.01% |
| High High | 38,015 | 27.39% |
| Underrange | 34,264 | 24.69% |
| Low Low | 5,178 | 3.73% |
| High | 1,638 | 1.18% |
| Overrange | 12 | 0.01% |

**Interpretation**: Majority of readings are Normal (43%), with significant High High readings (27%), suggesting periodic flood-risk conditions.

### Location (Sensor Sites)
| Location | Count | Percentage |
|----------|-------|------------|
| 84_bitter root | 30,745 | 22.15% |
| 84_avalon | 29,127 | 20.98% |
| 84_US_59 | 27,150 | 19.56% |
| 84_southwood_oaks | 25,932 | 18.68% |
| Southwood Laverne | 25,848 | 18.62% |

**Interpretation**: Balanced representation across all 5 locations (~20% each), suggesting uniform data collection.

### Season
| Season | Count | Percentage |
|--------|-------|------------|
| Summer | 87,573 | 63.09% |
| Fall | 38,433 | 27.69% |
| Spring | 9,926 | 7.15% |
| Winter | 2,870 | 2.07% |

**Interpretation**: Heavy summer dominance (63%), consistent with Houston's subtropical climate. Limited winter data (2%), indicating data collection during warmer months.

---

## 3. Correlation Matrix - Numeric Variables

### Strong Correlations (|r| > 0.3)

| Variable 1 | Variable 2 | Correlation | Interpretation |
|-----------|-----------|-------------|-----------------|
| Temperature | Month | **-0.5677** | Strong negative: Temperature decreases as months progress (Apr→Dec) |
| Temperature | Humidity | **-0.5251** | Strong negative: Higher temps associated with lower humidity |
| Wind Speed | Wind Gust | **0.5000** | Moderate positive: Stronger gusts when base wind is stronger |
| Humidity | Wind Speed | **-0.3892** | Moderate negative: High humidity associated with calmer winds |
| Wind Direction | Wind Speed | **0.3644** | Weak positive: Certain wind directions slightly stronger |
| Temperature | Pressure | **-0.3193** | Weak negative: Cooler temps with higher pressure systems |
| Temperature | Wind Gust | **-0.3185** | Weak negative: Cooler temps have stronger wind gusts |
| Pressure | Month | **0.3054** | Weak positive: Pressure increases through end of year |

### Weak/Negligible Correlations
- **Rain variables** show very weak correlations with most weather metrics (rain is sparse event)
- **Cloud cover** shows weak correlations across the board
- **Precipitation forecast** shows weak correlations with observed precipitation (forecast error)

---

## 4. Data Quality Notes

### Missing Values
- **snow_depth_cm_hourly**: 100% missing (not applicable to Houston climate)
- **sunshine_min_hourly**: 100% missing (not available in dataset)
- **precipitation_mm_hourly**: 6% missing (sparse rainfall events)
- **wind_gust_kmh_hourly**: 93% missing (sporadic measurement)
- **All other weather metrics**: <0.05% missing

### Data Integrity
✅ No duplicate timestamps per location  
✅ Consistent date range (Apr 15 - Dec 20, 2025)  
✅ All numeric values within expected ranges  
✅ Categorical values properly classified  

---

## 5. Recommendations for Analysis

1. **Modeling**: Temperature is highly seasonal (r=-0.57 with month) - consider seasonal decomposition
2. **Feature Engineering**: Wind speed and gust are strongly correlated (r=0.50) - may need dimensionality reduction
3. **Precipitation Modeling**: Rain events are rare but important - consider separate event classification
4. **Temporal Patterns**: Strong seasonal patterns evident - ARIMA or seasonal models recommended
5. **Data Imbalance**: Winter months underrepresented - stratified cross-validation recommended
6. **Missing Forecasts**: Precipitation forecast has different distribution than observed - investigate bias

---

## Notebook Cells Summary

The analysis in `Exploratory_Analysis.ipynb` includes:

1. **Data Loading & Inspection** - Overview of 138K records across 21 columns
2. **Numeric Summary Statistics** - Table with all 5 summary statistics
3. **Categorical Breakdowns** - Percentage distributions for State, Location, Season
4. **Correlation Matrix** - Full correlation table + heatmap visualization
5. **Missing Value Analysis** - Data quality assessment
6. **Distribution Plots** - Histograms for each numeric variable
7. **Boxplots** - Outlier detection visualizations
8. **Summary Report** - Key findings and insights

All visualizations (heatmap, distributions, boxplots) included for publication-ready analysis.

---

*Analysis completed: February 19, 2026*
