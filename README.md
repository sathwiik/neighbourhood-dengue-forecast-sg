# Neighbourhood Dengue Forecast — Singapore

> Predicting the emergence and intensity of dengue outbreak clusters **2 weeks in advance** at the neighbourhood level, using localised weather patterns and historical case data.

Singapore experiences recurring dengue outbreaks driven by complex interactions between climate, urban density, and vector behaviour. This project builds a full data science pipeline — from raw government data through geospatial processing, exploratory analysis, and deep learning — to forecast active cluster locations at the **URA subzone level**, giving public health agencies an actionable early-warning signal.

---

## Pipeline Overview

### 1. Data Collection
Five public datasets were identified, sourced, and justified against domain requirements:

| Dataset | Provider | Granularity | Period |
|---------|----------|-------------|--------|
| Dengue Cluster Snapshots | SGCharts / NEA | Street-address, every 4–8 days | 2013 – 2020 |
| Weekly Infectious Disease Bulletin | Ministry of Health | National weekly total | 2016 – 2020 |
| Daily Rainfall by Station | NEA via Data.gov.sg | Station-level daily | 2016 – 2020 |
| 4-Day Weather Forecast Records | NEA via Data.gov.sg | Island-wide daily | 2016 – 2020 |
| URA Subzone Boundaries | Urban Redevelopment Authority | Polygon (GeoJSON) | — |

### 2. Exploratory Data Analysis
Each dataset was explored independently before any merging:
- **Rainfall:** visualised station-level daily totals across 2016–2020; verified range (0–217 mm/day) against Singapore climatological norms
- **Temperature & Humidity:** plotted weekly averages; applied domain-bounded cleaning (e.g. temp 26–37°C high, 40–100% humidity) to remove erroneous sensor readings
- **Dengue clusters:** inspected snapshot frequency by year, case distributions per locale, and confirmed the 2020 spike (2.2× normal) as a real outbreak rather than noise
- **National bulletin:** charted epi-week trends; confirmed alignment with known outbreak years (2016, 2019, 2020)

### 3. Data Cleaning
- Removed out-of-range weather readings using Singapore-specific meteorological bounds
- Fixed a confirmed bad coordinate (geocoding error for one cluster address) with a manual correction
- Handled heterogeneous rainfall formats — 2016 used monthly station CSVs; 2017–2020 used 5-minute sensor readings aggregated to daily totals on load
- Imputed missing temperature and humidity using forward/backward fill, justified by Singapore's low day-to-day climate variability

### 4. Feature Engineering
- **Spatial join:** geocoded dengue cluster addresses to URA subzone polygons using `geopandas` point-in-polygon matching
- **Weekly aggregation:** collapsed daily records to ISO week × subzone grain for all datasets
- **Nearest-station rainfall:** assigned each subzone to its closest rainfall station via KD-tree spatial lookup; computed weekly total, mean daily, and standard deviation of rainfall
- **Subzone area:** computed from the projected CRS (EPSG:3414) in km² for use in the outbreak threshold
- **Seasonality encoding:** sine/cosine of ISO week number to capture Singapore's bimodal dengue cycle
- **Forecast flag:** binary step feature distinguishing historical from forecast-weather steps in the model input

### 5. Modelling

#### Transformer (Primary Model)
A custom Transformer encoder trained to predict subzone-level case counts **2 weeks ahead**. Self-attention allows the model to directly relate any two weeks in the input window — a rainfall event in week 1 and a temperature spike in week 5 can jointly influence the prediction.

- **Input:** 8-step sequence — 6 historical weeks + 2 forecast weather steps, 13 features per step
- **Outbreak threshold:** NEA epidemic definition (30 cases / 100,000 / week) converted to a per-km² area-calibrated threshold using Singapore's population density
- **Loss:** weighted MSE giving outbreak weeks up to 9× more influence, preventing the model from collapsing to predicting the mean

#### Logistic Regression (Baseline)
Trained on the same features and folds as a performance benchmark.

### 6. Evaluation
Two temporal folds with rolling forward training windows:

| Fold | Train | Test |
|------|-------|------|
| 1 | 2016 – 2018 | 2019 |
| 2 | 2016 – 2019 | 2020 |

The Transformer outperforms logistic regression on the 2019 fold. On the 2020 fold — which saw a 2.2× national case spike — the baseline benefits from higher false-positive rates being rewarded by the unusual outbreak density, highlighting the importance of out-of-distribution robustness in public health forecasting.

---

## Repository Structure

```
neighbourhood-dengue-forecast-sg/
├── dengue_cluster_prediction.ipynb   # Full pipeline: data → EDA → features → models → results
└── data/
    └── raw/                          # Source data (not tracked — see above)
        ├── sgcharts/                 # Dengue cluster snapshots (CSV per date)
        ├── SC3021_rainfall/          # Daily rainfall by station, 2016
        ├── Historical4dayWeatherForecast{2016..2020}.csv
        ├── WeeklyInfectiousDiseaseBulletinCases.csv
        └── ura_subzones.geojson      # URA planning subzone boundaries
```

---

## Setup

```bash
pip install pandas numpy geopandas matplotlib scipy torch scikit-learn
```

```bash
jupyter notebook dengue_cluster_prediction.ipynb
```
