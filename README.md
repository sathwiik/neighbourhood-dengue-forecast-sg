# Dengue Cluster Prediction in Singapore

> Predicting the emergence and intensity of dengue outbreak clusters **2 weeks in advance** using localised weather patterns and historical case data.

Singapore experiences recurring dengue outbreaks driven by complex interactions between climate, urban density, and vector behaviour. This project builds a geospatial machine learning pipeline that forecasts future active cluster locations at the **URA subzone level**, giving public health agencies an actionable early-warning signal.

---

## Highlights

- **End-to-end pipeline** — raw data ingestion, spatial feature engineering, model training, and evaluation in a single notebook
- **Geospatial reasoning** — dengue snapshots geocoded and joined to URA planning subzones via `geopandas`
- **Multi-source data fusion** — cluster case records, daily rainfall, 4-day weather forecasts, and national disease bulletins aligned on a common weekly grid
- **Custom Transformer encoder** benchmarked against a logistic regression baseline
- **Rigorous evaluation** — rolling time-series cross-validation with area-calibrated precision thresholds

---

## Repository Structure

```
dengue-cluster-prediction/
├── dengue_cluster_prediction.ipynb   # Full pipeline: data → features → models → results
├── data/
│   └── raw/                          # Source data (not tracked — see Data Sources)
│       ├── sgcharts/                 # Dengue cluster snapshots (CSV per date)
│       ├── SC3021_rainfall/          # Daily rainfall by station, 2016
│       ├── Historical4dayWeatherForecast{2016..2020}.csv
│       ├── WeeklyInfectiousDiseaseBulletinCases.csv
│       └── ura_subzones.geojson      # URA planning subzone boundaries
└── scripts/
    ├── pipeline.py                   # Data loading → feature engineering → model training
    └── inspect_sgcharts.py           # Utility: merge all SGCharts CSVs → dengue_all.csv
```

---

## Data Sources

| # | Dataset | Provider | Granularity | Period |
|---|---------|----------|-------------|--------|
| 1 | Dengue Cluster Snapshots | SGCharts / NEA | Street-address, every 4–8 days | 2013 – 2020 |
| 2 | Weekly Infectious Disease Bulletin | Ministry of Health | National weekly total | 2016 – 2020 |
| 3 | Daily Rainfall by Station | NEA via Data.gov.sg | Station-level daily | 2016 – 2020 |
| 4 | 4-Day Weather Forecast Records | NEA via Data.gov.sg | Station-level daily | 2016 – 2020 |
| 5 | URA Subzone Boundaries | Urban Redevelopment Authority | Polygon (GeoJSON) | — |

---

## Models

### Transformer (Primary Model)
A custom Transformer encoder trained to predict subzone-level dengue case counts **2 weeks ahead**. Self-attention allows the model to directly relate any two weeks in the input window — a rainfall event in week 1 and a temperature spike in week 5 can jointly influence the prediction, unlike sequential models that process steps one at a time.

- **Input:** 8-step sequence — 6 historical weeks + 2 forecast weather steps
- **Features per step:** weather (8) + national cases (1) + local cases (1) + forecast flag + seasonality = 13 dims
- **Outbreak threshold:** NEA epidemic definition (30 cases / 100,000 / week) converted to a per-km² area-calibrated threshold
- **Loss:** weighted MSE giving outbreak weeks up to 9× more influence, preventing collapse to predicting the mean

### Logistic Regression (Baseline)
A standard logistic regression trained on the same features and folds, used as a performance benchmark.

### Evaluation
Two temporal folds with rolling forward training windows:

| Fold | Train | Test |
|------|-------|------|
| 1 | 2016 – 2018 | 2019 |
| 2 | 2016 – 2019 | 2020 |

The Transformer outperforms logistic regression on the 2019 fold. On the 2020 fold — which saw a 2.2× national case spike — the baseline benefits from higher false-positive rates being rewarded by the unusual outbreak density.

---

## Setup

```bash
pip install pandas numpy geopandas matplotlib scipy torch scikit-learn
```

Open the notebook:

```bash
jupyter notebook dengue_cluster_prediction.ipynb
```
