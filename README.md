# Dengue Cluster Prediction in Singapore

> Predicting the emergence and intensity of dengue outbreak clusters **2 weeks in advance** using localised weather patterns and historical case data.

Singapore experiences recurring dengue outbreaks driven by complex interactions between climate, urban density, and vector behaviour. This project builds a geospatial machine learning pipeline that forecasts future active cluster locations at the **URA subzone level**, giving public health agencies an actionable early-warning signal.

---

## Highlights

- **End-to-end pipeline** — raw data ingestion, spatial feature engineering, model training, and evaluation in a single notebook
- **Geospatial reasoning** — dengue snapshots geocoded and joined to URA planning subzones via `geopandas`
- **Multi-source data fusion** — cluster case records, daily rainfall, 4-day weather forecasts, and national disease bulletins aligned on a common weekly grid
- **Two deep learning architectures** compared: LSTM and a custom dual-input Transformer
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
    ├── inspect_sgcharts.py           # Utility: merge all SGCharts CSVs → dengue_all.csv
    ├── lstm_active_only.py           # LSTM on active-only subzone-weeks (rolling CV)
    ├── transformer_v1.py             # Single-stream Transformer with forecast weather steps
    ├── transformer_dual.py           # Dual-input Transformer (local + national trend)
    └── transformer_holdout.py        # Side-by-side holdout evaluation: V1 vs V2
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

### LSTM — Active-Only
Trained exclusively on subzone-weeks with at least one recorded case, removing the heavy class imbalance of zero-case weeks. Rolling-window cross-validation with three precision threshold strategies: fixed count, per-subzone historical mean, and area-proportional calibration.

### Transformer V1 — Single-Stream
An 8-step input sequence combining 6 weeks of historical weather and case data with 2 weeks of forecast weather, allowing the model to incorporate forward-looking meteorological signals. Evaluated on 2019 and 2020 test folds.

### Transformer V2 — Dual-Input
Two parallel input streams — a local sequence encoder and a separate MLP processing the national case trend — are fused before the prediction head. This disentangles local cluster dynamics from island-wide outbreak momentum. Evaluated on 2019 and 2020 test folds.

### Holdout Comparison
V1 and V2 are benchmarked on a fixed 10% holdout (every 10th week across 2016–2020) to measure generalisation independently of fold choice.

---

## Setup

```bash
pip install pandas numpy geopandas matplotlib scipy torch scikit-learn
```

Open the notebook:

```bash
jupyter notebook dengue_cluster_prediction.ipynb
```
