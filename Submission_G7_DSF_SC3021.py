# %% [markdown]
# # Project: Spatial Prediction of Dengue Clusters in Singapore
# ### SC3021 Lab Deliverable
# **Team:** Jerome, Sathwiik, Marvin
# **Stakeholders:** NEA / Public Health Officials
#
# **Research Question:**
# Can we predict the emergence and intensity of dengue clusters 2-4 weeks in advance
# by analysing localised meteorological patterns?

# %% [markdown]
# ## 0. Imports & Configuration

# %%
import pandas as pd
import numpy as np
import glob
import os
import sqlite3

import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Folder paths (update these if running on Colab)
DENGUE_FOLDERS = [
    "sgcharts/incorrect_latitude_longitude",
    "sgcharts/csv"
]
RAINFALL_2016_FOLDER = "SC3021_rainfall"
RAINFALL_HIST_FILES  = [
    "HistoricalRainfallacrossSingapore2017.csv.download/HistoricalRainfallacrossSingapore2017.csv",
    "HistoricalRainfallacrossSingapore2018.csv.download/HistoricalRainfallacrossSingapore2018.csv",
    "HistoricalRainfallacrossSingapore2019.csv.download/HistoricalRainfallacrossSingapore2019.csv",
]
WEATHER_FILES = [
    "Historical4dayWeatherForecast2016.csv",
    "Historical4dayWeatherForecast2017.csv",
    "Historical4dayWeatherForecast2018.csv",
    "Historical4dayWeatherForecast2019.csv",
]
URA_FILE   = "ura_subzones.geojson"
DATE_START = "2016-03-01"
DATE_END   = "2019-12-31"

# %% [markdown]
# ---
# ## DS1 - SGCharts Dengue Cluster Archive
# **Source:** sgcharts/csv and sgcharts/incorrect_latitude_longitude
#
# **Schema (no header):** cases_in_location, address, latitude, longitude,
# recent_cases_last_2_weeks, cluster_total_cases, cluster_id,
# snapshot_yymmdd, active_cluster_count

# %% [markdown]
# ### 1. Load Dengue Data

# %%
# Read all CSV snapshot files from both folders and combine into one DataFrame
DENGUE_COLS = [
    'cases_in_location', 'address', 'latitude', 'longitude',
    'recent_cases_last_2_weeks', 'cluster_total_cases',
    'cluster_id', 'snapshot_yymmdd', 'active_cluster_count'
]

frames = []
for folder in DENGUE_FOLDERS:
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        date_part = os.path.basename(f).split('-')[0]
        try:
            snap_date = pd.to_datetime(date_part, format='%y%m%d')
        except ValueError:
            continue
        df = pd.read_csv(f, header=None, encoding='utf-8')
        df = df.dropna(how='all')
        if len(df.columns) != 9:
            continue
        df.columns = DENGUE_COLS
        df['snapshot_date'] = snap_date
        frames.append(df)

dengue_raw = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(dengue_raw)} rows across {dengue_raw['snapshot_date'].nunique()} snapshots")
print(f"Date range: {dengue_raw['snapshot_date'].min().date()} -> {dengue_raw['snapshot_date'].max().date()}")

# %% [markdown]
# ### 2. Inspect Dengue Data

# %%
print("=== Shape ===")
print(dengue_raw.shape)
print("\n=== Missing Values ===")
print(dengue_raw.isnull().sum())
print("\n=== Snapshots per Year ===")
print(dengue_raw.groupby(dengue_raw['snapshot_date'].dt.year)['snapshot_date'].nunique())
print("\n=== Sample ===")
print(dengue_raw.head(3).to_string())

# %% [markdown]
# ### 3. Clean Dengue Data

# %%
dengue = dengue_raw.copy()

# Force numeric types on coordinate and case columns
for col in ['latitude', 'longitude', 'cases_in_location',
            'recent_cases_last_2_weeks', 'cluster_total_cases']:
    dengue[col] = pd.to_numeric(dengue[col], errors='coerce')

# Fix the one confirmed bad coordinate:
# upper serangoon crescent block 470a was ~5.2km off in the incorrect_latitude_longitude folder
mask = dengue['address'] == 'upper serangoon crescent (block 470a)'
dengue.loc[mask, 'latitude']  = 1.3792
dengue.loc[mask, 'longitude'] = 103.9017

# Drop placeholder -1 recent_cases (pre-Dec 2013 data artefact)
dengue = dengue[dengue['recent_cases_last_2_weeks'] >= 0]

# Drop rows with missing coordinates
dengue = dengue.dropna(subset=['latitude', 'longitude'])

# Remove duplicates that arise from the two folders overlapping in 2013-2015
dengue = dengue.drop_duplicates(subset=['snapshot_date', 'address', 'cluster_id'])

# Restrict to study window: Mar 2016 - Dec 2019
dengue = dengue[(dengue['snapshot_date'] >= DATE_START) &
                (dengue['snapshot_date'] <= DATE_END)]

# Keep only coordinates within Singapore bounding box
dengue = dengue[dengue['latitude'].between(1.1, 1.5) &
                dengue['longitude'].between(103.6, 104.1)]

# Add ISO year and week — isocalendar year ensures week 1 maps to the correct year
dengue['year']     = dengue['snapshot_date'].dt.isocalendar().year.astype(int)
dengue['iso_week'] = dengue['snapshot_date'].dt.isocalendar().week.astype(int)

# Where multiple snapshots fall in the same ISO week keep only the latest per address
# (each snapshot is point-in-time state so the latest supersedes earlier ones)
dengue = dengue.sort_values('snapshot_date')
dengue = dengue.drop_duplicates(subset=['address', 'year', 'iso_week'], keep='last')

print(f"Clean dengue: {dengue.shape}")
print(f"Date range:   {dengue['snapshot_date'].min().date()} -> {dengue['snapshot_date'].max().date()}")
print(f"Snapshots:    {dengue['snapshot_date'].nunique()}")
print(f"Unique addresses: {dengue['address'].nunique()}")

# %% [markdown]
# ---
# ## DS2 - Rainfall Data
# **2016 source:** SC3021_rainfall - daily totals, one CSV per station per month
#
# **2017-2019 source:** HistoricalRainfallacrossSingapore - 5-minute readings
# aggregated to daily totals on load to keep memory manageable

# %% [markdown]
# ### 4. Load Rainfall 2016

# %%
# Filenames: DAILYDATA_S{id}_{YYYYMM}.csv
# Extract station_id from filename (e.g. S06) to match 2017-2019 station metadata
rain_2016_frames = []
for f in sorted(glob.glob(os.path.join(RAINFALL_2016_FOLDER, "*.csv"))):
    parts      = os.path.basename(f).replace('.csv', '').split('_')
    station_id = parts[1]

    df = pd.read_csv(f, encoding='latin1')
    df.columns = df.columns.str.strip()

    # Rainfall column name has encoding issues in the header - find it by keyword
    rainfall_col = [c for c in df.columns if 'Rainfall' in c and 'Daily' in c][0]
    df = df.rename(columns={rainfall_col: 'daily_rainfall_mm', 'Station': 'station_name'})

    df['date']              = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df['station_id']        = station_id
    df['daily_rainfall_mm'] = pd.to_numeric(df['daily_rainfall_mm'], errors='coerce').fillna(0)

    rain_2016_frames.append(df[['date', 'station_id', 'station_name', 'daily_rainfall_mm']])

rain_2016 = pd.concat(rain_2016_frames, ignore_index=True)
print(f"2016 rainfall: {len(rain_2016)} rows, {rain_2016['station_id'].nunique()} stations")

# %% [markdown]
# ### 5. Load Rainfall 2017-2019

# %%
# Each file has ~5M rows of 5-minute readings
# Aggregate to daily totals immediately on load to keep memory low
# Station coordinates are embedded - extract a lookup table here too
rain_hist_frames      = []
station_lookup_frames = []

for f in RAINFALL_HIST_FILES:
    df = pd.read_csv(f, usecols=[
        'date', 'station_id', 'station_name',
        'location_longitude', 'location_latitude', 'reading_value'
    ])
    df['date']          = pd.to_datetime(df['date'])
    df['reading_value'] = pd.to_numeric(df['reading_value'], errors='coerce').fillna(0)

    # Save unique station coordinates for the spatial lookup
    station_lookup_frames.append(
        df[['station_id', 'station_name', 'location_longitude', 'location_latitude']]
        .drop_duplicates('station_id')
    )

    # Sum 5-minute readings into daily totals per station
    daily = df.groupby(['date', 'station_id', 'station_name'])['reading_value'].sum().reset_index()
    daily.rename(columns={'reading_value': 'daily_rainfall_mm'}, inplace=True)
    rain_hist_frames.append(daily)

rain_hist      = pd.concat(rain_hist_frames, ignore_index=True)
station_lookup = (pd.concat(station_lookup_frames)
                  .drop_duplicates('station_id')
                  .reset_index(drop=True))

print(f"2017-2019 rainfall: {len(rain_hist)} rows, {rain_hist['station_id'].nunique()} stations")
print(f"Station coordinate lookup: {len(station_lookup)} stations")

# %% [markdown]
# ### 6. Combine & Inspect Rainfall

# %%
# Combine 2016 and 2017-2019 into one daily rainfall DataFrame
rain_all = pd.concat(
    [rain_2016, rain_hist[['date', 'station_id', 'station_name', 'daily_rainfall_mm']]],
    ignore_index=True
)

# Filter to study window
rain_all = rain_all[(rain_all['date'] >= DATE_START) & (rain_all['date'] <= DATE_END)]

# Drop station S82 - missing 9 out of 10 months in 2016 making it unreliable
rain_all = rain_all[rain_all['station_id'] != 'S82']

print("=== Daily Rainfall Summary ===")
print(f"Rows:       {len(rain_all)}")
print(f"Stations:   {rain_all['station_id'].nunique()}")
print(f"Date range: {rain_all['date'].min().date()} -> {rain_all['date'].max().date()}")
print("\n=== Missing Values ===")
print(rain_all.isnull().sum())

# %% [markdown]
# ### 7. Assign Each URA Subzone to its Nearest Rainfall Station
# Rainfall is spatially variable so each subzone gets its own rainfall value
# based on the nearest physical station using a KDTree nearest-neighbour search.

# %%
# Load URA subzones
ura_gdf = gpd.read_file(URA_FILE)
if ura_gdf.crs is None or ura_gdf.crs.to_epsg() != 4326:
    ura_gdf = ura_gdf.set_crs(epsg=4326)

# Compute subzone centroid coordinates
ura_gdf['centroid'] = ura_gdf.geometry.centroid
subzone_coords = np.array([[p.x, p.y] for p in ura_gdf['centroid']])

# Build station coordinate array from those that appear in rain_all
station_coords_df = station_lookup[
    station_lookup['station_id'].isin(rain_all['station_id'])
].dropna(subset=['location_longitude', 'location_latitude']).reset_index(drop=True)

station_coords = station_coords_df[['location_longitude', 'location_latitude']].values

# Find nearest station for every subzone centroid
tree = KDTree(station_coords)
_, idx = tree.query(subzone_coords)
ura_gdf['nearest_station_id'] = station_coords_df.iloc[idx]['station_id'].values

print(f"Subzones assigned: {len(ura_gdf)}")
print(f"Unique stations used: {ura_gdf['nearest_station_id'].nunique()}")
print(ura_gdf[['SUBZONE_N', 'PLN_AREA_N', 'nearest_station_id']].head(8).to_string(index=False))

# %% [markdown]
# ### 8. Aggregate Rainfall to Weekly per Subzone

# %%
# Step 1: sum daily rainfall to weekly totals per station
rain_all['year']     = rain_all['date'].dt.isocalendar().year.astype(int)
rain_all['iso_week'] = rain_all['date'].dt.isocalendar().week.astype(int)

rain_weekly_station = (
    rain_all
    .groupby(['year', 'iso_week', 'station_id'])['daily_rainfall_mm']
    .sum()
    .reset_index()
    .rename(columns={'daily_rainfall_mm': 'weekly_rainfall_mm'})
)

# Step 2: map each subzone to its nearest station then pull in weekly rainfall
subzone_station_map = ura_gdf[['SUBZONE_N', 'PLN_AREA_N', 'REGION_N', 'nearest_station_id']].copy()

rain_weekly_subzone = subzone_station_map.merge(
    rain_weekly_station,
    left_on='nearest_station_id',
    right_on='station_id',
    how='left'
).drop(columns='station_id')

print(f"Weekly rainfall per subzone shape: {rain_weekly_subzone.shape}")
print(f"Subzones with rainfall data: {rain_weekly_subzone['SUBZONE_N'].nunique()}")
print(rain_weekly_subzone.head(5).to_string(index=False))

# %% [markdown]
# ---
# ## DS3 - Temperature & Humidity (Island-wide)
# **Source:** Historical 4-day Weather Forecast 2016-2019
#
# Temperature and humidity are treated as consistent across Singapore -
# one value per week island-wide. We retain high, low, average and range
# to capture both central tendency and daily variability.

# %% [markdown]
# ### 9. Load Temperature & Humidity Data

# %%
weather_frames = []
for f in WEATHER_FILES:
    df = pd.read_csv(f)
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    weather_frames.append(df[[
        'forecast_date',
        'temperature_high', 'temperature_low',
        'relative_humidity_high', 'relative_humidity_low'
    ]])

weather_raw = pd.concat(weather_frames, ignore_index=True)
weather_raw = weather_raw[(weather_raw['forecast_date'] >= DATE_START) &
                           (weather_raw['forecast_date'] <= DATE_END)]

# Multiple forecast entries may cover the same date - take the mean
weather_daily = weather_raw.groupby('forecast_date').mean().reset_index()

print(f"Daily weather rows: {len(weather_daily)}")
print(f"Date range: {weather_daily['forecast_date'].min().date()} -> {weather_daily['forecast_date'].max().date()}")

# %% [markdown]
# ### 10. Inspect Weather Data

# %%
print("=== Missing Values ===")
print(weather_daily.isnull().sum())
print("\n=== Distributions ===")
print(weather_daily.describe().round(2))

# %% [markdown]
# ### 11. Aggregate Weather to Weekly (Island-wide)

# %%
weather_daily['year']     = weather_daily['forecast_date'].dt.isocalendar().year.astype(int)
weather_daily['iso_week'] = weather_daily['forecast_date'].dt.isocalendar().week.astype(int)

weather_weekly = weather_daily.groupby(['year', 'iso_week']).agg(
    temp_high      = ('temperature_high',       'mean'),
    temp_low       = ('temperature_low',        'mean'),
    humidity_high  = ('relative_humidity_high', 'mean'),
    humidity_low   = ('relative_humidity_low',  'mean'),
).reset_index()

# Derive averages and ranges from the weekly highs and lows
weather_weekly['temp_avg']       = (weather_weekly['temp_high'] + weather_weekly['temp_low']) / 2
weather_weekly['temp_range']     = weather_weekly['temp_high'] - weather_weekly['temp_low']
weather_weekly['humidity_avg']   = (weather_weekly['humidity_high'] + weather_weekly['humidity_low']) / 2
weather_weekly['humidity_range'] = weather_weekly['humidity_high'] - weather_weekly['humidity_low']

print(f"Weekly weather shape: {weather_weekly.shape}")
print(weather_weekly.head(5).to_string(index=False))

# %% [markdown]
# ---
# ## DS4 - URA Master Plan 2019 Subzone Boundaries
# **Source:** ura_subzones.geojson (downloaded from data.gov.sg)
#
# 332 subzones across 55 planning areas and 5 regions. Used as the spatial
# unit for both dengue aggregation and rainfall station assignment.

# %% [markdown]
# ### 12. Inspect URA Subzones

# %%
# ura_gdf was already loaded in Section 7
print(f"Subzones:        {len(ura_gdf)}")
print(f"Planning areas:  {ura_gdf['PLN_AREA_N'].nunique()}")
print(f"Regions:         {ura_gdf['REGION_N'].nunique()}")
print(f"CRS:             {ura_gdf.crs}")
print(f"Null geometries: {ura_gdf.geometry.isnull().sum()}")

fig, ax = plt.subplots(figsize=(10, 8))
ura_gdf.plot(ax=ax, color='lightblue', edgecolor='grey', linewidth=0.3, alpha=0.7)
plt.title("Singapore URA Subzones - Master Plan 2019")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Data Preparation

# %% [markdown]
# ### 13. Spatial Join - Dengue Points to URA Subzones

# %%
# Convert dengue lat/lon to GeoDataFrame and assign each point to a subzone
dengue_gdf = gpd.GeoDataFrame(
    dengue,
    geometry=gpd.points_from_xy(dengue['longitude'], dengue['latitude']),
    crs="EPSG:4326"
)

dengue_gdf = gpd.sjoin(
    dengue_gdf,
    ura_gdf[['SUBZONE_N', 'PLN_AREA_N', 'REGION_N', 'geometry']],
    how='left',
    predicate='within'
)

unmatched = dengue_gdf['SUBZONE_N'].isnull().sum()
print(f"Matched:  {len(dengue_gdf) - unmatched} / {len(dengue_gdf)}")
print(f"Dropped:  {unmatched} ({100 * unmatched / len(dengue_gdf):.1f}%)")

dengue_gdf = dengue_gdf.dropna(subset=['SUBZONE_N'])

fig, ax = plt.subplots(figsize=(10, 8))
ura_gdf.plot(ax=ax, color='lightyellow', edgecolor='grey', linewidth=0.3, alpha=0.6)
dengue_gdf.plot(ax=ax, color='red', markersize=2, alpha=0.3)
plt.title("Dengue Cluster Locations (Mar 2016 - Dec 2019)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 14. Aggregate Dengue to Weekly per Subzone

# %%
# One row per subzone per week - sum cases_in_location across all addresses in the subzone
dengue_weekly = (
    dengue_gdf
    .groupby(['year', 'iso_week', 'SUBZONE_N', 'PLN_AREA_N', 'REGION_N'])
    .agg(weekly_cases=('cases_in_location', 'sum'))
    .reset_index()
)

print(f"Weekly dengue shape:        {dengue_weekly.shape}")
print(f"Subzones with cases:        {dengue_weekly['SUBZONE_N'].nunique()}")
print(f"Year-weeks with cases:      {dengue_weekly[['year', 'iso_week']].drop_duplicates().shape[0]}")
print(dengue_weekly.head(5).to_string(index=False))

# %% [markdown]
# ---
# ## Feature Engineering
# The Aedes mosquito breeding cycle plus dengue viral incubation is approximately
# 14 days. We shift rainfall and weather features back 2 weeks so conditions at
# time T are used to predict dengue cases at time T+2.

# %% [markdown]
# ### 15. Apply 2-Week Lag to Rainfall (per Subzone)

# %%
# Sort by subzone then time so the shift stays within each subzone independently
rain_weekly_subzone = rain_weekly_subzone.sort_values(['SUBZONE_N', 'year', 'iso_week'])
rain_weekly_subzone['rainfall_lag2'] = (
    rain_weekly_subzone.groupby('SUBZONE_N')['weekly_rainfall_mm'].shift(2)
)

print("Rainfall columns:", ['weekly_rainfall_mm', 'rainfall_lag2'])
print(rain_weekly_subzone[
    ['SUBZONE_N', 'year', 'iso_week', 'weekly_rainfall_mm', 'rainfall_lag2']
].head(8).to_string(index=False))

# %% [markdown]
# ### 16. Apply 2-Week Lag to Temperature & Humidity (Island-wide)

# %%
weather_weekly = weather_weekly.sort_values(['year', 'iso_week'])

for col in ['temp_avg', 'temp_high', 'temp_low', 'temp_range',
            'humidity_avg', 'humidity_high', 'humidity_low', 'humidity_range']:
    weather_weekly[f'{col}_lag2'] = weather_weekly[col].shift(2)

lag_cols = [c for c in weather_weekly.columns if 'lag2' in c]
print("Lagged weather columns:", lag_cols)
print(weather_weekly[
    ['year', 'iso_week', 'temp_avg', 'temp_avg_lag2', 'humidity_avg', 'humidity_avg_lag2']
].head(5).to_string(index=False))

# %% [markdown]
# ---
# ## Merge - Final Analytical Table
# **Structure:** one row per subzone x week
#
# Rainfall varies by subzone (nearest-station assignment).
# Temperature and humidity are island-wide - the same weekly value is
# broadcast across all subzones for that week.

# %% [markdown]
# ### 17. Merge Dengue + Rainfall + Weather

# %%
# Step 1: dengue weekly LEFT JOIN rainfall weekly on subzone + year + week
merged = dengue_weekly.merge(
    rain_weekly_subzone[['SUBZONE_N', 'year', 'iso_week', 'weekly_rainfall_mm', 'rainfall_lag2']],
    on=['SUBZONE_N', 'year', 'iso_week'],
    how='left'
)

# Step 2: JOIN island-wide weather on year + week
merged = merged.merge(
    weather_weekly,
    on=['year', 'iso_week'],
    how='left'
)

print(f"Final merged shape: {merged.shape}")
print(f"\nColumns:\n{merged.columns.tolist()}")
print(f"\nMissing values:\n{merged.isnull().sum()}")
print(f"\nSample:\n{merged.head(5).to_string(index=False)}")

# %% [markdown]
# ### 18. Final Table Summary

# %%
print("=== Final Analytical Table ===")
print(f"Total rows:               {len(merged)}")
print(f"Subzones covered:         {merged['SUBZONE_N'].nunique()}")
print(f"Year-weeks covered:       {merged[['year', 'iso_week']].drop_duplicates().shape[0]}")
print(f"Years:                    {sorted(merged['year'].unique())}")
print(f"Total weekly cases:       {merged['weekly_cases'].sum():.0f}")
print(f"Avg cases/subzone-week:   {merged['weekly_cases'].mean():.2f}")

print("\n=== Descriptive Statistics ===")
print(merged[[
    'weekly_cases', 'weekly_rainfall_mm', 'rainfall_lag2',
    'temp_avg', 'temp_range', 'humidity_avg', 'humidity_range'
]].describe().round(2))

# %% [markdown]
# ---
# ## Visualisation

# %% [markdown]
# ### 19. Dengue Cases vs Lagged Rainfall, Temperature and Humidity

# %%
# Aggregate to island-wide weekly totals for time-series plots
island_weekly = merged.groupby(['year', 'iso_week']).agg(
    total_cases       = ('weekly_cases',   'sum'),
    avg_rainfall_lag2 = ('rainfall_lag2',  'mean'),
    temp_avg          = ('temp_avg',       'first'),
    temp_range        = ('temp_range',     'first'),
    humidity_avg      = ('humidity_avg',   'first'),
    humidity_range    = ('humidity_range', 'first'),
).reset_index()

island_weekly['label'] = (island_weekly['year'].astype(str) + '-W' +
                          island_weekly['iso_week'].astype(str).str.zfill(2))
x = range(len(island_weekly))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Cases vs 2-week lagged rainfall
ax1, ax2 = axes[0], axes[0].twinx()
ax1.plot(x, island_weekly['total_cases'],       color='red',  label='Weekly Cases')
ax2.plot(x, island_weekly['avg_rainfall_lag2'], color='blue', label='Rainfall Lag-2 (mm)', alpha=0.6)
ax1.set_ylabel('Dengue Cases', color='red')
ax2.set_ylabel('Rainfall mm (2-week lag)', color='blue')
axes[0].set_title('Weekly Dengue Cases vs 2-Week Lagged Rainfall')
axes[0].set_xticks(list(x)[::8])
axes[0].set_xticklabels(island_weekly['label'].iloc[::8], rotation=45, ha='right', fontsize=7)

# Plot 2: Cases vs temperature average and range
ax3, ax4 = axes[1], axes[1].twinx()
ax3.plot(x, island_weekly['total_cases'], color='red',    label='Weekly Cases')
ax4.plot(x, island_weekly['temp_avg'],    color='orange', label='Temp Avg', alpha=0.7)
ax4.fill_between(x,
    island_weekly['temp_avg'] - island_weekly['temp_range'] / 2,
    island_weekly['temp_avg'] + island_weekly['temp_range'] / 2,
    alpha=0.15, color='orange', label='Temp Range')
ax3.set_ylabel('Dengue Cases', color='red')
ax4.set_ylabel('Temperature C', color='orange')
axes[1].set_title('Weekly Dengue Cases vs Temperature (Avg & Range)')
axes[1].set_xticks(list(x)[::8])
axes[1].set_xticklabels(island_weekly['label'].iloc[::8], rotation=45, ha='right', fontsize=7)

# Plot 3: Cases vs humidity average and range
ax5, ax6 = axes[2], axes[2].twinx()
ax5.plot(x, island_weekly['total_cases'],  color='red',   label='Weekly Cases')
ax6.plot(x, island_weekly['humidity_avg'], color='green', label='Humidity Avg', alpha=0.7)
ax6.fill_between(x,
    island_weekly['humidity_avg'] - island_weekly['humidity_range'] / 2,
    island_weekly['humidity_avg'] + island_weekly['humidity_range'] / 2,
    alpha=0.15, color='green', label='Humidity Range')
ax5.set_ylabel('Dengue Cases', color='red')
ax6.set_ylabel('Relative Humidity %', color='green')
axes[2].set_title('Weekly Dengue Cases vs Humidity (Avg & Range)')
axes[2].set_xticks(list(x)[::8])
axes[2].set_xticklabels(island_weekly['label'].iloc[::8], rotation=45, ha='right', fontsize=7)

plt.tight_layout()
plt.savefig('dengue_weather_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## Data Storage

# %% [markdown]
# ### 20. Store Final Table in SQLite

# %%
if os.path.exists('dengue_weather_sg.db'):
    os.remove('dengue_weather_sg.db')

conn = sqlite3.connect('dengue_weather_sg.db')

merged.to_sql('dengue_weather_weekly', conn, if_exists='replace', index=False)
ura_gdf.drop(columns=['geometry', 'centroid']).to_sql('ura_planning_areas', conn, if_exists='replace', index=False)

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Tables saved:", tables['name'].tolist())
for t in tables['name']:
    n = pd.read_sql(f"SELECT COUNT(*) as n FROM {t}", conn).iloc[0]['n']
    print(f"  {t}: {n} rows")

conn.close()

# %% [markdown]
# ---
# ## NN Preparation
# Normalise the analysis dataset so it is ready for the LSTM + Transformer model.
#
# **Strategy**
# - Rainfall features: `log1p` transform first (right-skewed, zero-inflated), then `StandardScaler`
# - Temperature and humidity features: `StandardScaler` directly (near-normal, narrow range)
# - Lag features: scaled together with their base features — same transform, same scaler
# - Target (`weekly_cases`): `log1p` transform only (kept separate from feature matrix)
# - Scaler fitted on **training years (2016–2018) only**; applied to all years
#
# Output saved to `dengue_nn_ready.csv`

# %% [markdown]
# ### 21. Load Analysis Dataset & Define Feature Groups

# %%
from sklearn.preprocessing import StandardScaler
import joblib

df_nn = pd.read_csv('dengue_analysis_complete.csv')

# Sort consistently
df_nn = df_nn.sort_values(['year', 'iso_week', 'SUBZONE_N']).reset_index(drop=True)

# Feature groups
RAIN_COLS = [c for c in df_nn.columns if 'rainfall' in c]
TEMP_COLS = [c for c in df_nn.columns if 'temp' in c]
HUM_COLS  = [c for c in df_nn.columns if 'humidity' in c]
CASE_COLS = ['weekly_cases'] + [c for c in df_nn.columns if 'cases_lag' in c]

print("Rainfall features :", RAIN_COLS)
print("Temperature features:", TEMP_COLS)
print("Humidity features  :", HUM_COLS)
print("Case columns       :", CASE_COLS)

# %% [markdown]
# ### 22. Apply log1p to Rainfall and Cases, then StandardScaler

# %%
df_scaled = df_nn.copy()

# --- log1p transforms ---
for col in RAIN_COLS + CASE_COLS:
    df_scaled[col] = np.log1p(df_scaled[col])

# --- Train/test split: train on 2016-2018, test on 2019 ---
train_mask = df_scaled['year'] < 2019

# --- Fit scalers on training data only ---
rain_scaler = StandardScaler()
temp_scaler = StandardScaler()
hum_scaler  = StandardScaler()
case_scaler = StandardScaler()

rain_scaler.fit(df_scaled.loc[train_mask, RAIN_COLS])
temp_scaler.fit(df_scaled.loc[train_mask, TEMP_COLS])
hum_scaler.fit(df_scaled.loc[train_mask, HUM_COLS])
case_scaler.fit(df_scaled.loc[train_mask, CASE_COLS])

# --- Apply to full dataset ---
df_scaled[RAIN_COLS] = rain_scaler.transform(df_scaled[RAIN_COLS])
df_scaled[TEMP_COLS] = temp_scaler.transform(df_scaled[TEMP_COLS])
df_scaled[HUM_COLS]  = hum_scaler.transform(df_scaled[HUM_COLS])
df_scaled[CASE_COLS] = case_scaler.transform(df_scaled[CASE_COLS])

# Save scalers for inverse-transform at inference time
joblib.dump(rain_scaler, 'scaler_rain.pkl')
joblib.dump(temp_scaler, 'scaler_temp.pkl')
joblib.dump(hum_scaler,  'scaler_hum.pkl')
joblib.dump(case_scaler, 'scaler_cases.pkl')

print("Scalers saved: scaler_rain.pkl, scaler_temp.pkl, scaler_hum.pkl, scaler_cases.pkl")

# %% [markdown]
# ### 23. Verify Distributions Post-Scaling

# %%
feature_cols = RAIN_COLS + TEMP_COLS + HUM_COLS + CASE_COLS
desc = df_scaled[feature_cols].describe().loc[['mean','std','min','max']].round(3)
print(desc.T.to_string())

# Quick sanity: means should be ~0, stds ~1 on the training slice
train_desc = df_scaled.loc[train_mask, feature_cols].describe().loc[['mean','std']].round(3)
print("\nTraining split (should be mean≈0, std≈1):")
print(train_desc.T.to_string())

# %% [markdown]
# ### 24. Save NN-Ready Dataset

# %%
out_nn = 'dengue_nn_ready.csv'
df_scaled.to_csv(out_nn, index=False)

print(f"Saved: {out_nn}")
print(f"Shape: {df_scaled.shape}")
print(f"Columns: {df_scaled.columns.tolist()}")
print(f"\nTrain rows (2016-2018): {train_mask.sum()}")
print(f"Test rows  (2019):      {(~train_mask).sum()}")
print("\nSaved to dengue_weather_sg.db")

# %% [markdown]
# ---
# ## Neural Network Models
# Two separate models trained and evaluated on the same dataset.
# The better-performing model will be selected for final predictions.
#
# **Config**
# - `SEQ_LEN = 4` weeks of input history
# - `FORECAST_HORIZON = 2` weeks ahead to predict
# - Features: base weather/rainfall + `area_km2` (no lag columns — model learns temporal dependencies from the sequence)
# - Train: 2016–2018 · Test: 2019
# - Loss: weighted MSE (non-zero case weeks upweighted ×5)

# %% [markdown]
# ### 25. Add URA Subzone Area Feature

# %%
import geopandas as gpd
import joblib

# Load URA subzones, reproject to Singapore metric CRS (EPSG:3414) for accurate area
ura = gpd.read_file('ura_subzones.geojson').to_crs(epsg=3414)
ura['area_km2'] = ura.geometry.area / 1e6   # m² → km²

# Normalise area so it sits on the same scale as other features
from sklearn.preprocessing import StandardScaler as _SS
_area_scaler = _SS()
ura['area_km2'] = _area_scaler.fit_transform(ura[['area_km2']])

# Build subzone → area lookup (uppercase to match dengue dataset)
area_lookup = ura.set_index('SUBZONE_N')['area_km2'].to_dict()

# Merge into nn-ready dataset
df_nn2 = pd.read_csv('dengue_nn_ready.csv')
df_nn2['area_km2'] = df_nn2['SUBZONE_N'].map(area_lookup)
print(f"Subzones with area assigned: {df_nn2['area_km2'].notna().sum()} / {len(df_nn2)}")
print(f"Unmatched subzones: {df_nn2[df_nn2['area_km2'].isna()]['SUBZONE_N'].unique()}")

df_nn2['area_km2'] = df_nn2['area_km2'].fillna(0)
df_nn2.to_csv('dengue_nn_ready.csv', index=False)
print("dengue_nn_ready.csv updated with area_km2")

# %% [markdown]
# ### 26. Build Sequence Dataset

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

SEQ_LEN          = 4
FORECAST_HORIZON = 2
BATCH_SIZE       = 64
EPOCHS           = 100
PATIENCE         = 10
WEIGHT_NONZERO   = 5.0   # loss weight for weeks with actual cases

# Feature columns for sequence input (drop all lag columns — model learns from sequence)
LAG_COLS  = [c for c in df_nn2.columns if '_lag' in c]
ID_COLS   = ['year', 'iso_week', 'SUBZONE_N']
TARGET_COL = 'weekly_cases'

FEAT_COLS = [c for c in df_nn2.columns
             if c not in ID_COLS + LAG_COLS + [TARGET_COL]]

print(f"Input features ({len(FEAT_COLS)}): {FEAT_COLS}")
print(f"SEQ_LEN={SEQ_LEN}, HORIZON={FORECAST_HORIZON}")

def build_sequences(df, feat_cols, target_col, seq_len, horizon):
    """
    For each subzone build sliding-window sequences.
    Returns X (n, seq_len, features), y (n,), year of target week.
    """
    X_list, y_list, yr_list = [], [], []
    for zone, grp in df.groupby('SUBZONE_N'):
        grp = grp.sort_values(['year', 'iso_week']).reset_index(drop=True)
        feats  = grp[feat_cols].values.astype(np.float32)
        target = grp[target_col].values.astype(np.float32)
        years  = grp['year'].values
        n = len(grp)
        for i in range(n - seq_len - horizon + 1):
            t_idx = i + seq_len - 1 + horizon   # target row index
            X_list.append(feats[i : i + seq_len])
            y_list.append(target[t_idx])
            yr_list.append(years[t_idx])
    return np.array(X_list), np.array(y_list), np.array(yr_list)

X_all, y_all, yr_all = build_sequences(df_nn2, FEAT_COLS, TARGET_COL, SEQ_LEN, FORECAST_HORIZON)

# Hard boundary split — no leakage across train/test
train_idx = np.where(yr_all < 2019)[0]
test_idx  = np.where(yr_all >= 2019)[0]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

# Validation: last 15% of training sequences
val_cut  = int(len(X_train) * 0.85)
X_val,   y_val   = X_train[val_cut:], y_train[val_cut:]
X_train, y_train = X_train[:val_cut],  y_train[:val_cut]

print(f"\nSequences — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
print(f"Input shape: {X_train.shape}  (batch, seq_len, features)")

class DengueDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(DengueDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(DengueDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(DengueDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

def weighted_mse(pred, target, weight=WEIGHT_NONZERO):
    """Upweight non-zero case weeks to combat sparse-zero bias."""
    w = torch.where(target > 0, torch.full_like(target, weight), torch.ones_like(target))
    return (w * (pred - target) ** 2).mean()

def run_training(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_val, best_state, counter = float('inf'), None, 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        t_loss = sum(
            weighted_mse(model(X), y).item()
            for X, y in train_loader
        ) / len(train_loader)

        model.eval()
        with torch.no_grad():
            v_loss = sum(
                weighted_mse(model(X), y).item()
                for X, y in val_loader
            ) / len(val_loader)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if v_loss < best_val:
            best_val, best_state, counter = v_loss, copy.deepcopy(model.state_dict()), 0
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train {t_loss:.4f} | val {v_loss:.4f}")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses

def evaluate(model, loader, case_scaler, label=''):
    """Return MAE, RMSE, R² on actual (inverse-transformed) case counts."""
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X).numpy())
            actuals.append(y.numpy())
    preds   = np.concatenate(preds).reshape(-1, 1)
    actuals = np.concatenate(actuals).reshape(-1, 1)

    # Inverse transform: StandardScaler → expm1
    preds_real   = np.expm1(case_scaler.inverse_transform(preds)).clip(0)
    actuals_real = np.expm1(case_scaler.inverse_transform(actuals)).clip(0)

    mae  = np.mean(np.abs(preds_real - actuals_real))
    rmse = np.sqrt(np.mean((preds_real - actuals_real) ** 2))
    ss_res = np.sum((actuals_real - preds_real) ** 2)
    ss_tot = np.sum((actuals_real - actuals_real.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f"{label:20s} MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")
    return preds_real.flatten(), actuals_real.flatten(), {'mae': mae, 'rmse': rmse, 'r2': r2}

case_scaler = joblib.load('scaler_cases.pkl')
N_FEATURES  = X_train.shape[2]

# %% [markdown]
# ---
# ## Model A — LSTM

# %% [markdown]
# ### 27. LSTM: Definition & Training

# %%
class DengueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)

print("=== LSTM Training ===")
lstm_model = DengueLSTM(input_size=N_FEATURES)
lstm_model, lstm_train_loss, lstm_val_loss = run_training(lstm_model, train_loader, val_loader)

print("\nLSTM Evaluation:")
lstm_preds, lstm_actuals, lstm_metrics = evaluate(lstm_model, test_loader, case_scaler, label='LSTM (test 2019)')
torch.save(lstm_model.state_dict(), 'lstm_model.pt')
print("Saved: lstm_model.pt")

# %% [markdown]
# ---
# ## Model B — Transformer

# %% [markdown]
# ### 28. Transformer: Definition & Training

# %%
class DengueTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj    = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.proj(x) + self.pos_enc
        x = self.transformer(x)
        return self.head(x[:, -1, :]).squeeze(-1)

print("=== Transformer Training ===")
tf_model = DengueTransformer(input_size=N_FEATURES)
tf_model, tf_train_loss, tf_val_loss = run_training(tf_model, train_loader, val_loader)

print("\nTransformer Evaluation:")
tf_preds, tf_actuals, tf_metrics = evaluate(tf_model, test_loader, case_scaler, label='Transformer (test 2019)')
torch.save(tf_model.state_dict(), 'transformer_model.pt')
print("Saved: transformer_model.pt")

# %% [markdown]
# ---
# ## Evaluation & Comparison

# %% [markdown]
# ### 29. Side-by-Side Comparison

# %%
print("\n" + "="*55)
print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-"*55)
print(f"{'LSTM':<20} {lstm_metrics['mae']:>8.2f} {lstm_metrics['rmse']:>8.2f} {lstm_metrics['r2']:>8.3f}")
print(f"{'Transformer':<20} {tf_metrics['mae']:>8.2f} {tf_metrics['rmse']:>8.2f} {tf_metrics['r2']:>8.3f}")
print("="*55)

winner = 'LSTM' if lstm_metrics['rmse'] < tf_metrics['rmse'] else 'Transformer'
print(f"\nLower RMSE: {winner}")

# ── Training curves ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(lstm_train_loss, label='Train')
axes[0].plot(lstm_val_loss,   label='Val')
axes[0].set_title('LSTM — Training Curve')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Weighted MSE')
axes[0].legend()

axes[1].plot(tf_train_loss, label='Train')
axes[1].plot(tf_val_loss,   label='Val')
axes[1].set_title('Transformer — Training Curve')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Weighted MSE')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Predicted vs Actual (island-wide weekly sum) ───────────────────────────────
# Reconstruct weekly island totals from per-sequence predictions
# Use test set order (sequences are already time-ordered within each subzone)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(lstm_actuals,  label='Actual',      color='black', linewidth=1.5)
ax.plot(lstm_preds,    label='LSTM',        color='steelblue',  alpha=0.8)
ax.plot(tf_preds,      label='Transformer', color='darkorange', alpha=0.8)
ax.set_title('Predicted vs Actual Cases — 2019 Test Set (per sequence)')
ax.set_xlabel('Sequence index'); ax.set_ylabel('Weekly cases (actual count)')
ax.legend()
plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()
