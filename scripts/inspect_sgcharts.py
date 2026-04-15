#!/usr/bin/env python
# coding: utf-8
# Load all sgcharts .csv files from both folders, combine and save to dengue_all.csv

import os
import glob
import pandas as pd

FOLDERS = [
    "sgcharts/incorrect_latitude_longitude",
    "sgcharts/csv"
]

SGCHARTS_COLS = [
    'cases_in_location', 'address', 'latitude', 'longitude',
    'recent_cases_last_2_weeks', 'cluster_total_cases',
    'cluster_id', 'snapshot_yymmdd', 'active_cluster_count'
]

frames = []
skipped = []

for folder in FOLDERS:
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    print(f"\n{folder}: {len(csv_files)} files found")

    loaded = 0
    for f in csv_files:
        basename  = os.path.splitext(os.path.basename(f))[0]
        date_part = basename.split('-')[0]

        try:
            snap_date = pd.to_datetime(date_part, format='%y%m%d')
        except ValueError:
            skipped.append((basename, 'unparseable date'))
            continue

        try:
            try:
                df = pd.read_csv(f, header=None, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(f, header=None, encoding='latin1')

            df = df.dropna(how='all')
            ncols = len(df.columns)

            if ncols != 9:
                skipped.append((basename, f'{ncols} cols (expected 9)'))
                continue

            df.columns = SGCHARTS_COLS
            for col in ['latitude', 'longitude', 'cases_in_location',
                        'recent_cases_last_2_weeks', 'cluster_total_cases']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['snapshot_date'] = snap_date
            df['source_folder'] = os.path.basename(folder)
            frames.append(df)
            loaded += 1

        except Exception as e:
            skipped.append((basename, str(e)[:80]))

    print(f"  Loaded: {loaded}")

# ── Combine ────────────────────────────────────────────────────────────────────

print(f"\nSkipped: {len(skipped)} files")
for s in skipped:
    print(f"  - {s[0]}: {s[1]}")

dengue_all = pd.concat(frames, ignore_index=True).sort_values('snapshot_date')

# Drop duplicates (same address + same snapshot date appearing in both folders)
before = len(dengue_all)
dengue_all = dengue_all.drop_duplicates(subset=['snapshot_date', 'address', 'cluster_id'])
after = len(dengue_all)
print(f"\nDuplicate rows removed: {before - after}")

# Validate coordinates within Singapore bounding box
valid = (
    dengue_all['latitude'].between(1.1, 1.5) &
    dengue_all['longitude'].between(103.6, 104.1)
)
n_dropped = (~valid).sum()
if n_dropped > 0:
    print(f"Rows outside Singapore bounds dropped: {n_dropped}")
    dengue_all = dengue_all[valid]

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("COMBINED DENGUE DATASET")
print("="*60)
print(f"Total rows:       {len(dengue_all)}")
print(f"Date range:       {dengue_all['snapshot_date'].min().date()} → {dengue_all['snapshot_date'].max().date()}")
print(f"Unique snapshots: {dengue_all['snapshot_date'].nunique()}")
print(f"Unique clusters:  {dengue_all['cluster_id'].nunique()}")
print(f"Years covered:    {sorted(dengue_all['snapshot_date'].dt.year.unique())}")

print("\nSnapshots per year:")
print(dengue_all.groupby(dengue_all['snapshot_date'].dt.year)['snapshot_date'].nunique().to_string())

# Check gaps > 30 days
dates = dengue_all['snapshot_date'].drop_duplicates().sort_values().reset_index(drop=True)
gaps = dates.diff().dt.days
big_gaps = gaps[gaps > 30]
if len(big_gaps):
    print(f"\nGaps > 30 days ({len(big_gaps)} found):")
    for idx in big_gaps.index:
        print(f"  {int(big_gaps[idx])} days before {dates[idx].date()}")

print("\nSample rows:")
print(dengue_all.head(5).to_string())

# ── Save ───────────────────────────────────────────────────────────────────────

out = "dengue_all.csv"
dengue_all.to_csv(out, index=False)
print(f"\nSaved to {out}")
