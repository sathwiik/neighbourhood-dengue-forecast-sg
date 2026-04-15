#!/usr/bin/env python
# coding: utf-8
"""
Single-stream Transformer — Fold 2 (test 2019) only.

Sequence: 8 steps = 6 historical + 2 forecast
  Steps 1-6  (historical): weather(8) + sg_national(1) + local_cases(1)
                            + forecast_flag=0 + sin + cos  →  13 dims
  Steps 7-8  (forecast):   weather(8) + sg=0            + cases=0
                            + forecast_flag=1 + sin + cos  →  13 dims

Historical steps use sliding-window over consecutive active rows (same
approach that produced 81.2% precision before).
Forecast-step weather is looked up from the full calendar grid for the
2 weeks ending at the target: [last_hist_wk+1, target_wk].

sg_total_cases: log1p / log1p(1792) — full bulletin max, never OOD.
"""

import os, warnings
os.chdir('/Users/sathwiiksai/Documents/DSFLab')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy

# ── Config ────────────────────────────────────────────────────────────────────
HIST_STEPS = 6
FORE_STEPS = 2
TOT_STEPS  = HIST_STEPS + FORE_STEPS   # 8
HORIZON    = 2
BS         = 64
EPOCHS     = 200
PATIENCE   = 20
LR         = 5e-4
WD         = 1e-4

WEATHER_FEATS = [
    'weekly_rainfall_mm', 'rainfall_mean_daily', 'rainfall_sd_daily',
    'temp_avg', 'temp_range', 'humidity_avg', 'humidity_range', 'area_km2'
]
N_FEAT = len(WEATHER_FEATS) + 4   # + sg(1) + cases(1) + flag(1) + sin(1) + cos(1) - wait
# weather(8) + sg(1) + local_cases(1) + forecast_flag(1) + sin(1) + cos(1) = 13
N_FEAT = len(WEATHER_FEATS) + 5   # 13

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading...")
df_full   = pd.read_csv('dengue_analysis_complete.csv')
df_full['week_idx'] = df_full['year'] * 52 + df_full['iso_week']
full_lookup = df_full.set_index(['SUBZONE_N', 'week_idx'])

df_active = df_full[df_full['weekly_cases'] > 0].copy()
print(f"  Full grid: {len(df_full)} rows  |  Active: {len(df_active)} rows")

# National dengue bulletin
bulletin  = pd.read_csv('WeeklyInfectiousDiseaseBulletinCases.csv')
dengue_sg = (bulletin[bulletin['disease'].str.contains('Dengue', case=False)]
             .groupby('epi_week')['no._of_cases'].sum().reset_index())
dengue_sg['week_idx'] = (dengue_sg['epi_week'].str[:4].astype(int) * 52
                         + dengue_sg['epi_week'].str[6:].astype(int))
sg_lookup = dengue_sg.set_index('week_idx')['no._of_cases'].to_dict()
MAX_SG    = dengue_sg['no._of_cases'].max()   # 1792 — full bulletin, never OOD
print(f"  National dengue max (full bulletin): {MAX_SG}")

# ── Normalisers ───────────────────────────────────────────────────────────────
def log1p_enc(x, max_val):
    return np.log1p(np.asarray(x, dtype=float)) / np.log1p(float(max_val))

def log1p_dec(x, max_val):
    return np.expm1(np.asarray(x, dtype=float) * np.log1p(float(max_val)))

def week_sincos(week_idx):
    iso = week_idx % 52 or 52
    return np.sin(2 * np.pi * iso / 52), np.cos(2 * np.pi * iso / 52)

# ── Model ─────────────────────────────────────────────────────────────────────
class DengueTransformer(nn.Module):
    def __init__(self, n_feat=13, d_model=64, nhead=4, layers=2, drop=0.3):
        super().__init__()
        self.proj    = nn.Linear(n_feat, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, TOT_STEPS, d_model))
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(32, 1))

    def forward(self, x):
        h = self.proj(x) + self.pos_enc
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(df_sub, w_sc, max_local):
    """
    Sliding window over consecutive active rows per subzone.
    Returns X (N, TOT_STEPS, 13), y (N,), meta DataFrame.
    """
    X_list, y_list, meta = [], [], []

    for zone, grp in df_sub.groupby('SUBZONE_N'):
        grp = grp.sort_values('week_idx').reset_index(drop=True)
        n   = len(grp)
        if n < HIST_STEPS + HORIZON:
            continue

        wf   = w_sc.transform(grp[WEATHER_FEATS].values.astype(float))
        wk   = grp['week_idx'].values
        c    = log1p_enc(grp['weekly_cases'].values, max_local)
        sg   = np.array([log1p_enc(sg_lookup.get(w, 0), MAX_SG) for w in wk])

        for i in range(n - HIST_STEPS - HORIZON + 1):
            t = i + HIST_STEPS - 1 + HORIZON   # target row index in grp

            # ── 6 historical steps ──────────────────────────────────────────
            hist = []
            for j in range(i, i + HIST_STEPS):
                s, co = week_sincos(wk[j])
                # weather(8) + sg(1) + cases(1) + flag=0(1) + sin(1) + cos(1)
                hist.append(np.concatenate([wf[j], [sg[j], c[j], 0.0, s, co]]))

            # ── 2 forecast steps (weather only, cases = 0) ──────────────────
            last_hist_wk  = wk[i + HIST_STEPS - 1]
            target_wk     = wk[t]
            forecast_wks  = [last_hist_wk + 1, target_wk]

            fore = []
            for fwk in forecast_wks:
                s, co = week_sincos(fwk)
                key   = (zone, fwk)
                if key in full_lookup.index:
                    fwf = w_sc.transform(
                        full_lookup.loc[key][WEATHER_FEATS].values
                        .astype(float).reshape(1, -1)).flatten()
                else:
                    fwf = np.zeros(len(WEATHER_FEATS), dtype=np.float32)
                # weather(8) + sg=0(1) + cases=0(1) + flag=1(1) + sin(1) + cos(1)
                fore.append(np.concatenate([fwf, [0.0, 0.0, 1.0, s, co]]))

            seq = np.array(hist + fore, dtype=np.float32)   # (8, 13)
            X_list.append(seq)
            y_list.append(float(c[t]))
            meta.append({'year': grp.at[t, 'year'],
                         'SUBZONE_N': zone,
                         'actual_raw': grp.at[t, 'weekly_cases'],
                         'area_km2': grp.at[t, 'area_km2']})

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.float32),
            pd.DataFrame(meta).reset_index(drop=True))

# ── Training ──────────────────────────────────────────────────────────────────
def train_model(X, y):
    cut   = int(len(y) * 0.85)
    tr_dl = DataLoader(SeqDataset(X[:cut],  y[:cut]),  batch_size=BS, shuffle=True)
    vl_dl = DataLoader(SeqDataset(X[cut:],  y[cut:]),  batch_size=BS)

    model = DengueTransformer(n_feat=X.shape[2])
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)

    def w_mse(p, t):
        w = 1.0 + 8.0 * t
        return (w * (p - t) ** 2).mean()

    best_val, best_st, wait = float('inf'), None, 0
    for ep in range(EPOCHS):
        model.train()
        for Xb, yb in tr_dl:
            opt.zero_grad()
            w_mse(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = sum(w_mse(model(Xb), yb).item() for Xb, yb in vl_dl) / len(vl_dl)
        sched.step(vl)
        if vl < best_val:
            best_val, best_st, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop ep{ep+1}  best_val={best_val:.4f}")
                break
        if (ep + 1) % 20 == 0:
            print(f"    ep{ep+1:3d}  val={vl:.4f}")

    model.load_state_dict(best_st)
    return model

# ── Eval helpers ──────────────────────────────────────────────────────────────
def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for Xb, _ in DataLoader(SeqDataset(X, np.zeros(len(X))), batch_size=BS):
            out.append(model(Xb).numpy())
    return np.clip(np.concatenate(out), 0.0, 1.0)

def prf(pred_b, act_b):
    tp = ((pred_b==1)&(act_b==1)).sum()
    fp = ((pred_b==1)&(act_b==0)).sum()
    fn = ((pred_b==0)&(act_b==1)).sum()
    p  = tp/(tp+fp) if tp+fp else 0.0
    r  = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f1, int(tp), int(fp), int(fn)

# ── Fixed K (medically derived) ───────────────────────────────────────────────
# K = (30 cases / 100,000 pop / week) × 8,030 people/km²  = 2.41
# Threshold = max(2, area_km2 × K)   — NEA 2-case floor
K_FIXED   = 2.41
NEA_FLOOR = 2.0

def get_threshold(area_km2):
    return max(NEA_FLOOR, area_km2 * K_FIXED)

# ── Run both folds ────────────────────────────────────────────────────────────
FOLDS = [
    dict(name='Fold 2 — train 2016-2018, test 2019',
         train_years=[2016, 2017, 2018], test_year=2019),
    dict(name='Fold 3 — train 2016-2019, test 2020',
         train_years=[2016, 2017, 2018, 2019], test_year=2020),
]

results = []
for fold in FOLDS:
    print(f"\n{'='*60}")
    print(f"  {fold['name']}")
    print(f"{'='*60}")

    df_tr = df_active[df_active['year'].isin(fold['train_years'])].copy()
    df_te = df_active[df_active['year'] == fold['test_year']].copy()
    print(f"  Train active: {len(df_tr)}  |  Test active: {len(df_te)}")

    w_sc      = StandardScaler()
    w_sc.fit(df_tr[WEATHER_FEATS].values.astype(float))
    max_local = df_tr['weekly_cases'].max()
    print(f"  Max local cases (train): {max_local:.0f}")

    X_tr, y_tr, _       = build_sequences(df_tr, w_sc, max_local)
    X_te, y_te, meta_te = build_sequences(df_te, w_sc, max_local)
    print(f"  Train seqs: {len(X_tr)}  |  Test seqs: {len(X_te)}")

    print("  Training...")
    model = train_model(X_tr, y_tr)

    preds_norm = predict(model, X_te)
    preds_raw  = log1p_dec(preds_norm, max_local)
    actual_raw = meta_te['actual_raw'].values.astype(float)

    mae = np.mean(np.abs(preds_raw - actual_raw))
    ss_r = np.sum((actual_raw - preds_raw)**2)
    ss_t = np.sum((actual_raw - actual_raw.mean())**2)
    r2   = 1 - ss_r / ss_t

    print(f"\n  Regression: MAE={mae:.1f}  R²={r2:.3f}")
    print(f"  Pred  range: {preds_raw.min():.1f} – {preds_raw.max():.1f}  mean={preds_raw.mean():.1f}")
    print(f"  Actual range: {actual_raw.min():.0f} – {actual_raw.max():.0f}  mean={actual_raw.mean():.1f}")

    meta_te = meta_te.reset_index(drop=True)
    pred_b  = np.zeros(len(meta_te), dtype=int)
    act_b   = np.zeros(len(meta_te), dtype=int)
    for i in range(len(meta_te)):
        thr       = get_threshold(meta_te.at[i, 'area_km2'])
        pred_b[i] = int(preds_raw[i] > thr)
        act_b[i]  = int(meta_te.at[i, 'actual_raw'] > thr)

    p, r, f1, tp, fp, fn = prf(pred_b, act_b)
    print(f"  K=2.41  Prec={p:.1%}  Rec={r:.1%}  F1={f1:.1%}  TP={tp} FP={fp} FN={fn}")
    results.append({'Fold': fold['name'][-9:], 'MAE': f'{mae:.1f}',
                    'R²': f'{r2:.3f}', 'Prec': f'{p:.1%}',
                    'Rec': f'{r:.1%}', 'F1': f'{f1:.1%}'})

print(f"\n{'='*60}\nSUMMARY  (K=2.41, max(2, area×K) threshold)\n{'='*60}")
print(pd.DataFrame(results).to_string(index=False))
