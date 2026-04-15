#!/usr/bin/env python
# coding: utf-8
"""
LSTM trained on ACTIVE-ONLY dengue subzone-weeks (weekly_cases > 0).
Rolling window CV: fold1=test2018, fold2=test2019, fold3=test2020.
3 precision metrics:
  1. Fixed threshold (10 cases)
  2. Per-subzone mean from training active weeks
  3. Per-subzone area × K  (K calibrated from training data)

Normalisation approach:
  - Weather:  StandardScaler (fitted on training rows)
  - Target / autoregressive cases: log1p scaled to [0,1] using training max
  - Model output: Sigmoid → decoded via expm1(pred * log1p(max_train))
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

# ── Config ─────────────────────────────────────────────────────────────────────
SEQ_LEN   = 6      # ↑ from 4: captures full onset-and-growth pattern
HORIZON   = 2
BS        = 64
EPOCHS    = 200
PATIENCE  = 20
LR        = 5e-4
WD        = 1e-4
FIXED_THR = 10      # metric 1: fixed case threshold

# Weather features + sg_total_cases (re-added, bounded log1p/max normalization)
WEATHER_FEATS = [
    'weekly_rainfall_mm', 'rainfall_mean_daily', 'rainfall_sd_daily',
    'temp_avg', 'temp_range', 'humidity_avg', 'humidity_range',
    'area_km2',
]
# sg_total_cases handled separately with its own [0,1] normalization
USE_SG_TOTAL = True

# Fold 1 (test 2018) removed — too little training data (2016–2017 only)
FOLDS = [
    dict(name='Fold2 (test 2019)', train_years=list(range(2016, 2019)), test_year=2019),
    dict(name='Fold3 (test 2020)', train_years=list(range(2016, 2020)), test_year=2020),
]

# ── Load & filter ───────────────────────────────────────────────────────────────
print("Loading data...")
df_full = pd.read_csv('dengue_analysis_complete.csv')
print(f"  Total rows:        {len(df_full)}")

df = df_full[df_full['weekly_cases'] > 0].copy()
print(f"  Active rows (>0):  {len(df)}")
print(f"  Year dist:\n{df.groupby('year').size().to_string()}")

df['week_idx'] = df['year'] * 52 + df['iso_week']
df = df.sort_values(['SUBZONE_N', 'week_idx']).reset_index(drop=True)

# ── Model definition ─────────────────────────────────────────────────────────
# n_features = len(WEATHER_FEATS) + 1 sg + 1 cases = 10
class DengueTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, layers=2, drop=0.3):
        super().__init__()
        self.proj    = nn.Linear(n_features, d_model)
        # learnable positional encoding for SEQ_LEN steps
        self.pos_enc = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=drop,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.proj(x) + self.pos_enc   # (B, SEQ_LEN, d_model)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)   # use last time-step token

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── Helpers ──────────────────────────────────────────────────────────────────
def make_case_scaler(max_log1p_cases):
    """Returns (encode_fn, decode_fn) to map cases → [0,1] and back."""
    def enc(cases):
        return np.log1p(np.array(cases, dtype=float)) / max_log1p_cases
    def dec(norm):
        return np.expm1(np.array(norm, dtype=float) * max_log1p_cases)
    return enc, dec


def build_sequences(df_sub, w_scaler, case_enc, sg_enc=None, active_targets=True):
    """
    Sliding-window sequences.
    Features per step: weather (8) + sg_total (1, optional) + auto-regressive cases (1)
    Target: normalised cases at step t+HORIZON.
    """
    X_list, y_list, meta = [], [], []

    for zone, grp in df_sub.groupby('SUBZONE_N'):
        grp = grp.sort_values('week_idx').reset_index(drop=True)
        n   = len(grp)
        if n < SEQ_LEN + HORIZON:
            continue

        wf  = w_scaler.transform(grp[WEATHER_FEATS].values.astype(float))
        cn  = case_enc(grp['weekly_cases'].values)   # [0,1]
        yr  = grp['year'].values

        parts = [wf]
        if sg_enc is not None:
            sg = sg_enc(grp['sg_total_cases'].values).reshape(-1, 1)
            parts.append(sg)
        parts.append(cn.reshape(-1, 1))
        all_feat = np.hstack(parts)   # (n, n_features)

        for i in range(n - SEQ_LEN - HORIZON + 1):
            t = i + SEQ_LEN - 1 + HORIZON
            if active_targets and grp['weekly_cases'].iloc[t] == 0:
                continue
            X_list.append(all_feat[i:i+SEQ_LEN].astype(np.float32))
            y_list.append(float(cn[t]))
            meta.append({
                'year': yr[t],
                'SUBZONE_N': zone,
                'actual_raw': grp['weekly_cases'].iloc[t],
                'area_km2': grp['area_km2'].iloc[t],
            })

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            pd.DataFrame(meta).reset_index(drop=True))


def train_model(X_tr, y_tr, n_feat):
    val_cut = int(len(X_tr) * 0.85)
    Xv, yv = X_tr[val_cut:], y_tr[val_cut:]
    Xt, yt = X_tr[:val_cut], y_tr[:val_cut]

    tr_dl = DataLoader(SeqDataset(Xt, yt), batch_size=BS, shuffle=True)
    vl_dl = DataLoader(SeqDataset(Xv, yv), batch_size=BS)

    model   = DengueTransformer(n_feat)
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)

    def weighted_mse(pred, target):
        # up-weight high-case sequences so model isn't pulled to predict the mean
        w = 1.0 + 8.0 * target   # target in [0,1]; outbreak week (0.8) gets 7.4× weight
        return (w * (pred - target) ** 2).mean()

    best_val, best_st, wait = float('inf'), None, 0
    for ep in range(EPOCHS):
        model.train()
        for Xb, yb in tr_dl:
            opt.zero_grad()
            loss = weighted_mse(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = sum(weighted_mse(model(Xb), yb).item() for Xb, yb in vl_dl) / len(vl_dl)
        sched.step(vl)
        if vl < best_val:
            best_val, best_st, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop epoch {ep+1}  (best val={best_val:.4f})")
                break
        if (ep+1) % 20 == 0:
            print(f"    ep{ep+1:3d}  val={vl:.4f}")

    model.load_state_dict(best_st)
    return model


def predict(model, X_te):
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X_te)).numpy()
    return np.clip(out, 0.0, 1.0)   # keep in valid [0,1] range


def precision_recall(pred_bin, actual_bin):
    tp = ((pred_bin == 1) & (actual_bin == 1)).sum()
    fp = ((pred_bin == 1) & (actual_bin == 0)).sum()
    fn = ((pred_bin == 0) & (actual_bin == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1, int(tp), int(fp), int(fn)


# ── Rolling CV ────────────────────────────────────────────────────────────────
fold_results = []

for fold in FOLDS:
    print(f"\n{'='*60}")
    print(f"  {fold['name']}")
    print(f"{'='*60}")

    train_yrs = fold['train_years']
    test_yr   = fold['test_year']

    df_tr = df[df['year'].isin(train_yrs)].copy()
    df_te = df[df['year'] == test_yr].copy()
    print(f"  Train active rows: {len(df_tr)}  |  Test active rows: {len(df_te)}")

    # ── Fit scalers on training data ─────────────────────────────────────────
    w_sc = StandardScaler()
    w_sc.fit(df_tr[WEATHER_FEATS].values.astype(float))

    # Case normalisation: log1p / log1p(max_train) → [0,1]
    max_log1p = np.log1p(df_tr['weekly_cases'].max())
    case_enc, case_dec = make_case_scaler(max_log1p)
    print(f"  Training max cases: {df_tr['weekly_cases'].max():.0f}  log1p_max={max_log1p:.3f}")

    # sg_total_cases normalisation: same bounded log1p/max approach
    sg_enc = None
    if USE_SG_TOTAL and 'sg_total_cases' in df_tr.columns:
        max_sg_log1p = np.log1p(df_tr['sg_total_cases'].max())
        sg_enc, _ = make_case_scaler(max_sg_log1p)
        print(f"  Training max sg_total: {df_tr['sg_total_cases'].max():.0f}  log1p_max={max_sg_log1p:.3f}")

    # ── Build sequences ───────────────────────────────────────────────────────
    X_tr_s, y_tr_s, meta_tr = build_sequences(df_tr, w_sc, case_enc, sg_enc, active_targets=True)
    X_te_s, y_te_s, meta_te = build_sequences(df_te, w_sc, case_enc, sg_enc, active_targets=True)

    print(f"  Train seqs: {len(X_tr_s)}  |  Test seqs: {len(X_te_s)}")
    if len(X_tr_s) == 0 or len(X_te_s) == 0:
        print("  Skipping fold — not enough sequences")
        continue

    n_feat = X_tr_s.shape[2]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("  Training LSTM...")
    model = train_model(X_tr_s, y_tr_s, n_feat)

    # ── Predict ──────────────────────────────────────────────────────────────
    preds_norm = predict(model, X_te_s)       # in [0, 1]
    preds_raw  = case_dec(preds_norm)          # actual cases
    actual_raw = meta_te['actual_raw'].values.astype(float)

    mae  = np.mean(np.abs(preds_raw - actual_raw))
    rmse = np.sqrt(np.mean((preds_raw - actual_raw)**2))
    ss_r = np.sum((actual_raw - preds_raw)**2)
    ss_t = np.sum((actual_raw - actual_raw.mean())**2)
    r2   = 1 - ss_r / ss_t if ss_t > 0 else float('nan')

    print(f"\n  Regression:  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}")
    print(f"  Pred  range: min={preds_raw.min():.1f}  mean={preds_raw.mean():.1f}  max={preds_raw.max():.1f}")
    print(f"  Actual range: min={actual_raw.min():.0f}  mean={actual_raw.mean():.1f}  max={actual_raw.max():.0f}")

    # ── Threshold computation from training data ──────────────────────────────
    # T2: per-subzone mean of training active cases
    thresh_mean   = df_tr.groupby('SUBZONE_N')['weekly_cases'].mean()
    global_tr_mean = df_tr['weekly_cases'].mean()

    # T3: area × K, calibrated so median(area×K) = median(thresh_mean)
    subzone_area = df_tr.groupby('SUBZONE_N')['area_km2'].first()
    common       = thresh_mean.index.intersection(subzone_area.index)
    median_mean  = thresh_mean[common].median()
    median_area  = subzone_area[common].median()
    K            = median_mean / median_area if median_area > 0 else 1.0
    thresh_area  = subzone_area * K
    print(f"\n  K = {K:.2f} cases/km²  (median subzone_mean={median_mean:.1f},  median_area={median_area:.2f} km²)")

    # ── Precision metrics ─────────────────────────────────────────────────────
    print(f"\n  {'Metric':<32} {'Prec':>6} {'Rec':>6} {'F1':>6}  TP/FP/FN")

    # Area × K only
    pred_bin   = np.zeros(len(meta_te), dtype=int)
    actual_bin = np.zeros(len(meta_te), dtype=int)
    for i in range(len(meta_te)):
        z   = meta_te.at[i, 'SUBZONE_N']
        thr = thresh_area.get(z, K * subzone_area.get(z, median_area))
        pred_bin[i]   = int(preds_raw[i] > thr)
        actual_bin[i] = int(meta_te.at[i, 'actual_raw'] > thr)
    prec, rec, f1, tp, fp, fn = precision_recall(pred_bin, actual_bin)
    print(f"\n  Area×K (K={K:.2f})  Prec={prec:.1%}  Rec={rec:.1%}  F1={f1:.1%}  TP={tp} FP={fp} FN={fn}")

    fold_results.append({
        'fold': fold['name'], 'mae': mae, 'rmse': rmse, 'r2': r2,
        'K': K, 'prec_area': prec, 'rec_area': rec, 'f1_area': f1,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY ACROSS FOLDS")
print(f"{'='*60}")
if fold_results:
    res = pd.DataFrame(fold_results)
    show = res[['fold', 'mae', 'r2', 'K', 'prec_area', 'rec_area', 'f1_area']]
    show.columns = ['Fold', 'MAE', 'R²', 'K', 'Prec(Area×K)', 'Rec(Area×K)', 'F1(Area×K)']
    print(show.to_string(index=False))
    print(f"\nAvg R²:          {res['r2'].mean():.3f}")
    print(f"Avg Prec Area×K: {res['prec_area'].mean():.1%}")
    print(f"Avg Rec  Area×K: {res['rec_area'].mean():.1%}")
    print(f"Avg F1   Area×K: {res['f1_area'].mean():.1%}")
