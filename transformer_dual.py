#!/usr/bin/env python
# coding: utf-8
"""
Dual-input Transformer for dengue outbreak prediction.

Input stream 1 — Local sequence (SEQ_LEN × local_features):
    weather (8) + autoregressive subzone cases (1) = 9 dims per step
    Context built from the FULL GRID (including zero-case weeks) so we always
    have a calendar-aligned 6-week window. Missing (subzone, week) pairs are
    zero-padded and masked out in Transformer attention.

Input stream 2 — National trend vector (SEQ_LEN values):
    Country-wide dengue cases per week from WeeklyInfectiousDiseaseBulletinCases.csv
    Same 6 calendar weeks as stream 1. Processed by a separate MLP then
    concatenated with the local encoding before the prediction head.

Targets: active subzone-weeks only (weekly_cases > 0).
Threshold: area × K (K calibrated per fold from training data).
Folds: test 2019, test 2020.
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
SEQ_LEN   = 6
HORIZON   = 2
BS        = 64
EPOCHS    = 200
PATIENCE  = 20
LR        = 5e-4
WD        = 1e-4

WEATHER_FEATS = [
    'weekly_rainfall_mm', 'rainfall_mean_daily', 'rainfall_sd_daily',
    'temp_avg', 'temp_range', 'humidity_avg', 'humidity_range', 'area_km2'
]   # 8 features; sg_total_cases replaced by dedicated national trend stream

FOLDS = [
    dict(name='Fold2 (test 2019)', train_years=list(range(2016, 2019)), test_year=2019),
    dict(name='Fold3 (test 2020)', train_years=list(range(2016, 2020)), test_year=2020),
]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")

# Full grid — used for 6-week context windows (includes zero-case weeks)
df_full = pd.read_csv('dengue_analysis_complete.csv')
df_full['week_idx'] = df_full['year'] * 52 + df_full['iso_week']
# Fast lookup: (SUBZONE_N, week_idx) → row
full_lookup = df_full.set_index(['SUBZONE_N', 'week_idx'])
print(f"  Full grid rows:    {len(df_full)}")

# ── 3-NN lookup from URA subzone centroids ─────────────────────────────────────
import geopandas as gpd
from scipy.spatial import cKDTree

ura = gpd.read_file('ura_subzones.geojson').to_crs(epsg=3414)
ura['cx'] = ura.geometry.centroid.x
ura['cy'] = ura.geometry.centroid.y
ura = ura[['SUBZONE_N', 'cx', 'cy']].dropna()

coords   = ura[['cx', 'cy']].values
tree     = cKDTree(coords)
# query 4 neighbours (first is self), take the 3 after
dists, idxs = tree.query(coords, k=4)
nn3_lookup = {}
for i, row in ura.iterrows():
    zone       = row['SUBZONE_N']
    neighbour_idxs = idxs[ura.index.get_loc(i)][1:]   # skip self
    nn3_lookup[zone] = ura.iloc[neighbour_idxs]['SUBZONE_N'].tolist()

print(f"  3-NN lookup built for {len(nn3_lookup)} subzones")

# Active targets only
df_active = df_full[df_full['weekly_cases'] > 0].copy()
print(f"  Active rows (>0):  {len(df_active)}")

# National dengue weekly from bulletin
bulletin  = pd.read_csv('WeeklyInfectiousDiseaseBulletinCases.csv')
dengue_sg = (
    bulletin[bulletin['disease'].str.contains('Dengue', case=False)]
    .groupby('epi_week')['no._of_cases'].sum()
    .reset_index()
)
dengue_sg['year']     = dengue_sg['epi_week'].str[:4].astype(int)
dengue_sg['iso_week'] = dengue_sg['epi_week'].str[6:].astype(int)
dengue_sg['week_idx'] = dengue_sg['year'] * 52 + dengue_sg['iso_week']
# lookup: week_idx → national dengue cases
sg_cases_lookup = dengue_sg.set_index('week_idx')['no._of_cases'].to_dict()
print(f"  National dengue weeks: {len(sg_cases_lookup)} (range {dengue_sg['year'].min()}–{dengue_sg['year'].max()})")

# ── Normalisers ───────────────────────────────────────────────────────────────
def make_log1p_scaler(max_val):
    """Encode/decode cases to [0,1] via log1p / log1p(max_val)."""
    M = np.log1p(float(max_val))
    enc = lambda v: np.log1p(np.asarray(v, dtype=float)) / M
    dec = lambda v: np.expm1(np.asarray(v, dtype=float) * M)
    return enc, dec

# ── Dual-input Transformer ────────────────────────────────────────────────────
class DualTransformer(nn.Module):
    """
    Stream 1: local SEQ_LEN × n_local_feat  →  TransformerEncoder → mean-pool → h_local
    Stream 2: national SEQ_LEN trend vector  →  MLP                → h_sg
    Output  : head( cat(h_local, h_sg) ) → scalar prediction
    """
    def __init__(self, n_local_feat, d_model=64, nhead=4, layers=2, drop=0.3):
        super().__init__()
        # Local stream
        self.proj    = nn.Linear(n_local_feat, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=drop,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        # National trend stream
        self.sg_mlp = nn.Sequential(
            nn.Linear(SEQ_LEN, 32), nn.ReLU(), nn.Dropout(drop)
        )

        # Combined head
        self.head = nn.Sequential(
            nn.Linear(d_model + 32, 32), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(32, 1)
        )

    def forward(self, x_local, x_sg, pad_mask=None):
        """
        x_local : (B, SEQ_LEN, n_local_feat)
        x_sg    : (B, SEQ_LEN)  — national trend, normalised
        pad_mask: (B, SEQ_LEN) bool — True = padded position, ignored in attention
        """
        h = self.proj(x_local) + self.pos_enc
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        # safe mean-pool: average only unmasked positions
        if pad_mask is not None:
            keep = (~pad_mask).float().unsqueeze(-1)   # (B, SEQ, 1)
            h = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        sg_h     = self.sg_mlp(x_sg)
        combined = torch.cat([h, sg_h], dim=1)
        return self.head(combined).squeeze(-1)


class DualDataset(Dataset):
    def __init__(self, X_local, X_sg, masks, y, ctx_wt):
        self.X_local = torch.FloatTensor(X_local)
        self.X_sg    = torch.FloatTensor(X_sg)
        self.masks   = torch.BoolTensor(masks)
        self.y       = torch.FloatTensor(y)
        self.ctx_wt  = torch.FloatTensor(ctx_wt)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.X_local[i], self.X_sg[i], self.masks[i], self.y[i], self.ctx_wt[i]


# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(df_targets, w_scaler, case_enc, sg_enc, active_targets=True):
    """
    For each active target row, assemble:
      - 6-week local context from the full grid (zero-pad missing weeks)
      - 6-week national trend vector
      - padding mask (True = padded position)
      - context_weight = n_real_weeks / SEQ_LEN  (for weighted training loss)
    Features per step: weather(8) + local_cases(1) + nn3_avg(1) + sin(1) + cos(1) = 12 dims
    """
    X_local_list, X_sg_list, mask_list, y_list, ctx_wt_list, meta = [], [], [], [], [], []

    for zone, grp in df_targets.groupby('SUBZONE_N'):
        grp      = grp.sort_values('week_idx').reset_index(drop=True)
        nbrs     = nn3_lookup.get(zone, [])

        for row_i in range(len(grp)):
            t_widx = grp.at[row_i, 'week_idx']
            t_raw  = grp.at[row_i, 'weekly_cases']
            t_yr   = grp.at[row_i, 'year']

            if active_targets and t_raw == 0:
                continue

            end_ctx   = t_widx - HORIZON
            ctx_weeks = list(range(end_ctx - SEQ_LEN + 1, end_ctx + 1))

            local_seq = []
            sg_seq    = []
            pad_flags = []
            n_real    = 0

            for wk in ctx_weeks:
                # ── Cyclical week-of-year ──────────────────────────────────
                iso_wk = wk % 52 or 52
                w_sin  = np.sin(2 * np.pi * iso_wk / 52)
                w_cos  = np.cos(2 * np.pi * iso_wk / 52)

                # ── National trend ─────────────────────────────────────────
                sg_seq.append(float(sg_enc(sg_cases_lookup.get(wk, 0.0))))

                # ── Local + 3-NN ───────────────────────────────────────────
                key = (zone, wk)
                if key in full_lookup.index:
                    row_data  = full_lookup.loc[key]
                    wf_scaled = w_scaler.transform(
                        row_data[WEATHER_FEATS].values.astype(float).reshape(1, -1)
                    ).flatten()
                    c_norm = float(case_enc(float(row_data['weekly_cases'])))

                    # 3-NN average cases (normalised same scale as local)
                    nn_vals = [float(full_lookup.loc[(nz, wk)]['weekly_cases'])
                               for nz in nbrs if (nz, wk) in full_lookup.index]
                    nn_norm = float(case_enc(np.mean(nn_vals))) if nn_vals else 0.0

                    # weather(8) + local(1) + nn_avg(1) + sin(1) + cos(1) = 12
                    local_seq.append(np.concatenate([wf_scaled, [c_norm, nn_norm, w_sin, w_cos]]))
                    pad_flags.append(False)
                    n_real += 1
                else:
                    local_seq.append(np.array([0]*10 + [w_sin, w_cos], dtype=np.float32))
                    pad_flags.append(True)

            if all(pad_flags):
                continue

            X_local_list.append(np.array(local_seq, dtype=np.float32))
            X_sg_list.append(np.array(sg_seq, dtype=np.float32))
            mask_list.append(pad_flags)
            y_list.append(float(case_enc(t_raw)))
            ctx_wt_list.append(n_real / SEQ_LEN)   # 1.0 = full context, <1 = partial
            meta.append({
                'year':       t_yr,
                'SUBZONE_N':  zone,
                'actual_raw': t_raw,
                'area_km2':   grp.at[row_i, 'area_km2'],
            })

    return (np.array(X_local_list, dtype=np.float32),
            np.array(X_sg_list,    dtype=np.float32),
            np.array(mask_list,    dtype=bool),
            np.array(y_list,       dtype=np.float32),
            np.array(ctx_wt_list,  dtype=np.float32),
            pd.DataFrame(meta).reset_index(drop=True))


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(X_local, X_sg, masks, y, ctx_wt, n_feat):
    cut = int(len(y) * 0.85)
    tr = DualDataset(X_local[:cut], X_sg[:cut], masks[:cut], y[:cut], ctx_wt[:cut])
    vl = DualDataset(X_local[cut:], X_sg[cut:], masks[cut:], y[cut:], ctx_wt[cut:])
    tr_dl = DataLoader(tr, batch_size=BS, shuffle=True)
    vl_dl = DataLoader(vl, batch_size=BS)

    model = DualTransformer(n_feat)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)

    def w_mse(pred, target, ctx_w):
        # case weight: up-weight high-case sequences
        # context weight: down-weight sequences with partial history
        w = (1.0 + 8.0 * target) * ctx_w
        return (w * (pred - target) ** 2).mean()

    best_val, best_st, wait = float('inf'), None, 0
    for ep in range(EPOCHS):
        model.train()
        for Xl, Xs, Xm, yb, cw in tr_dl:
            opt.zero_grad()
            loss = w_mse(model(Xl, Xs, Xm), yb, cw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl_loss = sum(w_mse(model(Xl, Xs, Xm), yb, cw).item()
                          for Xl, Xs, Xm, yb, cw in vl_dl) / len(vl_dl)
        sched.step(vl_loss)
        if vl_loss < best_val:
            best_val, best_st, wait = vl_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop ep{ep+1}  best_val={best_val:.4f}")
                break
        if (ep+1) % 20 == 0:
            print(f"    ep{ep+1:3d}  val={vl_loss:.4f}")

    model.load_state_dict(best_st)
    return model


def predict(model, X_local, X_sg, masks):
    model.eval()
    dummy = np.ones(len(X_local), dtype=np.float32)
    dl = DataLoader(DualDataset(X_local, X_sg, masks,
                                np.zeros(len(X_local), dtype=np.float32), dummy),
                    batch_size=BS)
    out = []
    with torch.no_grad():
        for Xl, Xs, Xm, _, _cw in dl:
            out.append(model(Xl, Xs, Xm).numpy())
    return np.clip(np.concatenate(out), 0.0, 1.0)


def precision_recall_f1(pred_bin, actual_bin):
    tp = ((pred_bin==1)&(actual_bin==1)).sum()
    fp = ((pred_bin==1)&(actual_bin==0)).sum()
    fn = ((pred_bin==0)&(actual_bin==1)).sum()
    p  = tp/(tp+fp) if tp+fp>0 else 0.0
    r  = tp/(tp+fn) if tp+fn>0 else 0.0
    f1 = 2*p*r/(p+r) if p+r>0 else 0.0
    return p, r, f1, int(tp), int(fp), int(fn)


# ── Rolling CV ────────────────────────────────────────────────────────────────
fold_results = []

for fold in FOLDS:
    print(f"\n{'='*62}")
    print(f"  {fold['name']}")
    print(f"{'='*62}")

    train_yrs = fold['train_years']
    test_yr   = fold['test_year']

    df_tr_active = df_active[df_active['year'].isin(train_yrs)].copy()
    df_te_active = df_active[df_active['year'] == test_yr].copy()
    print(f"  Train active: {len(df_tr_active)} rows  |  Test active: {len(df_te_active)} rows")

    # ── Fit scalers on training active rows ──────────────────────────────────
    w_sc = StandardScaler()
    w_sc.fit(df_tr_active[WEATHER_FEATS].values.astype(float))

    max_local  = df_tr_active['weekly_cases'].max()
    case_enc, case_dec = make_log1p_scaler(max_local)

    # Use full bulletin range for sg max so 2019/2020 outbreak peaks stay in [0,1]
    max_sg = dengue_sg['no._of_cases'].max()   # 1792 across 2012-2022
    sg_enc, _ = make_log1p_scaler(max_sg)
    print(f"  Max local cases (train): {max_local:.0f}  |  Max national (full bulletin): {max_sg:.0f}")

    # ── Build sequences ───────────────────────────────────────────────────────
    X_l_tr, X_s_tr, M_tr, y_tr, cw_tr, meta_tr = build_sequences(
        df_tr_active, w_sc, case_enc, sg_enc, active_targets=True)
    X_l_te, X_s_te, M_te, y_te, cw_te, meta_te = build_sequences(
        df_te_active, w_sc, case_enc, sg_enc, active_targets=True)

    print(f"  Train seqs: {len(y_tr)}  |  Test seqs: {len(y_te)}")
    # Report how many sequences had at least one padded step
    padded_tr = (M_tr.any(axis=1)).sum()
    padded_te = (M_te.any(axis=1)).sum()
    print(f"  Sequences with ≥1 padded week: train={padded_tr} ({100*padded_tr/max(len(y_tr),1):.1f}%)  "
          f"test={padded_te} ({100*padded_te/max(len(y_te),1):.1f}%)")

    n_feat = X_l_tr.shape[2]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("  Training Dual-Input Transformer...")
    model = train_model(X_l_tr, X_s_tr, M_tr, y_tr, cw_tr, n_feat)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    preds_norm = predict(model, X_l_te, X_s_te, M_te)
    preds_raw  = case_dec(preds_norm)
    actual_raw = meta_te['actual_raw'].values.astype(float)

    mae  = np.mean(np.abs(preds_raw - actual_raw))
    ss_r = np.sum((actual_raw - preds_raw)**2)
    ss_t = np.sum((actual_raw - actual_raw.mean())**2)
    r2   = 1 - ss_r/ss_t if ss_t > 0 else float('nan')

    print(f"\n  Regression: MAE={mae:.1f}  R²={r2:.3f}")
    print(f"  Pred  range: {preds_raw.min():.1f} – {preds_raw.max():.1f}  mean={preds_raw.mean():.1f}")
    print(f"  Actual range: {actual_raw.min():.0f} – {actual_raw.max():.0f}  mean={actual_raw.mean():.1f}")

    # ── Area × K threshold ────────────────────────────────────────────────────
    thresh_mean  = df_tr_active.groupby('SUBZONE_N')['weekly_cases'].mean()
    subzone_area = df_tr_active.groupby('SUBZONE_N')['area_km2'].first()
    common       = thresh_mean.index.intersection(subzone_area.index)
    K            = thresh_mean[common].median() / subzone_area[common].median()
    thresh_area  = subzone_area * K
    median_area  = subzone_area[common].median()
    print(f"\n  K = {K:.2f} cases/km²")

    pred_bin   = np.zeros(len(meta_te), dtype=int)
    actual_bin = np.zeros(len(meta_te), dtype=int)
    for i in range(len(meta_te)):
        z   = meta_te.at[i, 'SUBZONE_N']
        thr = thresh_area.get(z, K * subzone_area.get(z, median_area))
        pred_bin[i]   = int(preds_raw[i] > thr)
        actual_bin[i] = int(meta_te.at[i, 'actual_raw'] > thr)

    prec, rec, f1, tp, fp, fn = precision_recall_f1(pred_bin, actual_bin)
    print(f"  Area×K  Prec={prec:.1%}  Rec={rec:.1%}  F1={f1:.1%}  TP={tp} FP={fp} FN={fn}")

    fold_results.append({'fold': fold['name'], 'mae': mae, 'r2': r2,
                         'K': K, 'prec': prec, 'rec': rec, 'f1': f1})

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("SUMMARY")
print(f"{'='*62}")
res = pd.DataFrame(fold_results)
print(res[['fold','mae','r2','K','prec','rec','f1']].to_string(index=False))
print(f"\nAvg R²:   {res['r2'].mean():.3f}")
print(f"Avg Prec: {res['prec'].mean():.1%}")
print(f"Avg Rec:  {res['rec'].mean():.1%}")
print(f"Avg F1:   {res['f1'].mean():.1%}")
