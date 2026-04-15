#!/usr/bin/env python
# coding: utf-8
"""
Holdout evaluation: 10% of weeks (every 10th) evenly spread across 2016-2020.
Train on remaining 90%. Test all 3 versions on the same holdout weeks.

V1 — Single-stream, 6 hist steps
     Features/step: weather(8) + sg(1) + cases(1) + sin(1) + cos(1) = 12

V2 — Single-stream, 6 hist + 2 forecast weather steps
     Hist features:     weather(8) + sg(1) + cases(1) + flag=0 + sin + cos = 13
     Forecast features: weather(8) + sg=0  + cases=0  + flag=1 + sin + cos = 13

V3 — Dual-input: local stream (6 hist) + separate sg-trend MLP
     Local/step: weather(8) + cases(1) + sin(1) + cos(1) = 11
     sg vector:  6 values (normalised national cases)

Threshold: area × K (K from training data).
Sequences: calendar-based for both train and test
           (every active row gets a context window looked up from full grid).
"""

import os, warnings
os.chdir('/Users/sathwiiksai/Documents/DSFLab')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch, torch.nn as nn, copy
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
HIST  = 6
FORE  = 2
BS    = 64
EPOCHS = 200
PATIENCE = 20
LR    = 5e-4
WD    = 1e-4

WEATHER_FEATS = [
    'weekly_rainfall_mm', 'rainfall_mean_daily', 'rainfall_sd_daily',
    'temp_avg', 'temp_range', 'humidity_avg', 'humidity_range', 'area_km2'
]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df_full = pd.read_csv('dengue_analysis_complete.csv')
df_full['week_idx'] = df_full['year'] * 52 + df_full['iso_week']
full_lookup = df_full.set_index(['SUBZONE_N', 'week_idx'])

df_active = df_full[df_full['weekly_cases'] > 0].copy()

bulletin  = pd.read_csv('WeeklyInfectiousDiseaseBulletinCases.csv')
dengue_sg = (bulletin[bulletin['disease'].str.contains('Dengue', case=False)]
             .groupby('epi_week')['no._of_cases'].sum().reset_index())
dengue_sg['week_idx'] = (dengue_sg['epi_week'].str[:4].astype(int) * 52
                         + dengue_sg['epi_week'].str[6:].astype(int))
sg_lookup = dengue_sg.set_index('week_idx')['no._of_cases'].to_dict()
MAX_SG    = dengue_sg['no._of_cases'].max()   # 1792

# ── Holdout split: every 10th week ───────────────────────────────────────────
all_weeks    = sorted(df_full['week_idx'].unique())
holdout_wks  = set(all_weeks[9::10])                   # 24 weeks evenly spread
train_wks    = set(w for w in all_weeks if w not in holdout_wks)

df_tr = df_active[~df_active['week_idx'].isin(holdout_wks)].copy()
df_te = df_active[ df_active['week_idx'].isin(holdout_wks)].copy()
print(f"  Train active: {len(df_tr)} rows | Holdout active: {len(df_te)} rows")
print(f"  Holdout rows/year:\n{df_te.groupby('year').size().to_string()}")

# ── Fit scalers on training data ──────────────────────────────────────────────
w_sc = StandardScaler()
w_sc.fit(df_tr[WEATHER_FEATS].values.astype(float))
max_local = df_tr['weekly_cases'].max()
print(f"\n  max_local={max_local:.0f}  max_sg={MAX_SG}")

# ── NEA-aligned threshold ─────────────────────────────────────────────────────
# Per (subzone, iso_week): threshold = max(2, mean + 1*std) from training years.
# 2-case floor = NEA minimum cluster definition.
# mean+std per calendar week = season-adjusted epidemic threshold.
df_full['iso_week'] = df_full['week_idx'] - df_full['year'] * 52

grp_wk = (df_tr.copy()
          .assign(iso_week=lambda d: d['week_idx'] - d['year']*52)
          .groupby(['SUBZONE_N', 'iso_week'])['weekly_cases'])
nea_mean = grp_wk.mean()
nea_std  = grp_wk.std().fillna(0)
nea_thresh = (nea_mean + nea_std).clip(lower=2).rename('threshold')

# lookup function used in metrics
def get_nea_thr(zone, iso_wk):
    key = (zone, iso_wk)
    if key in nea_thresh.index:
        return nea_thresh[key]
    return 2.0   # NEA floor for unseen subzone-week combos

print(f"  NEA threshold stats (training):")
print(f"    median={nea_thresh.median():.1f}  mean={nea_thresh.mean():.1f}  "
      f"min={nea_thresh.min():.1f}  max={nea_thresh.max():.1f}")

# ── Normalisers ───────────────────────────────────────────────────────────────
def enc(x, M): return np.log1p(np.asarray(x, dtype=float)) / np.log1p(float(M))
def dec(x, M): return np.expm1(np.asarray(x, dtype=float) * np.log1p(float(M)))

def sincos(week_idx):
    iso = week_idx % 52 or 52
    return np.sin(2*np.pi*iso/52), np.cos(2*np.pi*iso/52)

def wx(zone, wk):
    """Weather features for (zone, wk), zeros if missing."""
    key = (zone, wk)
    if key in full_lookup.index:
        return w_sc.transform(
            full_lookup.loc[key][WEATHER_FEATS].values.astype(float).reshape(1,-1)
        ).flatten()
    return np.zeros(len(WEATHER_FEATS), dtype=np.float32)

# ── Sequence builders ─────────────────────────────────────────────────────────

def build_v1(df_targets):
    """6 hist steps. Features: weather(8)+sg(1)+cases(1)+sin+cos = 12."""
    X, y, meta = [], [], []
    for zone, grp in df_targets.groupby('SUBZONE_N'):
        grp = grp.sort_values('week_idx').reset_index(drop=True)
        for ri in range(len(grp)):
            t_wk  = grp.at[ri, 'week_idx']
            t_raw = grp.at[ri, 'weekly_cases']
            ctx   = list(range(t_wk - 2 - HIST + 1, t_wk - 2 + 1))  # 6 weeks
            steps = []
            n_real = 0
            for wk in ctx:
                s, co = sincos(wk)
                key   = (zone, wk)
                if key in full_lookup.index:
                    c_raw = float(full_lookup.loc[key]['weekly_cases'])
                    n_real += 1
                else:
                    c_raw = 0.0
                sg_v  = float(enc(sg_lookup.get(wk, 0), MAX_SG))
                c_v   = float(enc(c_raw, max_local))
                steps.append(np.concatenate([wx(zone,wk), [sg_v, c_v, s, co]]))
            if n_real == 0: continue
            X.append(np.array(steps, dtype=np.float32))
            y.append(float(enc(t_raw, max_local)))
            meta.append({'SUBZONE_N': zone, 'actual_raw': t_raw,
                         'area_km2': grp.at[ri,'area_km2'], 'year': grp.at[ri,'year'],
                         'iso_week': int(grp.at[ri,'week_idx'] - grp.at[ri,'year']*52)})
    return np.array(X,np.float32), np.array(y,np.float32), pd.DataFrame(meta).reset_index(drop=True)


def build_v2(df_targets):
    """6 hist + 2 forecast steps. 13 dims/step."""
    X, y, meta = [], [], []
    for zone, grp in df_targets.groupby('SUBZONE_N'):
        grp = grp.sort_values('week_idx').reset_index(drop=True)
        for ri in range(len(grp)):
            t_wk  = grp.at[ri, 'week_idx']
            t_raw = grp.at[ri, 'weekly_cases']
            ctx   = list(range(t_wk - 2 - HIST + 1, t_wk - 2 + 1))
            fore  = [t_wk - 1, t_wk]
            steps = []
            n_real = 0
            for wk in ctx:
                s, co = sincos(wk)
                key   = (zone, wk)
                if key in full_lookup.index:
                    c_raw = float(full_lookup.loc[key]['weekly_cases'])
                    n_real += 1
                else:
                    c_raw = 0.0
                sg_v = float(enc(sg_lookup.get(wk, 0), MAX_SG))
                c_v  = float(enc(c_raw, max_local))
                steps.append(np.concatenate([wx(zone,wk), [sg_v, c_v, 0.0, s, co]]))
            for wk in fore:
                s, co = sincos(wk)
                steps.append(np.concatenate([wx(zone,wk), [0.0, 0.0, 1.0, s, co]]))
            if n_real == 0: continue
            X.append(np.array(steps, dtype=np.float32))
            y.append(float(enc(t_raw, max_local)))
            meta.append({'SUBZONE_N': zone, 'actual_raw': t_raw,
                         'area_km2': grp.at[ri,'area_km2'], 'year': grp.at[ri,'year'],
                         'iso_week': int(grp.at[ri,'week_idx'] - grp.at[ri,'year']*52)})
    return np.array(X,np.float32), np.array(y,np.float32), pd.DataFrame(meta).reset_index(drop=True)


def build_v3(df_targets):
    """Dual: local seq 11 dims + sg vector 6 values."""
    X_loc, X_sg, y, meta = [], [], [], []
    for zone, grp in df_targets.groupby('SUBZONE_N'):
        grp = grp.sort_values('week_idx').reset_index(drop=True)
        for ri in range(len(grp)):
            t_wk  = grp.at[ri, 'week_idx']
            t_raw = grp.at[ri, 'weekly_cases']
            ctx   = list(range(t_wk - 2 - HIST + 1, t_wk - 2 + 1))
            steps, sg_vec = [], []
            n_real = 0
            for wk in ctx:
                s, co = sincos(wk)
                key   = (zone, wk)
                if key in full_lookup.index:
                    c_raw = float(full_lookup.loc[key]['weekly_cases'])
                    n_real += 1
                else:
                    c_raw = 0.0
                c_v   = float(enc(c_raw, max_local))
                sg_vec.append(float(enc(sg_lookup.get(wk, 0), MAX_SG)))
                steps.append(np.concatenate([wx(zone,wk), [c_v, s, co]]))
            if n_real == 0: continue
            X_loc.append(np.array(steps, dtype=np.float32))
            X_sg.append(np.array(sg_vec, dtype=np.float32))
            y.append(float(enc(t_raw, max_local)))
            meta.append({'SUBZONE_N': zone, 'actual_raw': t_raw,
                         'area_km2': grp.at[ri,'area_km2'], 'year': grp.at[ri,'year'],
                         'iso_week': int(grp.at[ri,'week_idx'] - grp.at[ri,'year']*52)})
    return (np.array(X_loc,np.float32), np.array(X_sg,np.float32),
            np.array(y,np.float32), pd.DataFrame(meta).reset_index(drop=True))

# ── Models ────────────────────────────────────────────────────────────────────
class SingleTransformer(nn.Module):
    def __init__(self, n_feat, n_steps, d=64, heads=4, layers=2, drop=0.3):
        super().__init__()
        self.proj    = nn.Linear(n_feat, d)
        self.pos_enc = nn.Parameter(torch.zeros(1, n_steps, d))
        enc_l = nn.TransformerEncoderLayer(d, heads, 128, drop, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc_l, layers)
        self.head = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32,1))
    def forward(self, x):
        h = self.proj(x) + self.pos_enc
        return self.head(self.enc(h).mean(1)).squeeze(-1)

class DualTransformer(nn.Module):
    def __init__(self, n_local, d=64, heads=4, layers=2, drop=0.3):
        super().__init__()
        self.proj    = nn.Linear(n_local, d)
        self.pos_enc = nn.Parameter(torch.zeros(1, HIST, d))
        enc_l = nn.TransformerEncoderLayer(d, heads, 128, drop, batch_first=True)
        self.enc    = nn.TransformerEncoder(enc_l, layers)
        self.sg_mlp = nn.Sequential(nn.Linear(HIST,32), nn.ReLU(), nn.Dropout(drop))
        self.head   = nn.Sequential(nn.Linear(d+32,32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32,1))
    def forward(self, x_loc, x_sg):
        h = self.proj(x_loc) + self.pos_enc
        h = self.enc(h).mean(1)
        return self.head(torch.cat([h, self.sg_mlp(x_sg)], 1)).squeeze(-1)

# ── Training helpers ──────────────────────────────────────────────────────────
class DS1(Dataset):
    def __init__(self,X,y): self.X,self.y=torch.FloatTensor(X),torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i],self.y[i]

class DS3(Dataset):
    def __init__(self,Xl,Xs,y):
        self.Xl,self.Xs,self.y=torch.FloatTensor(Xl),torch.FloatTensor(Xs),torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.Xl[i],self.Xs[i],self.y[i]

def wmse(p,t):
    return ((1+8*t)*(p-t)**2).mean()

def fit_single(model, X, y):
    cut = int(len(y)*0.85)
    tr = DataLoader(DS1(X[:cut],y[:cut]), BS, shuffle=True)
    vl = DataLoader(DS1(X[cut:],y[cut:]), BS)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
    best, bst, wait = float('inf'), None, 0
    for ep in range(EPOCHS):
        model.train()
        for Xb,yb in tr:
            opt.zero_grad(); wmse(model(Xb),yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl_l = sum(wmse(model(Xb),yb).item() for Xb,yb in vl)/len(vl)
        sched.step(vl_l)
        if vl_l<best: best,bst,wait=vl_l,copy.deepcopy(model.state_dict()),0
        else:
            wait+=1
            if wait>=PATIENCE:
                print(f"    Early stop ep{ep+1} val={best:.4f}"); break
        if (ep+1)%20==0: print(f"    ep{ep+1:3d} val={vl_l:.4f}")
    model.load_state_dict(bst); return model

def fit_dual(model, Xl, Xs, y):
    cut = int(len(y)*0.85)
    tr = DataLoader(DS3(Xl[:cut],Xs[:cut],y[:cut]), BS, shuffle=True)
    vl = DataLoader(DS3(Xl[cut:],Xs[cut:],y[cut:]), BS)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
    best, bst, wait = float('inf'), None, 0
    for ep in range(EPOCHS):
        model.train()
        for Xl_b,Xs_b,yb in tr:
            opt.zero_grad(); wmse(model(Xl_b,Xs_b),yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl_l = sum(wmse(model(Xl_b,Xs_b),yb).item() for Xl_b,Xs_b,yb in vl)/len(vl)
        sched.step(vl_l)
        if vl_l<best: best,bst,wait=vl_l,copy.deepcopy(model.state_dict()),0
        else:
            wait+=1
            if wait>=PATIENCE:
                print(f"    Early stop ep{ep+1} val={best:.4f}"); break
        if (ep+1)%20==0: print(f"    ep{ep+1:3d} val={vl_l:.4f}")
    model.load_state_dict(bst); return model

def pred_single(model, X):
    model.eval()
    out=[]
    with torch.no_grad():
        for Xb,_ in DataLoader(DS1(X,np.zeros(len(X))), BS):
            out.append(model(Xb).numpy())
    return np.clip(np.concatenate(out),0,1)

def pred_dual(model, Xl, Xs):
    model.eval()
    out=[]
    with torch.no_grad():
        for Xl_b,Xs_b,_ in DataLoader(DS3(Xl,Xs,np.zeros(len(Xl))), BS):
            out.append(model(Xl_b,Xs_b).numpy())
    return np.clip(np.concatenate(out),0,1)

def metrics(preds_raw, meta):
    actual = meta['actual_raw'].values.astype(float)
    mae  = np.mean(np.abs(preds_raw - actual))
    ss_r = np.sum((actual-preds_raw)**2); ss_t=np.sum((actual-actual.mean())**2)
    r2   = 1-ss_r/ss_t if ss_t>0 else float('nan')
    pb, ab = np.zeros(len(meta),int), np.zeros(len(meta),int)
    for i in range(len(meta)):
        z      = meta.at[i, 'SUBZONE_N']
        iso_wk = int(meta.at[i, 'year']) * 52   # derive iso_week from week_idx
        # week_idx is not stored in meta — recompute from year col if available
        # Use the iso_week stored directly in df_active for this row
        thr = get_nea_thr(z, meta.at[i, 'iso_week'])
        pb[i] = int(preds_raw[i] > thr)
        ab[i] = int(meta.at[i, 'actual_raw'] > thr)
    tp=((pb==1)&(ab==1)).sum(); fp=((pb==1)&(ab==0)).sum(); fn=((pb==0)&(ab==1)).sum()
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    f1=2*p*r/(p+r) if p+r else 0
    return mae, r2, p, r, f1, int(tp), int(fp), int(fn)

# ── Build sequences (done once, shared across versions) ───────────────────────
print("\nBuilding sequences...")
X1_tr,y1_tr,_       = build_v1(df_tr)
X1_te,y1_te,meta_te = build_v1(df_te)

X2_tr,y2_tr,_        = build_v2(df_tr)
X2_te,y2_te,meta_te2 = build_v2(df_te)

Xl3_tr,Xs3_tr,y3_tr,_       = build_v3(df_tr)
Xl3_te,Xs3_te,y3_te,meta_te3 = build_v3(df_te)

print(f"  V1 — train:{len(X1_tr)} test:{len(X1_te)} feat:{X1_tr.shape[2]}")
print(f"  V2 — train:{len(X2_tr)} test:{len(X2_te)} feat:{X2_tr.shape[2]} steps:{X2_tr.shape[1]}")
print(f"  V3 — train:{len(Xl3_tr)} test:{len(Xl3_te)} local_feat:{Xl3_tr.shape[2]}")

# ── Train & evaluate ──────────────────────────────────────────────────────────
results = []

for label, build_fn in [
    ('V1 (6-hist, single-stream)',       None),
    ('V2 (6-hist+2-forecast, single)',   None),
    ('V3 (6-hist, dual-stream)',         None),
]:
    print(f"\n{'='*58}\n  {label}\n{'='*58}")

    if label.startswith('V1'):
        model = SingleTransformer(X1_tr.shape[2], HIST)
        model = fit_single(model, X1_tr, y1_tr)
        preds = dec(pred_single(model, X1_te), max_local)
        mae,r2,p,r,f1,tp,fp,fn = metrics(preds, meta_te.reset_index(drop=True))

    elif label.startswith('V2'):
        model = SingleTransformer(X2_tr.shape[2], HIST+FORE)
        model = fit_single(model, X2_tr, y2_tr)
        preds = dec(pred_single(model, X2_te), max_local)
        mae,r2,p,r,f1,tp,fp,fn = metrics(preds, meta_te2.reset_index(drop=True))

    else:
        model = DualTransformer(Xl3_tr.shape[2])
        model = fit_dual(model, Xl3_tr, Xs3_tr, y3_tr)
        preds = dec(pred_dual(model, Xl3_te, Xs3_te), max_local)
        mae,r2,p,r,f1,tp,fp,fn = metrics(preds, meta_te3.reset_index(drop=True))

    print(f"  MAE={mae:.1f}  R²={r2:.3f}")
    print(f"  Area×K  Prec={p:.1%}  Rec={r:.1%}  F1={f1:.1%}  TP={tp} FP={fp} FN={fn}")
    results.append({'Model':label,'MAE':round(mae,1),'R²':round(r2,3),
                    'Prec':f'{p:.1%}','Rec':f'{r:.1%}','F1':f'{f1:.1%}'})

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*58}\nSUMMARY\n{'='*58}")
print(pd.DataFrame(results).to_string(index=False))
