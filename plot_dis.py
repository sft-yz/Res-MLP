# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional: use gaussian_kde if available
try:
    from scipy.stats import gaussian_kde
    HAVE_GAUSSIAN_KDE = True
except Exception:
    HAVE_GAUSSIAN_KDE = False

from scipy.interpolate import interp1d

# ========== config ==========
metrics_files = {
    "MLP":      "mlp.csv",
    "MHA-MLP":  "mha.csv",
    "Res-MLP":  "res.csv",
}

# colors (true + per-model pastel + stronger line colors)
color_true = "#7FB3FF"
colors_model = {
    "MLP":     "#ff9b9b",
    "MHA-MLP": "#9bd99b",
    "Res-MLP": "#ffd79b"
}
line_colors = {
    "true": "blue",
    "MLP":  "#e65b5b",
    "MHA-MLP":"#2e8b57",
    "Res-MLP":"#ff8c1a"
}

alpha_bar = 0.85
alpha_line = 0.95
linewidth = 2.0

# plotting bins / kernel grid
BINS = 30
XS = None  # will compute based on combined range; if you prefer fixed range, set numpy linspace here
SMOOTH_SIGMA_BINS = max(1.0, BINS * 0.02)

# timestamp parse format (None = infer); if you know exact format (e.g. "%Y%m%d.%H%M") set it here to speed parsing
TIMESTAMP_FORMAT = None

# ========== helpers ==========
def safe_read_columns(path, need_cols):
    """Read only needed columns from CSV; return DataFrame with index either parsed timestamp or RangeIndex."""
    hdr = pd.read_csv(path, nrows=0)
    cols = list(hdr.columns)
    # pick case-sensitive actual column names
    chosen = []
    for key in need_cols:
        # find exact-match ignoring case
        for c in cols:
            if c.lower() == key.lower():
                chosen.append(c)
                break
        else:
            chosen.append(None)
    # ensure at least pred_涡间距 exists
    if chosen[2] is None:
        raise ValueError(f"文件 {path} 中找不到 'pred_vortex_dist_km' 列，现有列为: {cols}")
    usecols = [c for c in chosen if c is not None]
    df = pd.read_csv(path, usecols=usecols)
    # parse timestamp if present
    tscol = chosen[0]
    if tscol is not None:
        if TIMESTAMP_FORMAT:
            df['__ts__'] = pd.to_datetime(df[tscol], format=TIMESTAMP_FORMAT, errors='coerce')
        else:
            df['__ts__'] = pd.to_datetime(df[tscol], errors='coerce', infer_datetime_format=True)
        if df['__ts__'].isna().all():
            # fallback to RangeIndex
            df.index = pd.RangeIndex(len(df))
        else:
            df = df.set_index('__ts__')
    else:
        df.index = pd.RangeIndex(len(df))
    # standardize column names in returned df
    colmap = {}
    if chosen[1] is not None:
        colmap[chosen[1]] = 'true_vortex_dist_km'
    colmap[chosen[2]] = 'pred_vortex_dist_km'
    df = df.rename(columns=colmap)
    # convert numeric columns to float32
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    return df

def kde_or_fallback(vals, xs, bins_edges):
    """Return density values on xs for vals. Use gaussian_kde if available, else smooth hist + interp."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return np.zeros_like(xs)
    if HAVE_GAUSSIAN_KDE:
        try:
            kde = gaussian_kde(vals)
            y = kde(xs)
            return y
        except Exception:
            pass
    # fallback
    hist, ed = np.histogram(vals, bins=bins_edges, density=True)
    centers_hist = 0.5 * (ed[:-1] + ed[1:])
    sigma_bins = max(1.0, len(ed)-1 * 0.02)
    win = int(max(3, math.ceil(SMOOTH_SIGMA_BINS * 6)))
    kernel_x = np.arange(-win, win+1)
    kernel = np.exp(-0.5 * (kernel_x / SMOOTH_SIGMA_BINS)**2)
    kernel = kernel / kernel.sum()
    smooth = np.convolve(hist, kernel, mode='same')
    interp = interp1d(centers_hist, smooth, bounds_error=False, fill_value=0.0)
    return interp(xs)

def summary_stats_paired(true_arr, pred_arr):
    t = np.asarray(true_arr, dtype=float)
    p = np.asarray(pred_arr, dtype=float)
    mask = ~np.isnan(t) & ~np.isnan(p)
    n = mask.sum()
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    t2 = t[mask]; p2 = p[mask]
    bias = np.mean(p2 - t2)
    rmse = math.sqrt(np.mean((p2 - t2)**2))
    corr = np.corrcoef(t2, p2)[0,1] if t2.size>1 else np.nan
    return bias, rmse, corr, n

# ========== read each file (only needed columns) ==========
# Need columns: ['timestamp', 'true_涡间距', 'pred_涡间距'] in that order for safe_read_columns
per_model_df = {}
for name, path in metrics_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path} (model label: {name})")
    df = safe_read_columns(path, ['timestamp', 'true_vortex_dist_km', 'pred_vortex_dist_km'])
    per_model_df[name] = df

# choose global true series: prefer first file that has true_涡间距 non-empty
true_series = None
true_from = None
for name, df in per_model_df.items():
    if 'true_vortex_dist_km' in df.columns and df['true_vortex_dist_km'].notna().sum() > 0:
        true_series = df['true_vortex_dist_km'].dropna()
        true_from = name
        break
if true_series is None:
    raise ValueError("在任何输入文件中未找到 non-empty 的 'true_vortex_dist_km' 列，请检查 CSV。")

# try to align by timestamp if possible: use intersection of datetime indices when available
indices = []
for df in per_model_df.values():
    indices.append(df.index)
# detect datetime-like
is_dt = [pd.api.types.is_datetime64_any_dtype(idx) for idx in indices]
if any(is_dt):
    # intersection of all datetime indices that are datetime; otherwise fallback to truncate
    inter = true_series.index
    for name, df in per_model_df.items():
        if pd.api.types.is_datetime64_any_dtype(df.index):
            inter = inter.intersection(df.index)
    if len(inter) > 0:
        # reindex true and each pred to inter
        true_aligned = true_series.reindex(inter)
        preds = {}
        for name, df in per_model_df.items():
            preds[name] = df['pred_vortex_dist_km'].reindex(inter)
    else:
        # fallback truncate
        use_truncate = True
else:
    use_truncate = True

if 'use_truncate' in locals() and use_truncate:
    # positional truncate to min length among true and preds
    lengths = [true_series.size] + [df['pred_vortex_dist_km'].dropna().size for df in per_model_df.values()]
    L = min([l for l in lengths if l>0])
    if L == 0:
        raise ValueError("有效数据长度不足用于绘图（某些文件可能没有有效 涡间距）。")
    true_aligned = pd.Series(true_series.to_numpy()[:L], index=pd.RangeIndex(L))
    preds = {}
    for name, df in per_model_df.items():
        arr = df['pred_vortex_dist_km'].to_numpy(dtype='float32', copy=False)
        preds[name] = pd.Series(arr[:L], index=pd.RangeIndex(L))

# now we have true_aligned and preds dict
models = list(metrics_files.keys())

# compute combined range for histogram/xs
all_vals = [v.to_numpy(dtype=float) for v in [true_aligned] + [preds[m] for m in models]]
all_concat = np.concatenate([a[~np.isnan(a)] for a in all_vals]) if len(all_vals)>0 else np.array([])
if all_concat.size == 0:
    raise ValueError("没有可用的 涡间距 值用于绘图（均为 NaN）。")
vmin = np.percentile(all_concat, 0.5)
vmax = np.percentile(all_concat, 99.5)
if vmin == vmax:
    vmin = all_concat.min()
    vmax = all_concat.max()
# expand a little
pad = (vmax - vmin) * 0.02 if vmax>vmin else 0.1
vmin -= pad; vmax += pad

edges = np.linspace(vmin, vmax, BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
xs = np.linspace(vmin, vmax, 1000)

# ========== plotting: 1x3 ==========
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

for ax, model in zip(axes, models):
    # true and pred arrays (drop NaN for histogram/kde separately)
    t_all = true_aligned.to_numpy(dtype=float)
    p_all = preds[model].to_numpy(dtype=float)

    # histograms (density)
    t_in = t_all[(~np.isnan(t_all)) & (t_all >= vmin) & (t_all <= vmax)]
    p_in = p_all[(~np.isnan(p_all)) & (p_all >= vmin) & (p_all <= vmax)]

    hist_t, _ = np.histogram(t_in, bins=edges, density=True)
    hist_p, _ = np.histogram(p_in, bins=edges, density=True)

    width = centers[1] - centers[0]
    ax.bar(centers, hist_p, width=width * 0.92, align='center',
           color=colors_model[model], alpha=0.5, edgecolor='white', linewidth=0.35, label=model, zorder=1)
    ax.bar(centers, hist_t, width=width * 0.92, align='center',
           color=color_true, alpha=0.85, edgecolor='white', linewidth=0.35, label='True', zorder=2)

    # KDE overlay
    # compute KDE (or fallback) for true and pred
    y_t = kde_or_fallback(t_in, xs, edges)
    y_p = kde_or_fallback(p_in, xs, edges)
    ax.plot(xs, y_t, lw=linewidth, color=line_colors['true'], alpha=alpha_line, label='True KDE')
    ax.plot(xs, y_p, lw=linewidth, color=line_colors[model], alpha=alpha_line, linestyle='--', label=f'{model} KDE')

    # annotate stats (paired)
    bias, rmse, corr, npaired = summary_stats_paired(t_all, p_all)

    ax.set_xlim(vmin, vmax)
    ax.set_xlabel('vortex_dist(km)')
    ax.set_ylabel('Density')
    ax.set_title(f"True vs {model} ")
    ax.grid(alpha=0.12)
    ax.legend(fontsize=12, loc='upper left')
    ax.text(0.02, 0.75, f"bias={bias:.3f}\nρ={corr:.3f}",
            transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
out = "涡间距.png"
# fig.suptitle("涡间距 distributions: True vs Model (bars + KDE)", fontsize=14)
fig.savefig(out, dpi=200)
print(f"已保存: {out}")
plt.show()
