#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import scipy.io
import datetime
import matplotlib
matplotlib.use('Agg')   # headless safe
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pandas as pd

# ========== 用户配置 ==========
DATA_DIR = 'data2024'   
OUTPUT_DIR = '图片/例图'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_POINT = '20240122.1216'
FRAME_POINTS = 5400
H = 30
W = 180
LAT_MIN = 60.0
LAT_MAX = 89.0
LON_MIN = 0.0
LON_MAX = 358.0

BG_LAT_MIN = 50.0
BG_LAT_MAX = 90.0
ROTATE_DEG = 0.0    

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1r, lat1r, lon2r, lat2r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = math.sin(dlat/2)**2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def load_match_pot_dir(data_dir):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('_Match_Omni_SD_Pot.mat')])
    ts_map = {}
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        try:
            mat = scipy.io.loadmat(fpath, verify_compressed_data_integrity=False)
        except Exception as e:
            print(f"无法读取 {fpath}: {e}")
            continue
        pot_all = mat.get('Match_Omni_SD_Pot')
        if pot_all is None:
            print(f"{fpath} 未找到 Match_Omni_SD_Pot，跳过")
            continue
        date_str = os.path.basename(fname).split('_')[0] 
        frames = pot_all.shape[0] // FRAME_POINTS
        for fi in range(frames):
            blk = pot_all[fi*FRAME_POINTS:(fi+1)*FRAME_POINTS, :]
            # 解析 hour/min（与你原脚本一致）
            try:
                hour = int(blk[0, 0])
                minute = int(blk[0, 1])
            except Exception:
                continue
            try:
                dt = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(hours=hour, minutes=minute)
                ts = dt.strftime("%Y%m%d.%H%M")
            except Exception:
                ts = f"{date_str}.{hour:02d}{minute:02d}"
            try:
                pot_vals = blk[:, 12].astype(np.float32)
            except Exception:
                continue
            if pot_vals.size != FRAME_POINTS:
                continue
            pot_grid = pot_vals.reshape(H, W)
            ut_hour = float(hour) + float(minute)/60.0
            ts_map[ts] = {'pot': pot_grid, 'ut_hour': ut_hour}
    return ts_map

def save_potential_table(pot, out_dir, timestamp):
    latitudes = np.linspace(LAT_MIN, LAT_MAX, H)   # 与 pot 行一致，第一行对应 LAT_MIN
    longitudes = np.linspace(LON_MIN, LON_MAX, W)

    df_v = pd.DataFrame(pot, index=[f"{lat:.2f}" for lat in latitudes],
                        columns=[f"{lon:.2f}" for lon in longitudes])
    df_kv = df_v / 1000.0

    outxls = os.path.join(out_dir, f"pot_{timestamp}.xlsx")
    outcsv_v = os.path.join(out_dir, f"pot_{timestamp}_V.csv")
    outcsv_kv = os.path.join(out_dir, f"pot_{timestamp}_kV.csv")
    try:
        df_v.to_csv(outcsv_v, index=True)
        df_kv.to_csv(outcsv_kv, index=True)
    except Exception as e2:

def plot_true_potential(ts_map, timestamp, out_dir):
    if timestamp not in ts_map:
        raise ValueError(f"{timestamp} 未在数据中找到，可用示例: {list(ts_map.keys())[:6]}")
    rec = ts_map[timestamp]
    pot = rec['pot'].astype(np.float32)  
    ut_hour = rec['ut_hour']

    if (abs(ROTATE_DEG) % 360.0) != 0 and W > 0:
        shift_cols = int((ROTATE_DEG / 360.0) * W)  # 例如 ROTATE_DEG=90 时，shift_cols = W/4
              pot = np.roll(pot, -shift_cols, axis=1)

    save_potential_table(pot, out_dir, timestamp)

    pot_kv = pot / 1000.0

    central_lon = (180 - 15 * ut_hour) % 360.0
    fig = plt.figure(figsize=(6.8, 6.8))
    ax = plt.subplot(projection=ccrs.NorthPolarStereo(central_longitude=central_lon))
    ax.set_extent([0, 360, 55, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor='#d9fefc', zorder=0)  
    ax.add_feature(cfeature.LAND, facecolor='#f2ead6', zorder=0)  
    ax.coastlines(color='0.5', linewidth=0.5, zorder=1)  

    vmin = float(np.nanmin(pot_kv))
    vmax = float(np.nanmax(pot_kv))
    mesh = ax.pcolormesh(np.linspace(0, 360, W), np.linspace(60, 90, H), pot_kv,
                         transform=ccrs.PlateCarree(), shading='gouraud', vmin=vmin, vmax=vmax,
                         edgecolors='none', zorder=3, alpha=0.8)

    try:
        idx_max = np.unravel_index(np.nanargmax(pot_kv), pot_kv.shape)
        idx_min = np.unravel_index(np.nanargmin(pot_kv), pot_kv.shape)
        lon_max = idx_max[1] * (360.0 / (W - 1))
        lat_max = LAT_MIN + idx_max[0] * ((LAT_MAX - LAT_MIN) / (H - 1))
        lon_min = idx_min[1] * (360.0 / (W - 1))
        lat_min = LAT_MIN + idx_min[0] * ((LAT_MAX - LAT_MIN) / (H - 1))
        ax.plot(lon_max, lat_max, 'k+', markersize=6, markeredgewidth=1, transform=ccrs.PlateCarree(),
                zorder=4)
        ax.plot(lon_min, lat_min, 'rx', markersize=6, markeredgewidth=1, transform=ccrs.PlateCarree(),
                zorder=4)
    except Exception:
        pass

    dist_km = haversine_km(lon_max, lat_max, lon_min, lat_min)

    try:
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
            levels = np.linspace(vmin, vmax, 9)[1:-1]
            cs = ax.contour(np.linspace(0, 360, W), np.linspace(60, 90, H), pot_kv, levels=levels, transform=ccrs.PlateCarree(),
                            colors='black', linestyles='--', linewidths=0.8, zorder=4)
    except Exception:
        pass

    meridians = (central_lon + np.arange(0, 360, 15)) % 360.0
    for lon in meridians:
        ax.plot([lon, lon], [BG_LAT_MIN-15, BG_LAT_MAX],
                transform=ccrs.PlateCarree(),color='0.4',
                linewidth=0.8, linestyle='--', alpha=0.7, zorder=2)

    lon_line = np.linspace(0, 360, 721)
    for lat_tick in [50,60, 70, 80]:
        if BG_LAT_MIN <= lat_tick <= BG_LAT_MAX:
            ax.plot(lon_line, np.full_like(lon_line, lat_tick), transform=ccrs.PlateCarree(),
                    linewidth=0.8, linestyle='--',color='0.4', alpha=0.7, zorder=2)
            if lat_tick != 50:
                ax.text(10, lat_tick, f"{lat_tick}°N", transform=ccrs.PlateCarree(),
                        fontsize=9, ha='left', va='center', color='0.2', zorder=5)

    ax.plot(lon_max, lat_max, marker='P', color='black', markersize=3,transform=ccrs.PlateCarree(), zorder=6)
    ax.plot(lon_min, lat_min, marker='X', color='red', markersize=3,transform=ccrs.PlateCarree(), zorder=6)
    ax.text(lon_max + 2.5, lat_max + 0.8, 'CPmax', transform=ccrs.PlateCarree(),fontsize=10, ha='left', color='0.2',va='bottom', zorder=7)
    ax.text(lon_min + 2.5, lat_min + 0.8, 'CPmin', transform=ccrs.PlateCarree(),fontsize=10, ha='left', color='red',va='bottom', zorder=7)

    try:
        ax.plot([lon_max, lon_min], [lat_max, lat_min], transform=ccrs.Geodetic(),
                color='k', linewidth=1.2, zorder=8)
    except Exception:
        ax.plot([lon_max, lon_min], [lat_max, lat_min], transform=ccrs.PlateCarree(),
                color='k', linewidth=1.6, zorder=8)

    mid_lon = (lon_max + lon_min) / 2.0
    mid_lat = (lat_max + lat_min) / 2.0
    dx = (lon_min - lon_max)
    dy = (lat_min - lat_max)
    dy_deg = 10.0  
    ang = math.degrees(math.atan2(dy, dx))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180

    ax.text(mid_lon, mid_lat+dy_deg, f"vortex spacing", transform=ccrs.PlateCarree(),
                fontsize=12, color='red',ha='center', va='center', rotation=ang-27, rotation_mode='anchor', zorder=9)

    cax = fig.add_axes([0.95, 0.27, 0.02, 0.60])  # x, y, w, h
    cb = fig.colorbar(mesh, cax=cax, orientation='vertical')
    cb.set_label('Potential (kV)', fontsize=10)

    cpmax_val = float(np.nanmax(pot_kv))
    cpmin_val = float(np.nanmin(pot_kv))
    cpcp_val = cpmax_val - cpmin_val

    lon_max_display = lon_max % 360.0
    lon_min_display = lon_min % 360.0

    info = (
        f"CPmax = {cpmax_val:.2f} kV\n"
       # f"  (lat={lat_max:.2f}°N, lon={lon_max_display:.1f}°)\n"
        f"CPmin = {cpmin_val:.2f} kV\n"
        #f"  (lat={lat_min:.2f}°N, lon={lon_min_display:.1f}°)\n"
        f"CPCP = {cpcp_val:.2f} kV\n"
        f"vortex spacing = {dist_km:.0f} km"
    )

  
    fig.text(0.94, 0.15, info, fontsize=9, ha='left', va='bottom',
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', pad=5), zorder=10)

    fig.text(1.05, 0.105, "Hour (MLT)", fontsize=10, ha='right', va='bottom', zorder=11)

    mlt_labels = [21, 22, 23, 0, 1, 2, 3]
    mlt_positions = [0.125, 0.29, 0.405, 0.51, 0.615, 0.738, 0.895]
    y_pos = 0.08
    for x, mlt in zip(mlt_positions, mlt_labels):
        fig.text(x, y_pos, str(mlt), fontsize=9, ha='center', va='bottom', zorder=13)

    ax.set_title(f"{timestamp} UT", fontsize=12, pad=12)

    outfn = os.path.join(out_dir, f"pot_true_{timestamp}.png")
    plt.savefig(outfn, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {outfn}")

# ========== 主流程 ==========
if __name__ == '__main__':
    print("加载数据 ...")
    ts_map = load_match_pot_dir(DATA_DIR)
    if not ts_map:
        print("未加载到任何数据，请检查 DATA_DIR 和文件格式")
    else:
        if TIME_POINT not in ts_map:
            print(f"警告：指定时间点 {TIME_POINT} 不在数据中。可用示例：{list(ts_map.keys())[:6]}")
        try:
            plot_true_potential(ts_map, TIME_POINT, OUTPUT_DIR)
        except Exception as e:
            print("绘图出错：", e)