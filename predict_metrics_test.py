import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ncpu = os.cpu_count() or 4
os.environ['OMP_NUM_THREADS'] = str(min(ncpu, 8))
os.environ['MKL_NUM_THREADS'] = str(min(ncpu, 8))

import numpy as np
import scipy.io
import datetime
import tensorflow as tf

ncpu = os.cpu_count() or 4
tf.config.threading.set_intra_op_parallelism_threads(min(ncpu, 8))
tf.config.threading.set_inter_op_parallelism_threads(2)

from tensorflow.keras import models
from joblib import load, dump
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from scipy.io import savemat
import cartopy.crs as ccrs 
import xlsxwriter                 # 写sheet xlsx
import multiprocessing
import time

# 尝试导入地磁场库（若不可用则使用退化常数）
try:
    from geomag import geomag as gm_module
    GEO_AVAILABLE = True
except Exception:
    gm_module = None
    GEO_AVAILABLE = False

# ============ 常量、傅里叶参数 ============
K_geo = 4
K_mlt = 2
K_lat = 2
FRAME_POINTS = 5400
H = 30
W = 180
LAT_MIN = 60.0
LAT_MAX = 89.0
LON_MIN = 0.0
LON_MAX = 358.0

MODEL_PATH = '/root/autodl-tmp/my_project/code/Res.keras'     # 模型文件路径
SCALER_PATH = '/root/autodl-tmp/my_project/code/scaler.joblib'       # scaler 路径
GLOBAL_RANGE_NPY = '/root/autodl-tmp/my_project/code/global_pot_range.npy'  # global range npy
DATA_DIR = '/root/autodl-tmp/my_project/24test'                            # 数据目录
OUTPUT_DIR = '/root/autodl-tmp/my_project/Res_test'                      # 输出目录
PLOT_EVERY = 360   
N_FILES = None    
SAVE_MAT = False  # 是否保存 mat

# 支持字符串或数值输入；推荐使用字符串
START_TS = '20241011.0930'   
END_TS   = '20241231.2358'   

# === MODIFIED ===
CHUNK_SIZE = 16            
PREDICT_BATCH_SIZE = 2048 

# multiprocessing pool config for plotting
PLOT_POOL_WORKERS = max(1, multiprocessing.cpu_count()//4)  # 可调整

# ============ 数据加载与预处理 ============
def load_and_preprocess_data(data_dir, n_files=None):
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_Match_Omni_SD_Pot.mat')])
    if n_files is not None:
        all_files = all_files[:n_files]
    timestamps = []
    feat_list  = []
    pos_list   = []
    pot_list   = []
    frame_idx_list = []
    ut_list = []
    frame_ut_hours = []

    total_frames = 0
    for fname in all_files:
        fpath = os.path.join(data_dir, fname)
        try:
            mat = scipy.io.loadmat(fpath, verify_compressed_data_integrity=False)
        except Exception as e:
            print(f'加载 {fpath} 失败: {e}, 跳过')
            continue
        pot_all = mat.get('Match_Omni_SD_Pot')
        if pot_all is None:
            print(f'文件 {fpath} 中未找到变量 Match_Omni_SD_Pot，跳过')
            continue
        frames = pot_all.shape[0] // FRAME_POINTS
        date_str = os.path.basename(fname).split('_')[0]
        for fi in range(frames):
            blk = pot_all[fi*FRAME_POINTS:(fi+1)*FRAME_POINTS, :]
            try:
                hour, minute = int(blk[0,0]), int(blk[0,1])
            except Exception as e:
                print(f'帧解析 hour/min 失败 in {fname}, frame {fi}: {e}, 跳过此帧')
                continue
            try:
                full_t = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(hours=hour, minutes=minute)
                ts = full_t.strftime("%Y%m%d.%H%M")
            except Exception:
                ts = f"{date_str}.{hour:02d}{minute:02d}"
            global_fi = total_frames
            total_frames += 1
            timestamps.append(ts)
            frame_ut_hours.append(float(hour) + float(minute)/60.0)

            feat0 = blk[0, 2:8].astype(np.float32)  # Bx,By,Bz,Vx,Pd,AE
            ut_hour_frame = float(hour) + float(minute)/60.0

            for p in range(FRAME_POINTS):
                feat_list.append(feat0)
                pos_list.append(blk[p, 10:12].astype(np.float32))
                pot_list.append(float(blk[p, 12]))
                frame_idx_list.append(global_fi)
                ut_list.append(ut_hour_frame)

    if len(feat_list) == 0:
        raise RuntimeError('没有从 data_dir 中加载到任何帧，请检查文件和变量名')

    features = np.vstack(feat_list).astype(np.float32)
    pos_arr = np.vstack(pos_list).astype(np.float32)
    ut_arr = np.array(ut_list, dtype=np.float32)
    pot_values = np.array(pot_list, dtype=np.float32)
    frame_indices = np.array(frame_idx_list, dtype=np.int32)

    print(f"加载完毕：总帧数={total_frames}, 总样本点={features.shape[0]}")
    return timestamps, frame_ut_hours, features, pos_arr, pot_values, frame_indices


def build_position_features_from_posarr(pos_arr, ut_arr):
    lats = pos_arr[:, 0]
    lons = pos_arr[:, 1]
    lons_rad = np.deg2rad(lons)
    geo_fourier = np.column_stack(
        [np.sin(k * lons_rad) for k in range(1, K_geo + 1)] +
        [np.cos(k * lons_rad) for k in range(1, K_geo + 1)]
    )
    mlt_deg = ((lons / 15.0) + ut_arr) % 24 * 15.0
    mlt_rad = np.deg2rad(mlt_deg)
    mlt_fourier = np.column_stack(
        [np.sin(k * mlt_rad) for k in range(1, K_mlt + 1)] +
        [np.cos(k * mlt_rad) for k in range(1, K_mlt + 1)]
    )
    lats_rad = np.deg2rad(lats)
    lat_fourier = np.column_stack(
        [np.sin(k * lats_rad) for k in range(1, K_lat + 1)] +
        [np.cos(k * lats_rad) for k in range(1, K_lat + 1)]
    )
    positions = np.column_stack((lats.reshape(-1,1), geo_fourier, mlt_fourier, lat_fourier)).astype(np.float32)
    return positions


# ============ 更精确的封装函数（替换脚本中简化版本） ============
#  ---------- 梯度 ----------
def periodic_grad(pred, lats_deg, lons_deg):
    H_, W_ = pred.shape
    if len(lats_deg) > 1:
        dlat = np.deg2rad(lats_deg[1] - lats_deg[0])
    else:
        dlat = np.deg2rad(1.0)
    if len(lons_deg) > 1:
        dlon = np.deg2rad((lons_deg[1] - lons_deg[0]))
    else:
        dlon = np.deg2rad(1.0)
    dV_dlat = np.empty_like(pred)
    if H_ >= 3:
        dV_dlat[1:-1, :] = (pred[2:, :] - pred[:-2, :]) / (2.0 * dlat)
        dV_dlat[0, :] = (pred[1, :] - pred[0, :]) / dlat
        dV_dlat[-1, :] = (pred[-1, :] - pred[-2, :]) / dlat
    else:
        dV_dlat[:, :] = 0.0
    dV_dlon = (np.roll(pred, -1, axis=1) - np.roll(pred, 1, axis=1)) / (2.0 * dlon)
    R_E = 6.371e6
    dV_dlat_m = dV_dlat / R_E
    cos_lat = np.cos(np.deg2rad(lats_deg)).reshape(-1, 1)
    cos_lat_safe = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    dV_dlon_m = dV_dlon / (R_E * cos_lat_safe)
    return dV_dlat_m, dV_dlon_m


#  ---------- 地磁场助手：获取局部地磁场（返回 Tesla 单位） ----------
def get_geomag_grid(lat_grid, lon_grid, alt_km=300.0, date_obj=None):
    if date_obj is None:
        date_obj = datetime.date.today()
        print('geomag 日期解析失败，使用当天日期:', date_obj)
    H_ = len(lat_grid)
    W_ = len(lon_grid)
    Bn = np.zeros((H_, W_), dtype=np.float64)
    Be = np.zeros((H_, W_), dtype=np.float64)
    Bv = np.zeros((H_, W_), dtype=np.float64)
    if GEO_AVAILABLE and gm_module is not None:
        gm = gm_module.GeoMag()
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                try:
                    g = gm.GeoMag(lat, lon, alt_km, time=date_obj)
                    Bn[i, j] = float(g.bx) * 1e-9
                    Be[i, j] = float(g.by) * 1e-9
                    Bv[i, j] = float(g.bz) * 1e-9
                except Exception:
                    print(f'使用典型地磁')
                    Bn[i, j] = 0.0
                    Be[i, j] = 0.0
                    Bv[i, j] = 3e-5
    else:
        print(f'使用典型地磁')
        Bn[:, :] = 0.0
        Be[:, :] = 0.0
        Bv[:, :] = 3e-5
    return Bn, Be, Bv


# === MODIFIED ===
# compute_speed: 修改为可接受预计算的地磁网格（Bn,Be,Bv）
def compute_speed(pred_pot, alt_km=300.0, timestamp=None, date_obj=None, Bn=None, Be=None, Bv=None):
    """
    如果有 Bn,Be,Bv（预计算的），直接使用；否则按 timestamp/date_obj 调用 get_geomag_grid。
    返回 speed (H,W) 和 direction (H,W)
    """
    # 解析 date_obj if needed
    if Bn is None or Be is None or Bv is None:
        if date_obj is None and timestamp is not None:
            try:
                date_part = timestamp.split('.')[0]
                year  = int(date_part[:4])
                month = int(date_part[4:6])
                day   = int(date_part[6:8])
                date_obj = datetime.date(year, month, day)
            except Exception:
                date_obj = None
        # 生成地磁场网格（可能慢，建议外部缓存）
        lats_1d = np.linspace(LAT_MIN, LAT_MAX, pred_pot.shape[0])
        lons_1d = np.linspace(LON_MIN, LON_MAX, pred_pot.shape[1])
        Bn, Be, Bv = get_geomag_grid(lats_1d, lons_1d, alt_km=alt_km, date_obj=date_obj)

    # 计算梯度
    lats = np.linspace(LAT_MIN, LAT_MAX, pred_pot.shape[0])
    lons = np.linspace(LON_MIN, LON_MAX, pred_pot.shape[1])
    dV_dlat_m, dV_dlon_m = periodic_grad(pred_pot, lats, lons)
    E_north = -dV_dlat_m
    E_east  = -dV_dlon_m

    B_sq = (Bn**2 + Be**2 + Bv**2)
    B_sq = np.where(B_sq < 1e-25, 1e-25, B_sq)

    v_north = (E_east * Bv) / B_sq
    v_east = (-E_north * Bv) / B_sq

    speed = np.hypot(v_east, v_north)
    direction_v = np.degrees(np.arctan2(v_east, v_north))

    return speed, direction_v


#  ---------- 计算CPCP和涡间距 ----------
def calculate_cpcp_and_vortex_distance(potential, lats, lons):
    pot = np.array(potential, dtype=np.float64)
    if np.all(np.isnan(pot)):
        return np.nan, np.nan
    try:
        max_flat = np.nanargmax(pot)
        min_flat = np.nanargmin(pot)
    except ValueError:
        return np.nan, np.nan
    max_idx = np.unravel_index(int(max_flat), pot.shape)
    min_idx = np.unravel_index(int(min_flat), pot.shape)
    cpcp = float(pot[max_idx] - pot[min_idx])
    lat1, lon1 = float(lats[max_idx[0]]), float(lons[max_idx[1]])
    lat2, lon2 = float(lats[min_idx[0]]), float(lons[min_idx[1]])
    lat1_rad, lon1_rad = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2_rad, lon2_rad = np.deg2rad(lat2), np.deg2rad(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
    a = np.minimum(1.0, np.maximum(0.0, a))
    c = 2.0 * np.arcsin(np.sqrt(a))
    R = 6371.0
    distance_km = float(R * c)
    return cpcp, distance_km

# 小工具：更宽松的 START/END TS 解析（支持字符串或数值）
def parse_nullable_ts(ts):
    """
    接受 None / 'YYYYMMDD.HHMM' / numeric (e.g. 20241011.0930) 并返回 datetime 或 None
    推荐直接把 START_TS/END_TS 写成字符串 'YYYYMMDD.HHMM'
    """
    if ts is None:
        return None
    if isinstance(ts, str):
        # try 'YYYYMMDD.HHMM' then 'YYYYMMDDHHMM'
        try:
            return datetime.datetime.strptime(ts, "%Y%m%d.%H%M")
        except Exception:
            try:
                return datetime.datetime.strptime(ts, "%Y%m%d%H%M")
            except Exception:
                raise RuntimeError(f"TS 字符串格式错误，应为 'YYYYMMDD.HHMM'，你输入了: {ts}")
    if isinstance(ts, (int, float)):
        # numeric like 20241011.0930 -> date=20241011, frac*10000 -> HHMM
        try:
            date_int = int(ts)
            frac = abs(ts - date_int)
            hhmm = int(round(frac * 10000))
            date_str = f"{date_int:08d}"
            time_str = f"{hhmm:04d}"
            return datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
        except Exception:
            raise RuntimeError(f"TS 数值解析失败，请改为字符串格式 'YYYYMMDD.HHMM'，你输入了: {ts}")
    raise RuntimeError(f"无法识别 TS 类型：{type(ts)}，请使用字符串或 None")

# ============ 主流程 ============
def main():
    # === MODIFIED ===: 使用脚本顶部变量而不是 argparse
    model_path = MODEL_PATH
    scaler_path = SCALER_PATH
    global_range_npy = GLOBAL_RANGE_NPY
    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR
    plot_every = PLOT_EVERY
    n_files = N_FILES
    save_mat = SAVE_MAT
    start_ts = START_TS
    end_ts = END_TS
    chunk_size = CHUNK_SIZE
    predict_batch_size = PREDICT_BATCH_SIZE
    plot_pool_workers = PLOT_POOL_WORKERS
    # === MODIFIED === end

    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, 'pngs')
    os.makedirs(img_dir, exist_ok=True)

    print('🔍 使用主脚本的第一步加载数据（按帧展开）...')
    timestamps, frame_ut_hours, features, pos_arr, pot_values, frame_indices = load_and_preprocess_data(data_dir, n_files=n_files)

    # 把 timestamps 解析为 datetime 对象，便于筛选时间段
    ts_dt = []
    for ts in timestamps:
        try:
            dt = datetime.datetime.strptime(ts, "%Y%m%d.%H%M")
        except Exception:
            try:
                date_part, hm = ts.split('.')
                dt = datetime.datetime.strptime(date_part + hm, "%Y%m%d%H%M")
            except Exception:
                dt = None
        ts_dt.append(dt)

    # === MODIFIED === parse start/end using parse_nullable_ts
    start_dt = parse_nullable_ts(start_ts)
    end_dt = parse_nullable_ts(end_ts)
    # === MODIFIED === end

    # 找到在时间范围内的 frame 索引（按 frame 的 timestamp 列表）
    frames_in_range = []
    for fi, dt in enumerate(ts_dt):
        if dt is None:
            continue
        if start_dt is not None and dt < start_dt:
            continue
        if end_dt is not None and dt > end_dt:
            continue
        frames_in_range.append(fi)
    if len(frames_in_range) == 0:
        print("⚠️ 没有找到落在 START_TS/END_TS 范围内的帧（或所有帧的时间解析失败）。处理全部帧。")
        unique_frames = np.unique(frame_indices)
    else:
        unique_frames = np.array(frames_in_range, dtype=np.int32)
        print(f"🕒 将只处理时间范围内的帧: {len(unique_frames)} frames, from {frames_in_range[0]} to {frames_in_range[-1]}")

    per_sample_ut = np.array([frame_ut_hours[idx] for idx in frame_indices])
    positions = build_position_features_from_posarr(pos_arr, per_sample_ut)
    X_all = np.hstack((features, positions))

    print('🔁 加载 scaler、global_range、模型...')
    scaler = load(scaler_path)
    global_min, global_max = np.load(global_range_npy)
    model = tf.keras.models.load_model(model_path)

    # === MODIFIED ===: TF threading tweak (适用于 CPU)
    ncpu = os.cpu_count() or 4

    # === MODIFIED === end

    # 预构造 lat/lon grid （静态）
    lon_grid_static = np.tile(np.linspace(LON_MIN, LON_MAX, W), H)
    lat_grid_static = np.repeat(np.linspace(LAT_MIN, LAT_MAX, H), W)
    lon_grid_rad_static = np.deg2rad(lon_grid_static)
    lat_rad_static = np.deg2rad(lat_grid_static)
    geo_fourier_static = np.column_stack(
        [np.sin(k * lon_grid_rad_static) for k in range(1, K_geo+1)] +
        [np.cos(k * lon_grid_rad_static) for k in range(1, K_geo+1)]
    )
    lat_fourier_static = np.column_stack(
        [np.sin(k * lat_rad_static) for k in range(1, K_lat+1)] +
        [np.cos(k * lat_rad_static) for k in range(1, K_lat+1)]
    )

    # 缓存 mlt 与 geomag（按 ut_hour 或日期）
    mlt_cache = {}
    geomag_cache = {}

    # 预构造 lat/lon 1d for geomag
    lats_1d = np.linspace(LAT_MIN, LAT_MAX, H)
    lons_1d = np.linspace(LON_MIN, LON_MAX, W)

    # 以 chunked prediction 方式处理帧（=== MODIFIED ===）
    metrics = []
    unique_frames_sorted = np.sort(unique_frames)
    n_frames = len(unique_frames_sorted)

    # pool for plotting
    plot_pool = multiprocessing.Pool(processes=plot_pool_workers)

    t_all_start = time.time()
    for start in range(0, n_frames, chunk_size):
        chunk_frames = unique_frames_sorted[start:start+chunk_size]
        X_chunk_list = []
        true_pot_list = []
        feat0_list = []
        ts_list = []
        frame_idxs_list = []

        for frame_idx in chunk_frames:
            mask = (frame_indices == frame_idx)
            ts = timestamps[frame_idx]
            feat0 = features[mask][0]
            ut_hour = frame_ut_hours[frame_idx]

            # mlt 四ier 缓存
            ut_key = round(float(ut_hour), 4)
            if ut_key in mlt_cache:
                mlt_fourier = mlt_cache[ut_key]
            else:
                mlt_deg = ((lon_grid_static / 15.0) + ut_hour) % 24 * 15.0
                mlt_rad = np.deg2rad(mlt_deg)
                mlt_fourier = np.column_stack(
                    [np.sin(k * mlt_rad) for k in range(1, K_mlt+1)] +
                    [np.cos(k * mlt_rad) for k in range(1, K_mlt+1)]
                )
                mlt_cache[ut_key] = mlt_fourier

            positions_frame = np.column_stack((lat_grid_static.reshape(-1,1),
                                              geo_fourier_static,
                                              mlt_fourier,
                                              lat_fourier_static)).astype(np.float32)
            global_rep = np.tile(feat0, (FRAME_POINTS, 1))
            X_frame = np.hstack((global_rep, positions_frame))
            X_chunk_list.append(X_frame)

            true_pot_list.append(pot_values[mask].reshape(H, W))
            feat0_list.append(feat0)
            ts_list.append(ts)
            frame_idxs_list.append(frame_idx)

        # 合并并归一化
        X_chunk = np.vstack(X_chunk_list).astype(np.float32)
        X_chunk_norm = scaler.transform(X_chunk)

        t_pred0 = time.time()
        pred_chunk_norm = model.predict(X_chunk_norm, batch_size=predict_batch_size, verbose=0)
        t_pred = time.time() - t_pred0

        # 拆分回每帧并计算指标（保留你原有的指标计算）
        offset = 0
        for i, frame_idx in enumerate(chunk_frames):
            start_off = offset
            end_off = offset + FRAME_POINTS
            pred_norm = pred_chunk_norm[start_off:end_off].flatten()
            pred_pot = (pred_norm * (global_max - global_min) + global_min).reshape(H, W)
            true_pot = true_pot_list[i]
            ts = ts_list[i]
            feat0 = feat0_list[i]

            # 计算指标
            rmse_v = np.sqrt(np.mean((true_pot - pred_pot)**2))
            try:
                ssim_v = ssim(true_pot, pred_pot, data_range=true_pot.max()-true_pot.min())
            except Exception:
                ssim_v = np.nan
            try:
                lc_v = np.corrcoef(true_pot.flatten(), pred_pot.flatten())[0,1]
            except Exception:
                lc_v = np.nan
            lats = np.linspace(LAT_MIN, LAT_MAX, H)
            lons = np.linspace(LON_MIN, LON_MAX, W)
            true_cpcp, true_vd = calculate_cpcp_and_vortex_distance(true_pot, lats, lons)
            pred_cpcp, pred_vd = calculate_cpcp_and_vortex_distance(pred_pot, lats, lons)

            # geomag cache per date (避免重复大量计算)
            date_obj = None
            try:
                date_part = ts.split('.')[0]
                date_obj = datetime.date(int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8]))
            except Exception:
                date_obj = None

            if date_obj is not None and date_obj in geomag_cache:
                Bn, Be, Bv = geomag_cache[date_obj]
            else:
                Bn, Be, Bv = get_geomag_grid(lats_1d, lons_1d, alt_km=300.0, date_obj=date_obj)
                if date_obj is not None:
                    geomag_cache[date_obj] = (Bn, Be, Bv)

            # 这里调用 compute_speed 时传入预计算的 B grids，加速
            true_speed_mag, true_speed_dir = compute_speed(true_pot, timestamp=ts, date_obj=date_obj, Bn=Bn, Be=Be, Bv=Bv)
            pred_speed_mag, pred_speed_dir = compute_speed(pred_pot, timestamp=ts, date_obj=date_obj, Bn=Bn, Be=Be, Bv=Bv)
            
            try:
                speed_mag_lc = np.corrcoef(true_speed_mag.flatten(), pred_speed_mag.flatten())[0,1]
            except Exception:
                speed_mag_lc = np.nan
            try:
                speed_dir_lc = np.corrcoef(true_speed_dir.flatten(), pred_speed_dir.flatten())[0,1]
            except Exception:
                speed_dir_lc = np.nan

            metrics.append({'timestamp': ts,
                            'RMSE': float(rmse_v), 'SSIM': float(ssim_v) if not np.isnan(ssim_v) else np.nan, 'LC': float(lc_v) if not np.isnan(lc_v) else np.nan,
                            'true_CPCP': float(true_cpcp), 'pred_CPCP': float(pred_cpcp),
                            'true_vortex_dist_km': float(true_vd), 'pred_vortex_dist_km': float(pred_vd),
                            'Speed_Mag_LC': float(speed_mag_lc) if not np.isnan(speed_mag_lc) else np.nan,
                            'Speed_Dir_LC': float(speed_dir_lc) if not np.isnan(speed_dir_lc) else np.nan,
                            'Bx': float(feat0[0]), 'By': float(feat0[1]), 'Bz': float(feat0[2]), 'Vx': float(feat0[3]), 'Pd': float(feat0[4]), 'AE': float(feat0[5])})

            offset = end_off

        processed_to = min(start + chunk_size, n_frames)
        print(f'已处理 {processed_to}/{n_frames} 帧 (chunk predict time {t_pred:.2f}s), 最近 {ts_list[-1]}')

    # 结束 plot pool（等待所有绘图完成）
    plot_pool.close()
    plot_pool.join()

    # 保存 metrics.csv，并在最后一行增加均值（数值列）
    df_metrics = pd.DataFrame(metrics)
    if not df_metrics.empty:
        numeric_means = df_metrics.select_dtypes(include=[np.number]).mean()
        mean_row = {col: '' for col in df_metrics.columns}
        mean_row['timestamp'] = 'MEAN'
        for k, v in numeric_means.items():
            mean_row[k] = float(v)
        # append
        # pandas.append is deprecated in newer versions, but keep semantics:
        df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)

        print("指标均值（数值部分）：")
        print(numeric_means)
    else:
        print("⚠️ df_metrics 为空，未生成任何指标。")

    metrics_csv = os.path.join(output_dir, 'metrics.csv')
    df_metrics.to_csv(metrics_csv, index=False)
    print(f'已保存指标到: {metrics_csv}')

    t_all = time.time() - t_all_start
    print(f"全部处理完，耗时 {t_all:.1f}s")

if __name__ == '__main__':
    main()
