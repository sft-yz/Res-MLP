
import os
import math
import csv
import datetime
import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs

from joblib import load
from skimage.metrics import structural_similarity as ssim
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch

# ============ 常量 ============
FRAME_POINTS = 5400
H = 30
W = 180
LAT_MIN = 60.0
LAT_MAX = 89.0
LON_MIN = 0.0
LON_MAX = 358.0

K_geo = 4
K_mlt = 2
K_lat = 2

B_START = "20240122.0200"
B_END = "20240122.1000"

DATA_DIR = "24"
OUTPUT_DIR = "ab图"
MODEL_PATHS = ["mlp_24/best_model.keras", "mha_24/best_model.keras", "Res_24/best_model.keras"]
SCALER_PATHS = ["mlp_24/scaler.joblib"]
GLOBAL_RANGE_PATH = "mlp_24/global_pot_range.npy"

TIME_POINT_LIST = ['20240122.0200', '20240122.0642','20240122.0738','20240122.0852','20240122.0930']
PREDICT_BATCH_SIZE = 4096

def label_to_uthour(lbl):
    try:
        hh = int(lbl.split(".")[1][:2])
        mm = int(lbl.split(".")[1][2:4])
        return hh + mm / 60.0
    except Exception:
        return 0.0


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1r, lat1r, lon2r, lat2r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

def ts_to_dt(ts):
    return datetime.datetime.strptime(ts, "%Y%m%d.%H%M")

def dt_to_ts(dt):
    return dt.strftime("%Y%m%d.%H%M")

def build_complete_timeline(start_t, end_t, step_minutes=2):
    timeline = []
    step = datetime.timedelta(minutes=step_minutes)
    t = start_t
    while t <= end_t:
        timeline.append(t)
        t += step
    return timeline

def load_and_preprocess_data(data_dir):

    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_Match_Omni_SD_Pot.mat")])
    timestamp_to_frame = {}
    total_frames = 0

    for fname in all_files:
        fpath = os.path.join(data_dir, fname)
        try:
            mat = scipy.io.loadmat(fpath, verify_compressed_data_integrity=False)
        except Exception as e:
            print(f"加载 {fpath} 失败: {e}，跳过")
            continue

        pot_all = mat.get("Match_Omni_SD_Pot")
        if pot_all is None:
            print(f"文件 {fpath} 中未找到变量 Match_Omni_SD_Pot，跳过")
            continue

        frames = pot_all.shape[0] // FRAME_POINTS
        date_str = os.path.basename(fname).split("_")[0]

        for fi in range(frames):
            blk = pot_all[fi * FRAME_POINTS:(fi + 1) * FRAME_POINTS, :]

            try:
                hour, minute = int(blk[0, 0]), int(blk[0, 1])
            except Exception:
                continue

            try:
                full_t = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(hours=hour, minutes=minute)
                ts = full_t.strftime("%Y%m%d.%H%M")
            except Exception:
                ts = f"{date_str}.{hour:02d}{minute:02d}"

            try:
                pot_vals = blk[:, 12].astype(np.float32)
            except Exception:
                continue

            if pot_vals.size != FRAME_POINTS:
                continue

            pot_grid = pot_vals.reshape(H, W)
            feat0 = blk[0, 2:8].astype(np.float32)
            ut_hour_frame = float(hour) + float(minute) / 60.0

            timestamp_to_frame[ts] = {
                "pot": pot_grid,
                "feat0": feat0,
                "ut_hour": ut_hour_frame,
            }
            total_frames += 1

    print(f"加载完成：共读取帧数={total_frames}, 时间点数={len(timestamp_to_frame)}")
    return timestamp_to_frame

lon_grid_static = np.tile(np.linspace(LON_MIN, LON_MAX, W), H)
lat_grid_static = np.repeat(np.linspace(LAT_MIN, LAT_MAX, H), W)

lon_grid_rad_static = np.deg2rad(lon_grid_static)
lat_rad_static = np.deg2rad(lat_grid_static)

geo_fourier_static = np.column_stack(
    [np.sin(k * lon_grid_rad_static) for k in range(1, K_geo + 1)] +
    [np.cos(k * lon_grid_rad_static) for k in range(1, K_geo + 1)]
)

lat_fourier_static = np.column_stack(
    [np.sin(k * lat_rad_static) for k in range(1, K_lat + 1)] +
    [np.cos(k * lat_rad_static) for k in range(1, K_lat + 1)]
)


def predict_frame_from_feat0(model, scaler, feat0, ut_hour):
    mlt_deg = ((lon_grid_static / 15.0) + ut_hour) % 24 * 15.0
    mlt_rad = np.deg2rad(mlt_deg)

    mlt_fourier = np.column_stack(
        [np.sin(k * mlt_rad) for k in range(1, K_mlt + 1)] +
        [np.cos(k * mlt_rad) for k in range(1, K_mlt + 1)]
    )

    positions_frame = np.column_stack((
        lat_grid_static.reshape(-1, 1),
        geo_fourier_static,
        mlt_fourier,
        lat_fourier_static
    )).astype(np.float32)

    feat0_arr = np.array(feat0, dtype=np.float32).reshape(1, -1)
    global_rep = np.tile(feat0_arr, (FRAME_POINTS, 1))
    X_frame = np.hstack((global_rep, positions_frame)).astype(np.float32)

    X_frame_norm = scaler.transform(X_frame)
    pred_norm = model.predict(X_frame_norm, batch_size=PREDICT_BATCH_SIZE, verbose=0).flatten()

    return pred_norm.reshape(H, W)

def compute_metrics(pred, true, global_data_range=None):
    mask = ~np.isnan(true)
    diff = (pred - true)[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size > 0 else float("nan")

    try:
        data_range = global_data_range if global_data_range is not None else (np.nanmax(true) - np.nanmin(true))
        ssim_v = ssim(true, pred, data_range=float(data_range))
    except Exception:
        ssim_v = float("nan")

    try:
        t1 = true[mask].flatten()
        p1 = pred[mask].flatten()
        if t1.size > 1 and np.std(t1) > 0 and np.std(p1) > 0:
            lc = float(np.corrcoef(t1, p1)[0, 1])
        else:
            lc = float("nan")
    except Exception:
        lc = float("nan")

    try:
        cpcp_val = float(np.nanmax(pred) - np.nanmin(pred)) / 1000.0
    except Exception:
        cpcp_val = float("nan")

    try:
        idx_max = np.unravel_index(np.nanargmax(pred), pred.shape)
        idx_min = np.unravel_index(np.nanargmin(pred), pred.shape)

        lon_max = idx_max[1] * (360.0 / (W - 1))
        lat_max = LAT_MIN + idx_max[0] * ((LAT_MAX - LAT_MIN) / (H - 1))
        lon_min = idx_min[1] * (360.0 / (W - 1))
        lat_min = LAT_MIN + idx_min[0] * ((LAT_MAX - LAT_MIN) / (H - 1))

        vd = haversine_km(lon_max, lat_max, lon_min, lat_min)
    except Exception:
        vd = float("nan")

    return rmse, ssim_v, lc, cpcp_val, vd


def rotate_grid_clockwise(grid, rotate_deg):
    if grid is None:
        return grid
    if (abs(rotate_deg) % 360.0) == 0:
        return grid
    shift_cols = int((rotate_deg / 360.0) * W)
    return np.roll(grid, -shift_cols, axis=1)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("正在加载数据...")
    ts_map = load_and_preprocess_data(DATA_DIR)

    missing = [t for t in TIME_POINT_LIST if t not in ts_map]
    if missing:
        print(f"警告：以下时间点在数据中未找到，将跳过：{missing}")

    used_timepoints = [t for t in TIME_POINT_LIST if t in ts_map]
    if len(used_timepoints) == 0:
        print("未找到任何指定时间点，退出。")
        return

    global_min, global_max = np.load(GLOBAL_RANGE_PATH)

    # ---------- 模型与 scaler ----------
    print("加载模型与 scaler...")
    models = []
    for mp in MODEL_PATHS:
        try:
            models.append(tf.keras.models.load_model(mp))
        except Exception as e:
            print(f"加载模型 {mp} 失败: {e}")
            models.append(None)

    if len(SCALER_PATHS) == 1:
        single_scaler = load(SCALER_PATHS[0])
        scalers = [single_scaler] * len(MODEL_PATHS)
    else:
        scalers = [load(sp) for sp in SCALER_PATHS]
        while len(scalers) < len(MODEL_PATHS):
            scalers.append(scalers[-1])

    all_preds = {mp: [] for mp in MODEL_PATHS}
    truths = []
    metrics_rows = []

    ROTATE_DEG = 90.0
    for t in used_timepoints:
        true_grid = ts_map[t]["pot"]
        true_grid_rot = rotate_grid_clockwise(true_grid, ROTATE_DEG)
        truths.append(true_grid_rot)

        feat0 = ts_map[t]["feat0"]
        ut_hour = ts_map[t]["ut_hour"]

        for i, mp in enumerate(MODEL_PATHS):
            model = models[i]
            scaler = scalers[i]

            if model is None or scaler is None:
                pred_grid = np.full_like(true_grid, np.nan)
            else:
                pred_norm = predict_frame_from_feat0(model, scaler, feat0, ut_hour)
                pred_grid = (pred_norm * (global_max - global_min) + global_min).astype(np.float32)

            pred_grid_rot = rotate_grid_clockwise(pred_grid, ROTATE_DEG)
            all_preds[mp].append(pred_grid_rot)

    all_grids = []
    for j in range(len(used_timepoints)):
        all_grids.append(truths[j])
        for mp in MODEL_PATHS:
            all_grids.append(all_preds[mp][j])

    v_vmin = float(np.nanmin(all_grids))
    v_vmax = float(np.nanmax(all_grids))
    print("全局色标范围:", v_vmin, "～", v_vmax)

    ncols = len(used_timepoints)
    nrows = 4
    a_height = 3.6 * nrows
    b_height = 8
    total_height = a_height + b_height

    fig = plt.figure(figsize=(4.2 * ncols, total_height))
    fig.subplots_adjust(left=0.04, right=0.9, top=0.995, bottom=0.03)

    outer_gs = GridSpec(2, 1, height_ratios=[a_height, b_height], figure=fig, hspace=0.08)
    a_sub_gs = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_gs[0], wspace=0.05, hspace=0.3)

    axes = []

    # ---------- a 图：真实图 ----------
    for j, t in enumerate(used_timepoints):
        ax = fig.add_subplot(
            a_sub_gs[0, j],
            projection=ccrs.NorthPolarStereo(central_longitude=(-15 - 15 * label_to_uthour(t)) % 360)
        )

        lon_base = np.linspace(LON_MIN, LON_MAX, W)
        lon_cyclic = np.append(lon_base, 360.0)
        true_data_cyclic = np.concatenate([truths[j], truths[j][:, 0:1]], axis=1)

        lat_vals = np.linspace(LAT_MIN, LAT_MAX, H)
        pole_value = np.mean(true_data_cyclic[-1, :])
        pole_row = np.full((1, true_data_cyclic.shape[1]), pole_value)
        true_data_filled = np.concatenate([true_data_cyclic, pole_row], axis=0)
        lat_filled = np.append(lat_vals, 90.0)

        ax.set_extent([0, 360, 60, 90], crs=ccrs.PlateCarree())
        ax.coastlines(color="black", alpha=0.8, linewidth=0.8, zorder=1)

        ax.pcolormesh(
            lon_cyclic, lat_filled, true_data_filled,
            transform=ccrs.PlateCarree(),
            shading="gouraud",
            vmin=np.nanmin(truths[j]),
            vmax=np.nanmax(truths[j]),
            edgecolors="none",
            zorder=2,
            alpha=0.9
        )

        try:
            ax.contour(
                lon_cyclic, lat_filled, true_data_filled,
                levels=np.linspace(np.nanmin(truths[j]), np.nanmax(truths[j]), 8)[1:-1],
                transform=ccrs.PlateCarree(),
                colors="black",
                linestyles="--",
                linewidths=1,
                zorder=3
            )
        except Exception:
            pass

        try:
            idx_max = np.unravel_index(np.nanargmax(truths[j]), truths[j].shape)
            idx_min = np.unravel_index(np.nanargmin(truths[j]), truths[j].shape)

            lon_max = idx_max[1] * (360.0 / (W - 1))
            lat_max = LAT_MIN + idx_max[0] * ((LAT_MAX - LAT_MIN) / (H - 1))
            lon_min = idx_min[1] * (360.0 / (W - 1))
            lat_min = LAT_MIN + idx_min[0] * ((LAT_MAX - LAT_MIN) / (H - 1))

            ax.plot(lon_max, lat_max, "k+", markersize=6, markeredgewidth=1.5,
                    transform=ccrs.PlateCarree(), zorder=4)
            ax.plot(lon_min, lat_min, "rx", markersize=6, markeredgewidth=1.5,
                    transform=ccrs.PlateCarree(), zorder=4)
        except Exception:
            pass

        _, _, _, true_cpcp, true_vd = compute_metrics(truths[j], truths[j], global_data_range=(global_max - global_min))
        ax.set_title(f"SuperDARN {t}", fontsize=12)
        ax.text(0.5, -0.08, f"CPCP={true_cpcp:.1f} kV  Vortex={true_vd:.1f}km",
                transform=ax.transAxes, ha="center", va="top", fontsize=12)

        axes.append(ax)

    # ---------- a 图：三个模型预测 ----------
    model_names = [os.path.splitext(os.path.basename(mp))[0] for mp in MODEL_PATHS]

    for i, mp in enumerate(MODEL_PATHS):
        preds = all_preds[mp]
        for j in range(ncols):
            ax = fig.add_subplot(
                a_sub_gs[i + 1, j],
                projection=ccrs.NorthPolarStereo(central_longitude=(-15 - 15 * label_to_uthour(used_timepoints[j])) % 360)
            )

            pred = preds[j]

            lon_base = np.linspace(LON_MIN, LON_MAX, W)
            lon_cyclic = np.append(lon_base, 360.0)
            pred_data_cyclic = np.concatenate([pred, pred[:, 0:1]], axis=1)

            lat_vals = np.linspace(LAT_MIN, LAT_MAX, H)
            pole_value = np.mean(pred_data_cyclic[-1, :])
            pole_row = np.full((1, pred_data_cyclic.shape[1]), pole_value)
            pred_data_filled = np.concatenate([pred_data_cyclic, pole_row], axis=0)
            lat_filled = np.append(lat_vals, 90.0)

            ax.set_extent([0, 360, 60, 90], crs=ccrs.PlateCarree())
            ax.coastlines(color="black", alpha=0.8, linewidth=0.8, zorder=1)

            ax.pcolormesh(
                lon_cyclic, lat_filled, pred_data_filled,
                transform=ccrs.PlateCarree(),
                shading="gouraud",
                vmin=np.nanmin(pred),
                vmax=np.nanmax(pred),
                edgecolors="none",
                zorder=2,
                alpha=0.9
            )

            try:
                ax.contour(
                    lon_cyclic, lat_filled, pred_data_filled,
                    levels=np.linspace(np.nanmin(pred), np.nanmax(pred), 8)[1:-1],
                    transform=ccrs.PlateCarree(),
                    colors="black",
                    linestyles="--",
                    linewidths=1,
                    zorder=3
                )
            except Exception:
                pass

            try:
                idx_max = np.unravel_index(np.nanargmax(pred), pred.shape)
                idx_min = np.unravel_index(np.nanargmin(pred), pred.shape)

                lon_max = idx_max[1] * (360.0 / (W - 1))
                lat_max = LAT_MIN + idx_max[0] * ((LAT_MAX - LAT_MIN) / (H - 1))
                lon_min = idx_min[1] * (360.0 / (W - 1))
                lat_min = LAT_MIN + idx_min[0] * ((LAT_MAX - LAT_MIN) / (H - 1))

                ax.plot(lon_max, lat_max, "k+", markersize=6, markeredgewidth=1.5,
                        transform=ccrs.PlateCarree(), zorder=4)
                ax.plot(lon_min, lat_min, "rx", markersize=6, markeredgewidth=1.5,
                        transform=ccrs.PlateCarree(), zorder=4)
            except Exception:
                pass

            rmse, ssim_v, lc, cpcp_val, vd = compute_metrics(pred, truths[j], global_data_range=(global_max - global_min))
            txt = f"RMSE={rmse:.1f}  SSIM={ssim_v:.2f}  LC={lc:.2f}\nCPCP={cpcp_val:.1f}  Vortex={vd:.1f}km"

            ax.set_title(f"{model_names[i]} {used_timepoints[j]}", fontsize=12)
            ax.text(0.5, -0.05, txt, transform=ax.transAxes, ha="center", va="top", fontsize=12)

            axes.append(ax)
            metrics_rows.append({
                "time": used_timepoints[j],
                "model": model_names[i],
                "RMSE": rmse,
                "SSIM": ssim_v,
                "LC": lc,
                "CPCP_kV": cpcp_val,
                "Vortex_km": vd,
            })

    # ---------- a 图分隔线 ----------
    rows_axes = [axes[r * ncols:(r + 1) * ncols] for r in range(nrows)]
    for r in range(nrows - 1):
        bottoms = [ax.get_position().y0 for ax in rows_axes[r]]
        tops_next = [ax.get_position().y1 for ax in rows_axes[r + 1]]
        bottom_r = min(bottoms)
        top_next = max(tops_next)
        y_mid = (bottom_r + top_next) / 2.0
        y = y_mid - 0.007

        left = min(ax.get_position().x0 for ax in rows_axes[r]) + 0.002
        right = max(ax.get_position().x1 for ax in rows_axes[r]) - 0.002

        fig.add_artist(
            Line2D([left, right], [y, y],
                   transform=fig.transFigure,
                   color="black",
                   linewidth=2,
                   alpha=0.7)
        )

    # ---------- b 图：时间轴 ----------
    start_t = ts_to_dt(B_START) if B_START else ts_to_dt(used_timepoints[0])
    end_t = ts_to_dt(B_END) if B_END else ts_to_dt(used_timepoints[-1])

    key_list = [k for k in sorted(ts_map.keys()) if start_t <= ts_to_dt(k) <= end_t]
    key_dts = [ts_to_dt(k) for k in key_list]

    Bx_list, By_list, Bz_list, Vx_list, Pd_list, AE_list = ([] for _ in range(6))
    for k in key_list:
        f0 = ts_map[k]["feat0"]
        Bx_list.append(float(f0[0]))
        By_list.append(float(f0[1]))
        Bz_list.append(float(f0[2]))
        Vx_list.append(float(f0[3]))
        Pd_list.append(float(f0[4]))
        AE_list.append(float(f0[5]))

    b_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[1], hspace=0.1)
    ax_top = fig.add_subplot(b_gs[0, 0])
    ax_vx = fig.add_subplot(b_gs[1, 0])
    ax_vx_t = ax_vx.twinx()
    ax_ae = fig.add_subplot(b_gs[2, 0])

    shrink_factor = 0.93
    bbox_a = outer_gs[0].get_position(fig)
    a_center_x = bbox_a.x0 + bbox_a.width / 2.0
    new_b_width = bbox_a.width * shrink_factor
    new_b_x0 = a_center_x - new_b_width / 2.0

    for ax in [ax_top, ax_vx, ax_vx_t, ax_ae]:
        pos = ax.get_position()
        ax.set_position([new_b_x0, pos.y0, new_b_width, pos.height])

    # 三个参数曲线
    markers = ["o", "s", "^"]
    for vals, mk, label in zip([Bx_list, By_list, Bz_list], markers, ["Bx", "By", "Bz"]):
        ax_top.plot(key_dts, np.array(vals), label=label, marker=mk, linewidth=1.5, markersize=4)

    ax_top.set_ylabel("IMF (nT)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper left", ncol=3, fontsize=12)

    c_vx = "tab:blue"
    c_pd = "tab:orange"

    ax_vx.plot(key_dts, np.array(Vx_list), label="Vx", marker="o", linewidth=1.2, markersize=4, color=c_vx)
    ax_vx.set_ylabel("Vx", color=c_vx)
    ax_vx.tick_params(axis="y", colors=c_vx)

    ax_vx_t.plot(key_dts, np.array(Pd_list), label="Pd", marker="^", linestyle="--",
                 linewidth=1.2, markersize=4, color=c_pd)
    ax_vx_t.set_ylabel("Pd", color=c_pd)
    ax_vx_t.tick_params(axis="y", colors=c_pd)

    lines_a, labels_a = ax_vx.get_legend_handles_labels()
    lines_b, labels_b = ax_vx_t.get_legend_handles_labels()
    if lines_a or lines_b:
        ax_vx.legend(lines_a + lines_b, labels_a + labels_b, loc="upper left")

    ax_ae.plot(key_dts, np.array(AE_list), label="AE", marker="o", linewidth=1.2, markersize=4)
    ax_ae.set_ylabel("AE")
    ax_ae.grid(True, alpha=0.3)
    ax_ae.legend(loc="upper left")

    selected_dts = [ts_to_dt(t) for t in used_timepoints]

    b_mark_dts = []
    for dt in selected_dts:
        if dt in key_dts:
            b_mark_dts.append(dt)
        else:
            nearest_idx = min(
                range(len(key_dts)),
                key=lambda i: abs((key_dts[i] - dt).total_seconds())
            )
            b_mark_dts.append(key_dts[nearest_idx])

    for dt in b_mark_dts:
        for ax in (ax_top, ax_ae, ax_vx):
            ax.axvline(dt, color="k", linestyle="--", linewidth=1.5, alpha=0.6)

    # ---------- a 图最后一行文字与 b 图竖线之间的连接线 ----------
    try:
        last_row_axes = rows_axes[-1]
        fig.canvas.draw()

        for j, ax_map in enumerate(last_row_axes):
            if j >= len(b_mark_dts):
                continue

            dt = b_mark_dts[j]

            con = ConnectionPatch(
                xyA=(0.5, -0.19), coordsA=ax_map.transAxes,
                xyB=(mdates.date2num(dt), 1.0), coordsB=ax_top.get_xaxis_transform(),
                axesA=ax_map, axesB=ax_top,
                color="0.15",
                linewidth=1.4,
                alpha=0.95,
                zorder=7
            )
            fig.add_artist(con)

    except Exception as e:
        print(f"画连接线时出错（可忽略）：{e}")

    # b 图横轴刻度
    tick_dts = [start_t + (end_t - start_t) * frac for frac in np.linspace(0.0, 1.0, 5)]
    tick_labels = [dt.strftime("%Y.%m%d %H:%M") for dt in tick_dts]

    for ax in (ax_top, ax_vx):
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", labelbottom=False)

    ax_ae.set_xticks(tick_dts)
    ax_ae.set_xticklabels(tick_labels, fontsize=12)
    ax_ae.set_xlabel("Time (UTC)", fontsize=12, labelpad=8)

    for ax in (ax_top, ax_vx, ax_ae):
        ax.set_xlim(start_t, end_t)

    # ---------- 标注 a / b ----------
    a_frac = a_height / total_height
    fig.text(0.02, 0.99, "(a)", fontsize=16, fontweight="bold", va="top")
    b_label_y = 1.0 - a_frac
    fig.text(0.02, b_label_y, "(b)", fontsize=16, fontweight="bold", va="top")

    # ---------- 保存合成图 ----------
    out_png = os.path.join(OUTPUT_DIR, "combined_4x5_with_timeseries.pdf")
    try:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"合成图已保存: {out_png}")
    except Exception as e:
        print(f"保存合成图失败: {e}")
    finally:
        plt.close(fig)

    # ---------- 保存指标表 ----------
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    fieldnames = ["time", "model", "RMSE", "SSIM", "LC", "CPCP_kV", "Vortex_km"]
    try:
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in metrics_rows:
                writer.writerow(r)
        print(f"指标 CSV 已保存: {csv_path}")
    except Exception as e:
        print(f"写入 CSV 失败: {e}")


if __name__ == "__main__":
    main()