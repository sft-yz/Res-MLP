import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

ncpu = os.cpu_count() or 4
os.environ['OMP_NUM_THREADS'] = str(min(ncpu, 8))
os.environ['MKL_NUM_THREADS'] = str(min(ncpu, 8))

import numpy as np
import datetime
import tensorflow as tf
from joblib import load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# ============ 配置项（请按需修改） ============
MODEL_PATH = 'Res_24/best_model.keras'
SCALER_PATH = 'Res_24/scaler.joblib'
GLOBAL_RANGE_NPY = 'Res_24/global_pot_range.npy'
OUTPUT_DIR = 'Res_24/test'

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

PREDICT_BATCH_SIZE = 4096

def plot_predicted_potential(pred_data, timestamp, feature_vec, save_path):
    try:
        ut_hour = float(timestamp.split('.')[1][:2]) + float(timestamp.split('.')[1][2:]) / 60.0
    except Exception:
        ut_hour = 0.0
    central_lon = (-15 - 15 * ut_hour) % 360

    lon_base = np.linspace(LON_MIN, LON_MAX, W)
    lon_cyclic = np.append(lon_base, 360.0)
    pred_data_cyclic = np.concatenate([pred_data, pred_data[:, 0:1]], axis=1)

    lat_vals = np.linspace(LAT_MIN, LAT_MAX, H)
    pole_value = np.mean(pred_data_cyclic[-1, :])
    pole_row = np.full((1, pred_data_cyclic.shape[1]), pole_value)

    pred_data_filled = np.concatenate([pred_data_cyclic, pole_row], axis=0)
    lat_filled = np.append(lat_vals, 90.0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=central_lon))
    ax.set_extent([0, 360, 60, 90], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(lon_cyclic, lat_filled, pred_data_filled,
                         transform=ccrs.PlateCarree(), shading='gouraud',
                         vmin=np.nanmin(pred_data), vmax=np.nanmax(pred_data),
                         edgecolors='none', zorder=2, alpha=0.9)
    try:
        ax.contour(lon_cyclic, lat_filled, pred_data_filled,
                   levels=np.linspace(np.nanmin(pred_data), np.nanmax(pred_data), 8)[1:-1],
                   transform=ccrs.PlateCarree(), colors='black', linestyles='--', linewidths=1, zorder=3)
    except Exception:
        pass

    for lat in (60, 70, 80):
        ax.text(0, lat, f"{lat}°N", transform=ccrs.PlateCarree(), ha='right', va='center', fontsize=10, color='black', zorder=6)

    try:
        idx_max = np.unravel_index(np.nanargmax(pred_data), pred_data.shape)
        idx_min = np.unravel_index(np.nanargmin(pred_data), pred_data.shape)
        lon_max = idx_max[1] * (360.0 / (W - 1))
        lat_max = LAT_MIN + idx_max[0] * ((LAT_MAX - LAT_MIN) / (H - 1))
        lon_min = idx_min[1] * (360.0 / (W - 1))
        lat_min = LAT_MIN + idx_min[0] * ((LAT_MAX - LAT_MIN) / (H - 1))
        ax.plot(lon_max, lat_max, 'k+', markersize=8, markeredgewidth=2.5, transform=ccrs.PlateCarree(), zorder=4)
        ax.plot(lon_min, lat_min, 'rx', markersize=8, markeredgewidth=2.5, transform=ccrs.PlateCarree(), zorder=4)
    except Exception:
        pass

    try:
        cpcp_val = float(np.nanmax(pred_data) - np.nanmin(pred_data))
    except Exception:
        cpcp_val = np.nan
    cpcp_val = cpcp_val / 1000.0
    cpcp_label = f"{cpcp_val:.0f} kV"
    ax.text(0.98, 0.02, cpcp_label,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=30, color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=10)

    try:
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(mesh, cax=cax, orientation='vertical')
    except Exception:
        pass

    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"保存图片失败 ({save_path}): {e}")
    finally:
        plt.close(fig)

def rotate_grid_clockwise(grid, rotate_deg):
    if grid is None:
        return grid
    if (abs(rotate_deg) % 360.0) == 0 or W == 0:
        return grid
    shift_cols = int((rotate_deg / 360.0) * W)
    return np.roll(grid, -shift_cols, axis=1)

# ============ 主流程 ============
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUTPUT_DIR, 'pngs')
    os.makedirs(img_dir, exist_ok=True)

    print("加载 scaler 和 模型 ...")
    scaler = load(SCALER_PATH)
    global_min, global_max = np.load(GLOBAL_RANGE_NPY)
    model = tf.keras.models.load_model(MODEL_PATH)

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

    def label_to_uthour(lbl):
        try:
            hh = int(lbl.split('.')[1][:2])
            mm = int(lbl.split('.')[1][2:4])
            return hh + mm/60.0
        except Exception:
            return 0.0

    #AE大时
    MANUAL_INPUTS = [
         #左上 20240502.1632
        #([4.01800012588501, -6.997000217, 17.7189998626709, -395.5, 27.440000534057617, 1109], label_to_uthour('20240502.1632'), '20240502.1632'),
        ([2.2990000247955322,-8.284000397,16.988000869750977,-394,22.510000228881836,1161],label_to_uthour('20240502.1630'), '20240502.1630'),

        # 上中 20241217.0532
        #([-6.592000008, -0.828000009, 12.706999778747559, -507.7999878, 42.380001068115234, 1323], label_to_uthour('20241217.0532'), '20241217.0532'),
        ([-3.59800005,0.30799999833106995,10.38700008392334,-897.4000244,4.820000171661377,1350],label_to_uthour('20240511.2120'), '20240511.212'),

        # 右上 20240516.0642
        #([4.7779998779296875, 10.758000373840332, 12.63599967956543, -439.2999878, 11.039999961853027, 1352], label_to_uthour('20240516.0642'), '20240516.0642'),
        #([-0.610000014,35.62300109863281,60.7599983215332,-757.2000122,24.520000457763672,1207],label_to_uthour('20240510.2238'), '20240510.2238'),
        ([17.17799, 5.51900, 4.1360, -693, 7.48000, 1044], label_to_uthour('20241011.0956'), '20241011.0956'),

        # 左中 20240502.1554
        #([7.136000156402588, -15.96199989, -0.379000008, -398.1000061, 30.3799991607666, 1325], label_to_uthour('20240502.1554'), '20240502.1554'),
        ([3.8440001010894775,-15.57999992,-0.68900001,-422.5,9.670000076293945,1523],label_to_uthour(' 20240502.1946'), ' 20240502.1946'),

        # 右中 20240511.0030
        #([4.006999969482422, 37.132999420166016, -1.764999986, -757.4000244, 31.229999542236328, 1131], label_to_uthour('20240511.0030'), '20240511.0030'),
        ([ 10.61299991607666,29.45599937438965,-0.86500001,-725.5999756,30.950000762939453,1146],label_to_uthour('20240511.0822'), '20240511.0822'),

        # 左下 20240502.1512
        #([8.041000366210938, -13.16800022, -15.26399994, -388.8999939, 23.3799991607666, 1105], label_to_uthour('20240502.1512'), '20240502.1512'),
        #([4.373000144958496,-11.24600029,-10.66899967,-335.6000061,57.15999984741211,1036],label_to_uthour('20240303.1354'), '20240303.1354'),
        ([5.519999980926514,-11.96800041,-16.51499939,-399.7000122,27.1299991607666,1449],label_to_uthour('20240502.1446'), '20240502.1446'),

        # 下中 20240502.1402
        #([6.145999908447266, 0.8740000128746033, -18.90299988, -345.5, 13.210000038146973, 923], label_to_uthour('20240502.1402'), '20240502.1402'),
        ([4.195000171661377,-0.869000018,-15.12699986,-334.5,41.54999923706055,1005],label_to_uthour('20240303.1406'), '20240303.1406'),
        # -0.549000025    0.40700000524520874 - 15.84700012 - 329.8999939    37.689998626708984    1564  20240303.1442

        # 右下 20240511.0710
        #([-21.27700043, 18.91200065612793, -33.18399811, -683.5999756, 41.52000045776367, 1195], label_to_uthour('20240511.0710'), '20240511.0710'),
        ([3.4519999027252197,8.755000114440918,-17.45499992,-393,24.219999313354492,1120],label_to_uthour('20240502.1422'), '20240502.1422'),

    ]
    #AE小时
    MANUAL_INPUTS_small = [
        # 左上
        #([2.760999917984009, -10.64099979, 13.21399974822998, -436.6000061, 9.869999885559082, 70],label_to_uthour('20240813.0536'), '20240813.0536'),
        #([-3.007999897,-10.30799961,9.109999656677246,-459.5,9.489999771118164,107],label_to_uthour('20241006.1404'), '20241006.1404'),
        ([0.04699999, -5.7010, 9.949999, -668.2999878, 4.71000003, 117], label_to_uthour('20241011.2358'), '20241011.2358'),

        # 上
        #([-10.64099979, 0.050999999046325684, 14.654999732971191, -418.8999939, 5.059999942779541, 57],label_to_uthour('20240813.0936'), '20240813.0936'),
        ([3.747999906539917,-0.939999998,16.190000534057617,-505.7999878,10.239999771118164,75],label_to_uthour('20241006.1450'), '20241006.1450'),

        # 右上
        #([-9.277999878, 4.939000129699707, 9.524999618530273, -409.2000122, 4.559999942779541, 67],label_to_uthour('20240813.1300'), '20240813.1300'),
        ([5.491000175476074,7.673999786376953,11.581000328063965,-434.2000122,10.25,46],label_to_uthour('20241006.1346'), '20241006.1346'),

        # 左 (2024-07-30 10:50)
        #([-6.900000095, -6.618000031, -1.050999999, -467.8999939, 25.6200008392334, 535],label_to_uthour('20240730.1050'), '20240730.1050'),
        ([-6.066999912,-11.50199986,-0.504999995,-510.8999939,11.229999542236328,106],label_to_uthour('20241006.1010'), '20241006.1010'),

        # 右
        #([1.3489999771118164, 8.211999893188477, 0.43700000643730164, -432.2999878, 4.769999980926514, 59],label_to_uthour('20240427.1140'), '20240427.1140'),
        ([11.60200023651123,9.623000144958496,-0.250999987,-424.1000061,10.920000076293945,97],label_to_uthour('20241006.1312'), '20241006.1312'),

        # 左下
        #([-3.105999947, -4.986999989, -13.26799965, -526.2000122, 10.670000076293945, 53],label_to_uthour('20241006.1546'), '20241006.1546'),
        ([-5.72300005,-7.589000225,-9.723999977,-513.0999756,9.970000267028809,89],label_to_uthour('20241006.1544'), '20241006.1544'),

        # 下
        #([-2.585999966, -0.082000002, -9.166999817, -332.1000061, 9.529999732971191, 81],label_to_uthour('20240626.0648'), '20240626.0648'),
        ([-2.977999926,-0.800000012,-12.1239996,-341.6000061,19.489999771118164,168],label_to_uthour('20240321.1556'), '20240321.1556'),

        # 右下
        #([-2.444000006, 6.326000213623047, -6.756000042, -349.0, 36.970001220703125, 88],label_to_uthour('20240321.0614'), '20240321.0614'),
        ([-0.536000013,5.0920000076293945,-10.2869997,-350.8999939,33.939998626708984,90],label_to_uthour('20240321.0618'), '20240321.0618'),
    ]

    ENABLE_INTERACTIVE_INPUT = False  # True 则在控制台交互输入（每次一组），按回车两次结束

    def predict_from_feat0(feat0, ut_hour, out_label):
        mlt_deg = ((lon_grid_static / 15.0) + ut_hour) % 24 * 15.0
        mlt_rad = np.deg2rad(mlt_deg)
        mlt_fourier = np.column_stack(
            [np.sin(k * mlt_rad) for k in range(1, K_mlt+1)] +
            [np.cos(k * mlt_rad) for k in range(1, K_mlt+1)]
        )
        positions_frame = np.column_stack((
            lat_grid_static.reshape(-1,1),
            geo_fourier_static,
            mlt_fourier,
            lat_fourier_static
        )).astype(np.float32)

        feat0_arr = np.array(feat0, dtype=np.float32).reshape(1,-1)
        global_rep = np.tile(feat0_arr, (FRAME_POINTS, 1))
        X_frame = np.hstack((global_rep, positions_frame)).astype(np.float32)

        X_frame_norm = scaler.transform(X_frame)
        pred_norm = model.predict(X_frame_norm, batch_size=PREDICT_BATCH_SIZE, verbose=0).flatten()
        pred_pot = (pred_norm * (global_max - global_min) + global_min).reshape(H, W)
        ROTATE_DEG = 90.0
        pred_pot = rotate_grid_clockwise(pred_pot, ROTATE_DEG)

        out_png = os.path.join(img_dir, f"{out_label}.png")
        plot_predicted_potential(pred_pot, out_label, feat0_arr.flatten(), out_png)
        print(f"[手动预测] 已保存: {out_png}")
        return pred_pot

    preds = []
    labels = []
    print("开始处理 MANUAL_INPUTS 中的手动预测 ...")
    for feat0, ut_hour, label in MANUAL_INPUTS:
        try:
            p = predict_from_feat0(feat0, ut_hour, label)
            preds.append(p)
            labels.append(label)
        except Exception as e:
            print(f"手动预测 {label} 失败: {e}")

    def plot_8_panel(predictions, labels_list, save_path):
        n = len(predictions)
        if n == 0:
            print("没有可绘制的预测数据。")
            return

        cmap = plt.get_cmap()

        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 3, figure=fig, wspace=0.05, hspace=0.05)

        positions = [ (0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2) ]

        lon_base = np.linspace(LON_MIN, LON_MAX, W)
        lon_cyclic = np.append(lon_base, 360.0)

        lat_vals = np.linspace(LAT_MIN, LAT_MAX, H)
        lat_filled = np.append(lat_vals, 90.0)  # 加入 90°N

        lon2d, lat2d = np.meshgrid(lon_cyclic, lat_filled)

        for i, ((r, c), lbl, pred) in enumerate(zip(positions, labels_list, predictions)):
            try:
                ut = label_to_uthour(lbl)
            except Exception:
                ut = 0.0
            central_lon = (-15 - 15 * ut) % 360

            ax = fig.add_subplot(gs[r, c], projection=ccrs.NorthPolarStereo(central_longitude=central_lon))
            ax.set_extent([0, 360, 60, 90], crs=ccrs.PlateCarree())

            local_vmin = np.nanmin(pred)
            local_vmax = np.nanmax(pred)

            pred_cyclic = np.concatenate([pred, pred[:, 0:1]], axis=1)

            pole_value = np.mean(pred_cyclic[-1, :])
            pole_row = np.full((1, pred_cyclic.shape[1]), pole_value)
            pred_filled = np.concatenate([pred_cyclic, pole_row], axis=0)

            ax.coastlines(color='black', alpha=0.8, zorder=1)

            try:
                mesh = ax.pcolormesh(lon2d, lat2d, pred_filled,
                                     transform=ccrs.PlateCarree(), shading='gouraud',
                                     vmin=local_vmin, vmax=local_vmax, cmap=cmap,
                                     edgecolors='none', zorder=2, alpha=0.9)
            except Exception as e:
                print(f"绘制子图失败 ({lbl}): {e}")
                mesh = None

            try:
                cpcp_val = (np.nanmax(pred) - np.nanmin(pred)) / 1000.0  # kV
                cpcp_label = f"{cpcp_val:.0f} kV"
                ax.text(0.98, 0.02, cpcp_label,
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=13, color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                        zorder=10)
            except Exception:
                pass

            try:
                cs = ax.contour(lon_cyclic, lat_filled, pred_filled,
                                    levels=np.linspace(np.nanmin(pred), np.nanmax(pred), 9)[1:-1],
                                    transform=ccrs.PlateCarree(), colors='black', linestyles='--', linewidths=1,
                                    zorder=3)
            except Exception as e:
                print(f"[等势线] 子图 {lbl} 未绘制等高线或出错: {e}")

            try:
                idx_max = np.unravel_index(np.nanargmax(pred), pred.shape)
                idx_min = np.unravel_index(np.nanargmin(pred), pred.shape)
                lon_max = idx_max[1] * (360.0 / (W - 1))
                lat_max = LAT_MIN + idx_max[0] * ((LAT_MAX - LAT_MIN) / (H - 1))
                lon_min = idx_min[1] * (360.0 / (W - 1))
                lat_min = LAT_MIN + idx_min[0] * ((LAT_MAX - LAT_MIN) / (H - 1))
                ax.plot(lon_max, lat_max, 'k+', markersize=6, markeredgewidth=2.0, transform=ccrs.PlateCarree(), zorder=4)
                ax.plot(lon_min, lat_min, 'rx', markersize=6, markeredgewidth=2.0, transform=ccrs.PlateCarree(), zorder=4)
            except Exception:
                pass

        try:
            axc = fig.add_subplot(gs[1, 1])
            axc.set_xlim(-1.2, 1.2)
            axc.set_ylim(-1.2, 1.2)
            axc.set_aspect('equal', adjustable='box')

            axc.spines['top'].set_visible(False)
            axc.spines['right'].set_visible(False)
            axc.spines['bottom'].set_visible(False)
            axc.spines['left'].set_visible(False)
            axc.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

            arrow_kw = dict(arrowstyle='<->', linewidth=2.0, color='black', shrinkA=0, shrinkB=0)

            axc.annotate('', xy=(1.0, 0.0), xytext=(-1.0, 0.0), arrowprops=arrow_kw)   # 横向
            axc.annotate('', xy=(0.0, 1.0), xytext=(0.0, -1.0), arrowprops=arrow_kw)   # 纵向
            axc.annotate('', xy=(0.707, 0.707), xytext=(-0.707, -0.707), arrowprops=arrow_kw)  # 对角 \
            axc.annotate('', xy=(0.707, -0.707), xytext=(-0.707, 0.707), arrowprops=arrow_kw)  # 对角 /

            txt_kw = dict(fontsize=15, color='black', ha='center', va='center')
            axc.text(1.15, 0.0, '+Y', transform=axc.transData, **txt_kw)
            axc.text(-1.1, 0.0, '-Y', transform=axc.transData, **txt_kw)
            axc.text(0.0, 1.1, '+Z', transform=axc.transData, **txt_kw)
            axc.text(0.0, -1.1, '-Z', transform=axc.transData, **txt_kw)

            axc.plot(0, 0, marker='o', markersize=4, color='black')
        except Exception as e:
            print(f"绘制中间米字坐标轴失败: {e}")


        try:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"合并面板已保存: {save_path}")
        except Exception as e:
            print(f"保存合并面板失败: {e}")
        finally:
            plt.close(fig)

    # 生成并保存合并面板
    if len(preds) >= 8:
        combined_path = os.path.join(OUTPUT_DIR, 'combined_8panel.pdf')
        plot_8_panel(preds[:8], labels[:8], combined_path)
    else:
        print("预测结果数量不足 8 张，跳过合并绘图。")

    print("全部手动预测处理完成。图片保存在：", img_dir)

if __name__ == '__main__':
    main()
