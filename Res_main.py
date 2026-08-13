#  ---------- 环境  ----------
import os
import numpy as np
import scipy.io
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from skimage.metrics import structural_similarity as ssim
from geomag import geomag
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random

# 地磁模块
gm = geomag.GeoMag()

import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from scipy.io import savemat
import matplotlib.dates as mdates

# 可复现性
np.random.seed(42)
tf.random.set_seed(42)

# ==================== 第一步：数据加载与预处理 ====================
DATA_DIR = '24'
OUTPUT_DIR = 'Res_24'
IMG_SAVE_DIR = 'pngs'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.keras')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.joblib')
GLOBAL_RANGE_NPY = os.path.join(OUTPUT_DIR, 'global_pot_range.npy')

print("🔍 开始加载数据...")

all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_Match_Omni_SD_Pot.mat')])
mat_files = all_files

# 临时列表
timestamps = []
feat_list = []
pos_list = []
pot_list = []
frame_idx_list = []

ut_list = []  # 每个点对应的 UT hour (小时小数)
frame_ut_hours = []  # 每帧（frame index）对应的 UT hour（长度 = total_frames）
for fname in mat_files:
    date_str = fname.split('_')[0]
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, fname), verify_compressed_data_integrity=False)
    pot_all = mat.get('Match_Omni_SD_Pot')
    if pot_all is None:
        continue

    frames = pot_all.shape[0] // 5400
    for fi in range(frames):
        blk = pot_all[fi * 5400:(fi + 1) * 5400, :]
        hour, minute = int(blk[0, 0]), int(blk[0, 1])
        full_t = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(hours=hour, minutes=minute)
        ts = full_t.strftime("%Y%m%d.%H%M")

        global_fi = len(timestamps)
        timestamps.append(ts)
        frame_ut_hours.append(float(hour) + float(minute) / 60.0)  # 每帧保存 UT hour

        feat0 = blk[0, 2:8]  # Bx,By,Bz,Vx,Pd,AE
        ut_hour_frame = float(hour) + float(minute) / 60.0

        for p in range(5400):
            feat_list.append(feat0)
            pos_list.append(blk[p, 10:12])
            pot_list.append(blk[p, 12])
            frame_idx_list.append(global_fi)
            ut_list.append(ut_hour_frame)

# 转成 numpy
features = np.vstack(feat_list).astype(np.float32)
pos_arr = np.vstack(pos_list).astype(np.float32)
ut_arr = np.array(ut_list, dtype=np.float32)  # 每个样本对应的 UT

#  ---------- 傅里叶编码 ----------
K_geo = 4
K_mlt = 2
K_lat = 2

# geo 傅里叶
lons_deg_all = pos_arr[:, 1]
lons_rad = np.deg2rad(lons_deg_all)
geo_fourier = np.column_stack(
    [np.sin(k * lons_rad) for k in range(1, K_geo + 1)] +
    [np.cos(k * lons_rad) for k in range(1, K_geo + 1)]
)

# MLT — per-sample：使用 ut_arr
mlt_deg = ((lons_deg_all / 15.0) + ut_arr) % 24 * 15.0
mlt_rad = np.deg2rad(mlt_deg)
mlt_fourier = np.column_stack(
    [np.sin(k * mlt_rad) for k in range(1, K_mlt + 1)] +
    [np.cos(k * mlt_rad) for k in range(1, K_mlt + 1)]
)

# lat 傅里叶
lats_all = pos_arr[:, 0]
lats_rad = np.deg2rad(lats_all)
lat_fourier = np.column_stack(
    [np.sin(k * lats_rad) for k in range(1, K_lat + 1)] +
    [np.cos(k * lats_rad) for k in range(1, K_lat + 1)]
)

# 合并位置特征
positions = np.column_stack((lats_all, geo_fourier, mlt_fourier, lat_fourier)).astype(np.float32)
X_all = np.hstack((features, positions))

pot_values = np.array(pot_list, dtype=np.float32)
frame_indices = np.array(frame_idx_list, dtype=np.int32)
total_frames = len(timestamps)

print(f"📂 加载完毕，总帧数：{total_frames}，总点数：{features.shape[0]}")

#  ---------- 按帧顺序划分训练/验证/测试集----------
train_end = int(0.8 * total_frames)
train_mask = frame_indices < train_end
val_mask = frame_indices >= train_end

X_train, y_train = X_all[train_mask], pot_values[train_mask]
X_val, y_val = X_all[val_mask], pot_values[val_mask]

print(f"🔢 划分样本（按帧）：train_frames={train_end}, val_frames={total_frames - train_end}")

# ----------------- 打印训练/验证集时间区间 -----------------
def dataset_time_range(mask, frame_indices, timestamps):
    """返回 (start_frame, end_frame, start_dt, end_dt, n_frames, n_samples) 或 None（若样本为空）"""
    if not np.any(mask):
        return None
    frames = np.unique(frame_indices[mask])
    start_frame = int(frames.min())
    end_frame = int(frames.max())
    # timestamps 列表按全局 frame 顺序构建（len == total_frames），格式 "YYYYmmdd.HHMM"
    start_dt = datetime.datetime.strptime(timestamps[start_frame], "%Y%m%d.%H%M")
    end_dt = datetime.datetime.strptime(timestamps[end_frame], "%Y%m%d.%H%M")
    n_frames = len(frames)
    n_samples = int(mask.sum())
    return start_frame, end_frame, start_dt, end_dt, n_frames, n_samples


def pretty_print_range(name, rng):
    if rng is None:
        print(f"⚠️ {name} 集合为空（没有样本）")
        return
    start_frame, end_frame, start_dt, end_dt, n_frames, n_samples = rng
    print(f"=== {name} 集合 ===")
    print(f" frame 索引范围: {start_frame} — {end_frame} (共 {n_frames} frames)")
    print(f" 时间范围: {start_dt.strftime('%Y-%m-%d %H:%M')}  —  {end_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f" 样本数 (points): {n_samples}")
    print("")


# 计算并打印
train_range = dataset_time_range(train_mask, frame_indices, timestamps)
val_range = dataset_time_range(val_mask, frame_indices, timestamps)

print("\n📅 数据集时间区间（按 frame）:")
pretty_print_range("Train", train_range)
pretty_print_range("Val  ", val_range)

# ==================== 第二步：归一化 X & 保存全局电位范围 ====================
scaler = MinMaxScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
dump(scaler, SCALER_PATH)
print(f"💾 已保存归一化器")

global_min, global_max = y_train.min(), y_train.max()
print(f"[在线统计] 训练集全局电位范围：min={global_min:.3f}, max={global_max:.3f}")
np.save(GLOBAL_RANGE_NPY, np.array([global_min, global_max], dtype=np.float32))

y_train_norm = (y_train - global_min) / (global_max - global_min)
y_val_norm = (y_val - global_min) / (global_max - global_min)

train_ds = tf.data.Dataset.from_tensor_slices((X_train_norm, y_train_norm)).shuffle(10000).batch(1024).prefetch(2)
val_ds = tf.data.Dataset.from_tensor_slices((X_val_norm, y_val_norm)).batch(1024).prefetch(1)
print(f"训练/验证 样本数（点数）：{len(X_train)}/{len(X_val)}")

# ==================== 第三步：模型定义与训练 ====================
def build_resmlp(input_dim, hidden_dim=64, depth=3, dropout=0.2):
    """
    Residual MLP with LayerNorm + GELU.
    hidden_dim: 隐藏层大小（128）
    depth: 残差块数量（可调）
    dropout: dropout 比例
    """
    x_in = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('gelu')(x)

    for _ in range(depth):
        shortcut = x
        x = layers.Dense(hidden_dim)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('gelu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Add()([x, shortcut])

    x = layers.Dense(hidden_dim // 2, activation='gelu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)

    return models.Model(inputs=x_in, outputs=out)

input_dim = 6 + 1 + 2*K_geo + 2*K_mlt + 2*K_lat  
model = build_resmlp(input_dim=input_dim, hidden_dim=128, depth=3, dropout=0.2)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', tf.keras.metrics.MeanSquaredError(name='mse')]
)
model.summary()

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
cp = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[es, cp],
    verbose=2,
)

print("✅ Res-MLP模型训练完成")