import os
import time
import datetime
import shutil
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from geomag import geomag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.io import savemat

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =================== 基础目录 ===================
BASE_DIR     = '/root/autodl-tmp/my_project'
DATA_DIR     = os.path.join(BASE_DIR, '24')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'Res')
IMG_SAVE_DIR = os.path.join(OUTPUT_DIR, 'pngs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.keras')
SCALER_PATH     = os.path.join(OUTPUT_DIR, 'scaler.joblib')
GLOBAL_RANGE_NPY= os.path.join(OUTPUT_DIR, 'global_pot_range.npy')

# =================== 数据加载 ===================
all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_Match_Omni_SD_Pot.mat')])
mat_files = all_files[:34]

# 临时列表
timestamps = []
feat_list  = []
pos_list   = []
pot_list   = []
frame_idx_list = []
ut_list = []
frame_ut_hours = []

for fname in mat_files:
    date_str = fname.split('_')[0]
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, fname), verify_compressed_data_integrity=False)
    pot_all = mat.get('Match_Omni_SD_Pot')
    if pot_all is None:
        continue

    frames = pot_all.shape[0] // 5400
    for fi in range(frames):
        blk = pot_all[fi*5400:(fi+1)*5400, :]
        hour, minute = int(blk[0,0]), int(blk[0,1])
        full_t = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(hours=hour, minutes=minute)
        ts = full_t.strftime("%Y%m%d.%H%M")

        global_fi = len(timestamps)
        timestamps.append(ts)
        frame_ut_hours.append(float(hour) + float(minute)/60.0)

        feat0 = blk[0, 2:8]  # Bx,By,Bz,Vx,Pd,AE
        ut_hour_frame = float(hour) + float(minute)/60.0

        for p in range(5400):
            feat_list.append(feat0)
            pos_list.append(blk[p, 10:12])
            pot_list.append(blk[p, 12])
            frame_idx_list.append(global_fi)
            ut_list.append(ut_hour_frame)

# 转为 numpy
features      = np.vstack(feat_list).astype(np.float32)
pos_arr       = np.vstack(pos_list).astype(np.float32)
ut_arr        = np.array(ut_list, dtype=np.float32)

# =================== 编码 ===================
K_geo = 4
K_mlt = 2
K_lat = 2

lons_deg_all = pos_arr[:, 1]
lons_rad = np.deg2rad(lons_deg_all)
geo_fourier = np.column_stack(
    [np.sin(k * lons_rad) for k in range(1, K_geo + 1)] +
    [np.cos(k * lons_rad) for k in range(1, K_geo + 1)]
)

mlt_deg = ((lons_deg_all / 15.0) + ut_arr) % 24 * 15.0
mlt_rad = np.deg2rad(mlt_deg)
mlt_fourier = np.column_stack(
    [np.sin(k * mlt_rad) for k in range(1, K_mlt + 1)] +
    [np.cos(k * mlt_rad) for k in range(1, K_mlt + 1)]
)

lats_all = pos_arr[:, 0]
lats_rad = np.deg2rad(lats_all)
lat_fourier = np.column_stack(
    [np.sin(k * lats_rad) for k in range(1, K_lat + 1)] +
    [np.cos(k * lats_rad) for k in range(1, K_lat + 1)]
)

positions = np.column_stack((lats_all, geo_fourier, mlt_fourier, lat_fourier)).astype(np.float32)
X_all = np.hstack((features, positions))

pot_values    = np.array(pot_list, dtype=np.float32)
frame_indices = np.array(frame_idx_list, dtype=np.int32)
total_frames  = len(timestamps)

train_end = int(0.80 * total_frames)
train_mask = frame_indices < train_end
val_mask   = (frame_indices >= train_end) 

X_train, y_train = X_all[train_mask], pot_values[train_mask]
X_val,   y_val   = X_all[val_mask],   pot_values[val_mask]

scaler = MinMaxScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm   = scaler.transform(X_val)
dump(scaler, SCALER_PATH)

global_min, global_max = y_train.min(), y_train.max()
np.save(GLOBAL_RANGE_NPY, np.array([global_min, global_max], dtype=np.float32))

y_train_norm = (y_train - global_min) / (global_max - global_min)
y_val_norm   = (y_val   - global_min) / (global_max - global_min)

np.save(os.path.join(OUTPUT_DIR, 'X_train_norm.npy'), X_train_norm)
np.save(os.path.join(OUTPUT_DIR, 'y_train_norm.npy'), y_train_norm)
np.save(os.path.join(OUTPUT_DIR, 'X_val_norm.npy'), X_val_norm)
np.save(os.path.join(OUTPUT_DIR, 'y_val_norm.npy'), y_val_norm)

def safe_load_memmap(path):
    try:
        mm = np.load(path, mmap_mode='r')
        print(f"Loaded memmap: {path} -> shape={mm.shape}, dtype={mm.dtype}")
        return mm
    except Exception as e:
        arr = np.load(path)
        print(f"Loaded full array: {path} -> shape={arr.shape}, dtype={arr.dtype}")
        return arr

X_train_mm = safe_load_memmap(os.path.join(OUTPUT_DIR, 'X_train_norm.npy'))
y_train_mm = safe_load_memmap(os.path.join(OUTPUT_DIR, 'y_train_norm.npy'))
X_val_mm   = safe_load_memmap(os.path.join(OUTPUT_DIR, 'X_val_norm.npy'))
y_val_mm   = safe_load_memmap(os.path.join(OUTPUT_DIR, 'y_val_norm.npy'))

USE_SUBSAMPLED_FILE = True          
SUBSAMPLED_TRAIN_SAMPLES = 75_000_000 

BATCH_SIZE = 2048  
SHUFFLE_BUFFER = 200000  
AUTOTUNE = tf.data.AUTOTUNE

from numpy.lib.format import open_memmap

n_train = int(X_train_mm.shape[0])

out_X_path = os.path.join(OUTPUT_DIR, 'X_train_sub.npy')
out_y_path = os.path.join(OUTPUT_DIR, 'y_train_sub.npy')

if USE_SUBSAMPLED_FILE:
    target = int(min(SUBSAMPLED_TRAIN_SAMPLES, n_train))
    def is_valid_npy(path):
        try:
            arr = np.load(path, mmap_mode='r')
            return True
        except Exception:
            return False

    if os.path.exists(out_X_path) and os.path.exists(out_y_path) and is_valid_npy(out_X_path) and is_valid_npy(out_y_path):
        X_train_sub_mm = np.load(out_X_path, mmap_mode='r')
        y_train_sub_mm = np.load(out_y_path, mmap_mode='r')
    else:
        for p in (out_X_path, out_y_path):
            if os.path.exists(p):
                bak = p + '.bak.' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                shutil.move(p, bak)

        rng = np.random.default_rng(seed=42)
        selected_idx = rng.choice(n_train, size=target, replace=False)

        try:
            out_dim = int(X_train_mm.shape[1])
            X_out_mm = open_memmap(out_X_path, mode='w+', dtype='float32', shape=(target, out_dim))
            y_out_mm = open_memmap(out_y_path, mode='w+', dtype='float32', shape=(target,))

            chunk = 10000
            write_ptr = 0
            t0 = time.time()
            for start in range(0, target, chunk):
                end = min(target, start + chunk)
                idx_chunk = selected_idx[start:end]
                X_out_mm[write_ptr:write_ptr + (end - start), :] = X_train_mm[idx_chunk, :]
                y_out_mm[write_ptr:write_ptr + (end - start)] = y_train_mm[idx_chunk]
                write_ptr += (end - start)
                if write_ptr % (10*chunk) == 0 or write_ptr == target:
                    elapsed = time.time() - t0
                  
            del X_out_mm, y_out_mm
            elapsed_total = time.time() - t0
          
     
            X_train_sub_mm = np.load(out_X_path, mmap_mode='r')
            y_train_sub_mm = np.load(out_y_path, mmap_mode='r')
            print(f"Loaded subsampled memmap shapes: X={X_train_sub_mm.shape}, y={y_train_sub_mm.shape}")

        except Exception as e:
            for p in (out_X_path, out_y_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as e2:
            for p in (out_X_path, out_y_path):
                bak = p + '.bak.' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            raise
else:
    X_train_sub_mm = X_train_mm
    y_train_sub_mm = y_train_mm

import math

def make_dataset_from_memmap(X_mm, y_mm, batch_size=BATCH_SIZE, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X_mm, y_mm))
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

val_ds = make_dataset_from_memmap(X_val_mm, y_val_mm, batch_size=BATCH_SIZE, shuffle=False)

train_ds = make_dataset_from_memmap(X_train_sub_mm, y_train_sub_mm, batch_size=BATCH_SIZE, shuffle=True)
steps_per_epoch = math.ceil(X_train_sub_mm.shape[0] / BATCH_SIZE)

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
    epochs=20,
    callbacks=[es, cp],
    verbose=2,
    steps_per_epoch=steps_per_epoch
)

hist_df = pd.DataFrame(history.history)
history_csv_path = os.path.join(OUTPUT_DIR, 'training_history.csv')
hist_npy_path = os.path.join(OUTPUT_DIR, 'training_history.npy')
hist_df.to_csv(history_csv_path, index=False)
np.save(hist_npy_path, history.history)

train_mse = history.history.get('mse', history.history.get('loss'))
val_mse   = history.history.get('val_mse', history.history.get('val_loss'))
train_mae = history.history.get('mae')
val_mae   = history.history.get('val_mae')

epochs = range(1, len(history.history[list(history.history.keys())[0]]) + 1)

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
# MSE / loss
axes[0].plot(epochs, train_mse, label='train MSE')
if val_mse is not None:
    axes[0].plot(epochs, val_mse, label='val MSE')
axes[0].set_ylabel('MSE')
axes[0].set_title('Training MSE (loss)')
axes[0].legend()
axes[0].grid(True)

# MAE
if train_mae is not None:
    axes[1].plot(epochs, train_mae, label='train MAE')
if val_mae is not None:
    axes[1].plot(epochs, val_mae, label='val MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('Training MAE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
plot_path = os.path.join(IMG_SAVE_DIR, 'mse_mae_training_curve.png')
fig.savefig(plot_path, dpi=150)
plt.close(fig)
