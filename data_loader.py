"""
BioBot Data Loader
Loads and merges real biosense datasets:
  - dataset/iot-data/        : IoT sensors (temp, humidity, TVOC, CO2, PM, sound)
  - dataset/donnees_neusta.csv: Neusta (temp, humidity, PM, humidex, Vivabilite)
  - dataset/data202425_meteo_france.csv: Meteo France (wind, humidex, Vivabilite)
  - dataset/aquacheck/       : Soil moisture
Target: Vivabilite (livability score)
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Feature columns used for the model ──────────────────────────────────────
FEATURES = [
    'temperature', 'humidity', 'humidex',
    'wind', 'CO2', 'TVOC',
    'PM2_5', 'PM10', 'sound_level', 'soil_moisture'
]
TARGET = 'vivabilite'
SEQ_LENGTH = 24   # timesteps per sample


# ── 1. Load IoT sensor data ──────────────────────────────────────────────────
def load_iot_data(folder='dataset/iot-data'):
    records = []
    for path in sorted(glob.glob(os.path.join(folder, '*.json'))):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(records)
    df.rename(columns={
        'temperature': 'temperature',
        'humidity': 'humidity',
        'TVOC': 'TVOC',
        'CO2': 'CO2',
        'PM2.5': 'PM2_5',
        'PM10': 'PM10',
        'sound_level': 'sound_level'
    }, inplace=True)

    # parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # numeric conversion
    for col in ['temperature', 'humidity', 'TVOC', 'CO2', 'PM2_5', 'PM10', 'sound_level']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # group by minute, average across sensor IDs
    df = df.set_index('timestamp')
    df = df[['temperature', 'humidity', 'TVOC', 'CO2', 'PM2_5', 'PM10', 'sound_level']]
    df = df.resample('5min').mean()
    return df


# ── 2. Load Aquacheck soil moisture ─────────────────────────────────────────
def load_aquacheck(folder='dataset/aquacheck'):
    records = []
    for path in sorted(glob.glob(os.path.join(folder, '*.json'))):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except json.JSONDecodeError:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['soil_moisture'] = pd.to_numeric(df.get('soilMoisture (%)', np.nan), errors='coerce')
    df = df.set_index('timestamp')[['soil_moisture']]
    df = df.resample('5min').mean()
    return df


# ── 3. Load Neusta CSV ───────────────────────────────────────────────────────
def load_neusta(path='dataset/donnees_neusta.csv'):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # The CSV has columns: timestamp, Temperature, Humidity, <sensor cols>, temperature, humidity, PM1, PM2.5, PM10, Humidex, Vivabilite
    # Keep only the columns we need by position or exact name match
    keep = {}
    for col in df.columns:
        low = col.strip()
        if low == 'Temperature':
            keep['temperature'] = df[col]
        elif low == 'Humidity':
            keep['humidity'] = df[col]
        elif low == 'PM2.5':
            keep['PM2_5'] = df[col]
        elif low == 'PM10':
            keep['PM10'] = df[col]
        elif low == 'Humidex':
            keep['humidex'] = df[col]
        elif low == 'Vivabilite':
            keep['vivabilite'] = df[col]

    result = pd.DataFrame(keep, index=df.index)
    result.index = df['timestamp'].values

    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors='coerce')

    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    result = result.resample('5min').mean()
    return result


# ── 4. Load Meteo France CSV ─────────────────────────────────────────────────
def load_meteo_france(path='dataset/data202425_meteo_france.csv'):
    df = pd.read_csv(path, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['validity_time'], errors='coerce', utc=False)
    df = df.dropna(subset=['timestamp'])

    # strip timezone if present
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    df.rename(columns={
        'ff': 'wind',
        'Temp_C': 'temperature',
        'humidex': 'humidex',
        'Vivabilite': 'vivabilite',
        'u': 'humidity',
    }, inplace=True)

    for col in ['temperature', 'humidity', 'wind', 'humidex', 'vivabilite']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.set_index('timestamp').sort_index()
    cols = [c for c in ['temperature', 'humidity', 'wind', 'humidex', 'vivabilite'] if c in df.columns]
    df = df[cols].resample('5min').mean().interpolate('time')
    return df


# ── 5. Merge all sources ─────────────────────────────────────────────────────
def load_and_merge():
    print("Loading IoT data...")
    iot = load_iot_data()

    print("Loading Aquacheck soil moisture...")
    aqua = load_aquacheck()

    print("Loading Neusta data...")
    neusta = load_neusta()

    print("Loading Meteo France data...")
    meteo = load_meteo_france()

    print("Merging datasets...")

    # ── Primary source: Neusta (has Vivabilite + core bio fields) ────────────
    merged = neusta.copy()

    # ── Add wind from meteo (resample to 5min first, then join) ──────────────
    if 'wind' in meteo.columns:
        # meteo is hourly; reindex to neusta's 5min grid
        merged['wind'] = meteo['wind'].reindex(merged.index, method='nearest', tolerance='1h')

    # ── Fill humidex from meteo if missing in neusta ──────────────────────────
    if 'humidex' not in merged.columns or merged['humidex'].isna().all():
        if 'humidex' in meteo.columns:
            merged['humidex'] = meteo['humidex'].reindex(merged.index, method='nearest', tolerance='1h')

    # ── Add IoT columns (TVOC, CO2, sound) ───────────────────────────────────
    for col in ['TVOC', 'CO2', 'PM2_5', 'PM10', 'sound_level']:
        if col in iot.columns:
            merged[col] = iot[col].reindex(merged.index, method='nearest', tolerance='1h')

    # ── Soil moisture ─────────────────────────────────────────────────────────
    if not aqua.empty:
        merged['soil_moisture'] = aqua['soil_moisture'].reindex(merged.index, method='nearest', tolerance='1h')
    else:
        merged['soil_moisture'] = np.nan

    # ── Ensure all feature columns exist ─────────────────────────────────────
    for col in FEATURES:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[FEATURES + [TARGET]].copy()

    # Drop rows where the target is missing
    merged = merged.dropna(subset=[TARGET])

    # Forward/backward fill gaps up to 12 steps (1 hour at 5min resolution)
    merged[FEATURES] = merged[FEATURES].ffill(limit=12).bfill(limit=12)

    # Fill any still-missing feature columns with column median
    for col in FEATURES:
        if merged[col].isna().any():
            median_val = merged[col].median()
            merged[col] = merged[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    print(f"Merged dataset: {len(merged):,} rows, features: {merged.columns.tolist()}")
    return merged


# ── 6. Build sequences for CNN-LSTM ─────────────────────────────────────────
def build_sequences(df, seq_length=SEQ_LENGTH):
    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(df[FEATURES].values)
    target_data = df[TARGET].values

    X, y = [], []
    for i in range(len(feature_data) - seq_length):
        X.append(feature_data[i:i + seq_length])
        y.append(target_data[i + seq_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y, scaler


# ── 7. Train/val/test split ──────────────────────────────────────────────────
def split_data(X, y, train=0.70, val=0.15):
    n = len(X)
    t = int(n * train)
    v = int(n * (train + val))
    return (X[:t], y[:t]), (X[t:v], y[t:v]), (X[v:], y[v:])


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load_and_merge()
    print(df.describe())
    X, y, scaler = build_sequences(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    (Xtr, ytr), (Xv, yv), (Xte, yte) = split_data(X, y)
    print(f"Train: {Xtr.shape}, Val: {Xv.shape}, Test: {Xte.shape}")
