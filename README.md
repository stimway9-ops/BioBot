# BioBot - CNN-LSTM Hybrid for Livable Area Prediction

![Status](https://img.shields.io/badge/Status-Production_Ready-green)
![Model](https://img.shields.io/badge/Model-CNN--LSTM-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-yellow)

## What is BioBot?

BioBot is a **deep learning system** that predicts **environmental livability scores (0-1)** from 10 biosensor readings. Built with a hybrid CNN-LSTM architecture trained on 12,915 real-world sequences.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/stimway9-ops/BioBot.git
cd BioBot
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run web interface
streamlit run app.py

# Or make predictions
python predict.py
```

## Current Model

### Architecture: CNN-LSTM Hybrid
```
Input (24 timesteps × 10 features)
  → Conv1D Block 1 (64 filters, kernel=3)
  → Conv1D Block 2 (128 filters, kernel=3) 
  → Bidirectional LSTM (64+32 units)
  → Dense Layers (64→32→1)
  → Output: Vivabilite Score (0-1)
```

### Performance
| Metric | Value |
|--------|-------|
| **MAE** | 0.0416 |
| **RMSE** | 0.1638 |
| **MSE** | 0.0268 |
| **Accuracy** | ~96% |
| **Parameters** | 163,425 |
| **Model Size** | ~3MB |

### Input Features (10 sensors)
| Feature | Unit | Range | Source |
|---------|------|-------|--------|
| Temperature | °C | 0-50 | Neusta, IoT, Meteo |
| Humidity | % | 0-100 | Neusta, IoT, Meteo |
| Humidex | °C | 0-60 | Neusta, Meteo |
| Wind | km/h | 0-50 | Meteo France |
| Soil Moisture | % | 0-100 | Aquacheck |
| CO₂ | ppm | 300-2000 | IoT Sensors |
| TVOC | ppb | 0-1000 | IoT Sensors |
| PM2.5 | μg/m³ | 0-200 | IoT Sensors |
| PM10 | μg/m³ | 0-300 | IoT Sensors |
| Sound Level | dB | 30-100 | IoT Sensors |

---

## How to Use

### 1. Web Interface (Interactive)
```bash
streamlit run app.py
```
- Adjust sliders for all 10 sensors
- Click "Predict Livability"
- View gauge chart and radar plot

### 2. Command Line
```bash
python predict.py
```

### 3. Python API
```python
from cnn_lstm_model import BioBotCNNLSTM
import numpy as np

# Load model
model = BioBotCNNLSTM(verbose=0)
model.load('models/biobot.keras')

# Create input (24 timesteps × 10 features)
# Replace with your actual sensor data
X = np.random.rand(1, 24, 10)

# Predict
score = model.predict(X)[0][0]
print(f"Livability Score: {score:.3f}")
```

---

## Project Structure

```
BioBot/
├── models/
│   ├── biobot.keras              ✅ Trained model (working)
│   └── biobot_attention.keras    ⚠️ Experimental (Lambda issues)
├── dataset/
│   ├── iot-data/                 # IoT sensor readings
│   ├── aquacheck/                # Soil moisture data
│   ├── donnees_neusta.csv        # Environmental data
│   └── data202425_meteo_france.csv # Weather data
├── app.py                        # Streamlit web UI
├── cnn_lstm_model.py             # Model architecture
├── data_loader.py                # Data pipeline
├── train.py                      # Training script
├── predict.py                    # Inference script
├── DOCUMENTATION.md              # Full technical docs
├── RESULTS.md                    # Performance results
├── TEST_INSTRUCTIONS.md          # Testing guide
└── requirements.txt              # Dependencies
```

---

## Commands

| Command | Description |
|---------|-------------|
| `streamlit run app.py` | Web interface with sliders |
| `python predict.py` | CLI predictions on test set |
| `python train.py` | Retrain model from scratch |
| `python data_loader.py` | Test data loading pipeline |

---

## Model Improvement Roadmap

### Phase 1: Architecture Enhancements (Next 2 weeks)

#### 1. **Add Attention Mechanism**
The current LSTM processes sequences uniformly. Attention lets the model focus on important time steps.

```python
# In cnn_lstm_attention_model.py (fix Lambda issues)
from tensorflow.keras.layers import Attention, GlobalAveragePooling1D

def attention_3d_block(inputs):
    # Self-attention: each time step attends to all others
    attention = Attention()([inputs, inputs])
    # Average over time steps
    return GlobalAveragePooling1D()(attention)

# Replace in build():
x = Bidirectional(LSTM(64, return_sequences=True))(x)
attention_out = attention_3d_block(x)  # Context vector
lstm_last = Lambda(lambda x: x[:, -1, :])(x)  # Last time step
x = Concatenate()([attention_out, lstm_last])  # Combine
```

#### 2. **Increase Sequence Length**
```python
# In data_loader.py
SEQ_LENGTH = 48  # 4 hours at 5-min intervals (was 24 = 2h)
# or
SEQ_LENGTH = 96  # 8 hours
```

#### 3. **Dilated Causal Convolutions**
```python
# In cnn_lstm_model.py
x = Conv1D(128, kernel_size=3, activation='relu', 
           padding='causal', dilation_rate=2)(x)
x = Conv1D(128, kernel_size=3, activation='relu', 
           padding='causal', dilation_rate=4)(x)
# Captures longer patterns without increasing parameters
```

### Phase 2: Feature Engineering (Next 3 weeks)

#### 1. **Add Derived Features**
```python
# In data_loader.py
def compute_derived_features(df):
    # Heat Index
    df['heat_index'] = (
        -8.78469475556 +
        1.61139411 * df['temperature'] +
        2.33854883889 * df['humidity'] +
        -0.14611605 * df['temperature'] * df['humidity'] +
        -0.012308094 * df['temperature']**2 +
        -0.0164248277778 * df['humidity']**2 +
        0.002211732 * df['temperature']**2 * df['humidity'] +
        0.00072546 * df['temperature'] * df['humidity']**2 +
        -0.000003582 * df['temperature']**2 * df['humidity']**2
    )
    
    # Dew Point
    df['dew_point'] = df['temperature'] - ((100 - df['humidity']) / 5)
    
    # Rolling Statistics
    df['temp_rolling_mean'] = df['temperature'].rolling(window=12).mean()
    df['humidity_rolling_std'] = df['humidity'].rolling(window=12).std()
    
    # Temperature-Humidity Interaction
    df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
    
    return df
```

#### 2. **Add External Data**
```python
# Fetch from Météo-France API
import requests

def fetch_meteo_data():
    url = "https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/infrahoraire-6m"
    headers = {"apikey": "YOUR_API_KEY"}
    params = {
        "id-station": "075001001",
        "periode": "2026-03-29"
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()
```

### Phase 3: Training Improvements (Next 4 weeks)

#### 1. **Hyperparameter Tuning with Optuna**
```python
import optuna

def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128])
    
    # Build and train model
    model = build_model(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        lstm_units=lstm_units
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        verbose=0
    )
    
    # Return validation loss
    return history.history['val_loss'][-1]

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

#### 2. **Learning Rate Schedules**
```python
# In cnn_lstm_model.py
from tensorflow.keras.optimizers.schedules import CosineDecay

lr_schedule = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    alpha=1e-6  # minimum learning rate
)

optimizer = Adam(learning_rate=lr_schedule)
```

#### 3. **Data Augmentation**
```python
# In data_loader.py
def augment_data(X, y, noise_level=0.01):
    """Add Gaussian noise to sequences"""
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return np.vstack([X, X_noisy]), np.vstack([y, y])

# In train.py
X_train, y_train = augment_data(X_train, y_train)
```

### Phase 4: Deployment (Next 6 weeks)

#### 1. **Model Quantization**
```python
import tensorflow as tf

# Convert to TF-Lite for mobile/edge devices
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Save quantized model
with open('models/biobot_quantized.tflite', 'wb') as f:
    f.write(quantized_model)
```

#### 2. **FastAPI Endpoints**
```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SensorData(BaseModel):
    temperature: float
    humidity: float
    humidex: float
    wind: float
    soil_moisture: float
    CO2: float
    TVOC: float
    PM2_5: float
    PM10: float
    sound_level: float

@app.post("/predict")
async def predict(data: SensorData):
    X = preprocess_sensor_data(data)
    prediction = model.predict(X)
    return {"score": float(prediction[0][0])}
```

#### 3. **Docker Deployment**
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## Files Status

| File | Status | Purpose |
|------|--------|---------|
| `predict.py` | ✅ Working | Make predictions |
| `train.py` | ✅ Working | Train model |
| `app.py` | ✅ Working | Streamlit web UI |
| `cnn_lstm_model.py` | ✅ Working | Model architecture |
| `data_loader.py` | ✅ Working | Data pipeline |
| `cnn_lstm_attention_model.py` | ⚠️ Experimental | Lambda serialization issues |
| `train_attention.py` | ⚠️ Experimental | Same issue |
| `DOCUMENTATION.md` | ✅ Complete | Full technical docs |
| `RESULTS.md` | ✅ Complete | Performance results |

---

## Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Full technical documentation (architecture, training, API)
- **[RESULTS.md](RESULTS.md)** - Performance metrics and sample predictions
- **[TEST_INSTRUCTIONS.md](TEST_INSTRUCTIONS.md)** - Testing guide
- **[app.py](app.py)** - Interactive web interface

---

## Contributing

1. Fork repository
2. Create feature branch
3. Implement improvements
4. Test thoroughly
5. Submit pull request

---

## License

MIT

---

**Version**: 1.1.0  
**Last Updated**: March 29, 2026  
**Status**: Production Ready ✅
