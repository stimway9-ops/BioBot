# BioBot CNN-LSTM Model Results

## Model Architecture
- CNN-LSTM hybrid with two Conv1D blocks and two Bidirectional LSTM layers
- Input: 24 timesteps × 10 features (temperature, humidity, humidex, wind, CO2, TVOC, PM2.5, PM10, sound_level, soil_moisture)
- Output: Vivabilite score (livability index, 0-1 scale)

## Training Details
- Dataset: Merged real-world biosense data from:
  - IoT sensors (temperature, humidity, TVOC, CO2, PM, sound)
  - Meteo France (wind, humidex)
  - Neusta (temperature, humidity, PM, humidex, vivabilite)
  - Aquacheck (soil moisture)
- Total samples: 12,939 sequences
- Train/Val/Test split: 9040 / 1937 / 1938
- Epochs: 100 (with early stopping)
- Batch size: 32
- Optimizer: Adam (learning rate 0.001)
- Loss: Mean Squared Error
- Metrics: Mean Absolute Error

## Performance
- Test MAE: 0.0416 vivabilite units
- Test RMSE: 0.1638 vivabilite units
- Test MSE: 0.0268

## Interpretation
- On average, predictions are within ±0.04 vivabilite points of actual values
- For a 0-1 scale, this represents good accuracy for a complex environmental prediction task
- Sample predictions show close alignment with actual values (typically 0.01 vs 0.00)

## Model File
- Saved model: `models/biobot.keras`

## Usage
To make predictions:
```bash
cd C:\Users\stimw\Documents\BioBot
venv\Scripts\python.exe predict.py
```

## Potential Improvements
1. **Increase sequence length** (e.g., 48-96 timesteps) to capture longer temporal patterns
2. **Add more meteorological features** from public APIs (pressure, precipitation, UV index)
3. **Architecture enhancements**:
   - Add attention mechanism
   - Try temporal convolutional networks (TCN)
   - Experiment with Transformer-based models
4. **Feature engineering**:
   - Compute derived variables (dew point, heat index)
   - Add rolling statistics (mean, std over windows)
5. **Hyperparameter tuning**:
   - Learning rate schedules
   - Different filter sizes and LSTM unit counts
   - Batch size optimization

## Data Sources Used
- IoT sensor data: `dataset/iot-data/` (JSON files)
- Neusta data: `dataset/donnees_neusta.csv`
- Meteo France data: `dataset/data202425_meteo_france.csv`
- Aquacheck data: `dataset/aquacheck/` (JSON files)

## Next Steps
To integrate real-time data from Météo-France APIs:
1. Use the public data from https://donneespubliques.meteofrance.fr/
2. Fetch SYNOP or other observation datasets
3. Extract temperature, humidity, wind, pressure, etc.
4. Compute humidex if not provided
5. Merge with existing IoT/soil moisture pipelines
6. Retrain model with enriched feature set
