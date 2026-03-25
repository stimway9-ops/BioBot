# BioBot - CNN-LSTM Hybrid for Livable Area Prediction

AI model predicting livable area based on biosense data (temperature, humidity, wind) and human factors (clothing, activity).

## Features
- **CNN layers**: Extract spatial patterns from biosense sequences
- **Bidirectional LSTM**: Capture temporal dependencies
- **Human factors**: Activity level, clothing insulation

## Input Features
| Feature | Range | Description |
|---------|-------|-------------|
| Temperature | 15-40°C | Air temperature |
| Humidity | 20-95% | Water vapor in air |
| Humidex | Computed | Feels-like temperature |
| Wind | 0-30 km/h | Cooling effect |
| Solar Radiation | 0-1000 W/m² | Sun exposure |
| Activity | 1-3 | Physical activity level |
| Clothing | 0.1-1.0 | Clothing insulation |

## Usage
```bash
pip install -r requirements.txt
python train.py
```

## Output
Livable area score (0-100)
