"""
BioBot Prediction Script
Use this to make predictions with trained model
"""

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_generator import BioSenseDataGenerator

def predict_livable_area(model, sample_data):
    """Predict livable area score from biosense data"""
    prediction = model.predict(sample_data, verbose=0)
    return prediction[0][0]

def main():
    from cnn_lstm_model import BioBotCNN_LSTM
    
    print("Loading model...")
    model = BioBotCNN_LSTM(verbose=0)
    model.load_model('models/biobot_cnn_lstm.h5')
    
    print("\nGenerating sample predictions...")
    generator = BioSenseDataGenerator(n_samples=10, seq_length=24)
    X_test, y_test = generator.generate_dataset()
    
    print("\nPredictions:")
    print("-" * 40)
    for i in range(5):
        pred = model.predict(X_test[i:i+1])
        print(f"Sample {i+1}: Predicted={pred[0][0]:.1f}, Actual={y_test[i][0]:.1f}")
    
    print("\n" + "-" * 40)
    print("Feature Interpretation:")
    print("  Temperature: 15-40°C")
    print("  Humidity: 20-95%")
    print("  Humidex: Feels-like temp")
    print("  Wind: 0-30 km/h")
    print("  Solar Radiation: 0-1000 W/m²")
    print("  Activity: 1=Low, 2=Medium, 3=High")
    print("  Clothing: 0.1-1.0 (light to heavy)")

if __name__ == "__main__":
    main()
