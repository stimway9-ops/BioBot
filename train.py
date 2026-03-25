"""
BioBot Main Training Script
"""

from data_generator import BioSenseDataGenerator
from cnn_lstm_model import BioBotCNN_LSTM
import numpy as np
import os

def main():
    print("=" * 50)
    print("BioBot CNN-LSTM Hybrid - Livable Area Prediction")
    print("=" * 50)
    
    DATA_DIR = 'data'
    N_SAMPLES = 10000
    SEQ_LENGTH = 24
    EPOCHS = 100
    BATCH_SIZE = 32
    
    print("\n[1/4] Generating dataset...")
    generator = BioSenseDataGenerator(n_samples=N_SAMPLES, seq_length=SEQ_LENGTH)
    X, y = generator.generate_dataset()
    generator.save_dataset(X, y, DATA_DIR)
    
    print("\n[2/4] Loading data...")
    X_train = np.load(f'{DATA_DIR}/X_train.npy')
    y_train = np.load(f'{DATA_DIR}/y_train.npy')
    X_test = np.load(f'{DATA_DIR}/X_test.npy')
    y_test = np.load(f'{DATA_DIR}/y_test.npy')
    
    split = int(len(X_train) * 0.85)
    X_val = X_train[split:]
    y_val = y_train[split:]
    X_train = X_train[:split]
    y_train = y_train[:split]
    
    print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    print("\n[3/4] Building and training CNN-LSTM model...")
    model = BioBotCNN_LSTM(seq_length=SEQ_LENGTH, n_features=7)
    model.build_model()
    model.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("\n[4/4] Evaluating model...")
    results = model.evaluate(X_test, y_test)
    
    print("\nSaving model...")
    model.save_model('models/biobot_cnn_lstm.h5')
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Final Test RMSE: {results['rmse']:.2f}")
    print("=" * 50)
    
    model.plot_training()
    
    print("\nSample Predictions:")
    predictions = model.predict(X_test[:5])
    for i in range(5):
        print(f"  Sample {i+1}: Predicted={predictions[i][0]:.1f}, Actual={y_test[i][0]:.1f}")

if __name__ == "__main__":
    main()
