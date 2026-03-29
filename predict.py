"""
BioBot Prediction Script
Load the trained model and run predictions on new/test data.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from data_loader import load_and_merge, build_sequences, split_data, FEATURES
from cnn_lstm_model import BioBotCNNLSTM

MODEL_PATH = 'models/biobot.keras'


def main():
    print("BioBot - Vivabilite Predictor")
    print("-" * 40)

    # Load data & rebuild test set
    print("Loading data...")
    df = load_and_merge()
    X, y, scaler = build_sequences(df)
    _, _, (X_test, y_test) = split_data(X, y)

    # Load trained model
    print(f"Loading model from {MODEL_PATH} ...")
    model = BioBotCNNLSTM(verbose=0)
    model.load(MODEL_PATH)

    # Evaluate
    results = model.evaluate(X_test, y_test)

    # Print sample predictions
    print("\nSample Predictions:")
    print(f"  {'#':<4} {'Predicted':>12} {'Actual':>12} {'Error':>10}")
    print("  " + "-" * 40)
    preds = model.predict(X_test[:10])
    for i in range(10):
        err = abs(preds[i][0] - y_test[i][0])
        print(f"  {i+1:<4} {preds[i][0]:>12.2f} {y_test[i][0]:>12.2f} {err:>10.2f}")

    print("\nFeatures used:")
    for f in FEATURES:
        print(f"  - {f}")


if __name__ == '__main__':
    main()
