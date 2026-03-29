"""
BioBot Training Script with Attention Model
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_loader import load_and_merge, build_sequences, split_data, FEATURES
from cnn_lstm_attention_model import BioBotCNNLSTMAtt

EPOCHS     = 150
BATCH_SIZE = 32
MODEL_PATH = 'models/biobot_attention.keras'

def main():
    print("=" * 55)
    print("  BioBot CNN-LSTM-Attention  |  Vivabilite Prediction")
    print("=" * 55)

    # ── 1. Load & merge real data ──────────────────────────────
    print("\n[1/4] Loading and merging datasets...")
    df = load_and_merge()
    print(f"      Total rows after merge: {len(df):,}")
    print(f"      Features: {FEATURES}")

    # ── 2. Build sequences ────────────────────────────────────
    print("\n[2/4] Building sequences...")
    X, y, scaler = build_sequences(df)
    print(f"      X: {X.shape}  y: {y.shape}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    print(f"      Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── 3. Build & train model ────────────────────────────────
    print("\n[3/4] Building CNN-LSTM-Attention model...")
    model = BioBotCNNLSTMAtt(n_features=X_train.shape[2])
    model.build()

    print("\n      Training...")
    model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH
    )

    # ── 4. Evaluate ───────────────────────────────────────────
    print("\n[4/4] Evaluating on test set...")
    results = model.evaluate(X_test, y_test)

    # ── Save plot ─────────────────────────────────────────────
    model.plot_training()

    # ── Sample predictions ────────────────────────────────────
    print("\nSample predictions (first 5 test samples):")
    preds = model.predict(X_test[:5])
    for i in range(5):
        print(f"  [{i+1}] Predicted: {preds[i][0]:.2f}  |  Actual: {y_test[i][0]:.2f}")

    print("\n" + "=" * 55)
    print(f"  Done!  RMSE: {results['rmse']:.4f}  MAE: {results['mae']:.4f}")
    print("=" * 55)


if __name__ == '__main__':
    main()
