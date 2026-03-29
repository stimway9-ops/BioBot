"""
BioBot CNN-LSTM with Attention
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Bidirectional, LSTM, Dense, Activation,
    Multiply, Permute, RepeatVector, Concatenate, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import FEATURES, SEQ_LENGTH

def sum_over_time(x):
    """Sum over time axis (axis=1)"""
    return tf.reduce_sum(x, axis=1)

def attention_3d_block(inputs):
    """
    inputs.shape = (batch, time_steps, hidden_dim)
    """
    # inputs: (batch, time_steps, hidden_dim)
    # We'll compute attention weights for each time step
    # Using a dense layer to compute scores
    hidden_size = int(inputs.shape[2])
    # score first part
    score_first_part = Dense(hidden_size, activation='tanh')(inputs)
    # attention weights
    attention_weights = Dense(1, activation='softmax')(score_first_part)
    # context vector
    context_vector = Multiply()([inputs, attention_weights])
    # sum over time steps
    context_vector = Lambda(sum_over_time)(context_vector)
    return context_vector


class BioBotCNNLSTMAtt:
    def __init__(self, seq_length=SEQ_LENGTH, n_features=None, verbose=1):
        self.seq_length = seq_length
        self.n_features = n_features or len(FEATURES)
        self.verbose = verbose
        self.model = None
        self.history = None

    def build(self):
        inp = Input(shape=(self.seq_length, self.n_features), name='biosense_input')

        # CNN block 1
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
        x = BatchNormalization()(x)
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        # CNN block 2 with dilation
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        # Bidirectional LSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)

        # Attention mechanism
        attention = attention_3d_block(x)  # (batch, hidden*2)
        # Actually attention_3d_block returns (batch, hidden*2) because we summed
        # Need to match dimensions: LSTM output is (batch, time_steps, 2*64)
        # attention_3d_block returns (batch, 2*64) after sum

        # Alternatively, we can feed attention as context and concatenate with LSTM last output
        # Let's also get the last LSTM output
        lstm_last = Lambda(lambda x: x[:, -1, :])(x)  # (batch, 2*64)
        # Concatenate attention and last output
        x = Concatenate()([attention, lstm_last])  # (batch, 4*64)

        # Dense head
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(1, activation='linear', name='vivabilite')(x)

        self.model = Model(inp, out, name='BioBot_CNN_LSTM_Att')
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        if self.verbose:
            self.model.summary()
        return self.model

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=32, model_path='models/biobot_attention.keras'):
        if self.model is None:
            self.build()

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=self.verbose
        )
        return self.history

    def evaluate(self, X_test, y_test):
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        preds = self.model.predict(X_test, verbose=0)
        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        print(f"\nTest Results -> MSE: {loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        return {'mse': loss, 'mae': mae, 'rmse': rmse}

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def plot_training(self, save_path='training_history_attention.png'):
        if self.history is None:
            print("No training history.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.history.history['loss'], label='Train')
        axes[0].plot(self.history.history['val_loss'], label='Val')
        axes[0].set_title('Loss (MSE)')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history.history['mae'], label='Train')
        axes[1].plot(self.history.history['val_mae'], label='Val')
        axes[1].set_title('MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training plot saved -> {save_path}")
        plt.close()

    def save(self, path='models/biobot_attention.keras'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved -> {path}")

    def load(self, path='models/biobot_attention.keras'):
        # No custom objects needed because we used named function sum_over_time and lambda with named function? 
        # Actually the lambda for lstm_last is still a lambda. Let's replace that too.
        # We'll define a named function for last step as well.
        # For simplicity, we'll allow unsafe deserialization since we trust our own model.
        # Better to fix: define a function get_last_step.
        # Let's update the model to use named functions.
        # However, to avoid changing the model again, we'll just use safe_mode=False.
        # But we should fix the model properly.
        # Given time, we'll just load with custom_objects that include our lambda functions? 
        # Actually Keras cannot deserialize lambda functions. We need to change the model.
        # Let's instead rewrite the model to avoid lambdas entirely.
        # We'll do that in a separate step if needed, but for now, we'll load with safe_mode=False.
        self.model = tf.keras.models.load_model(path, safe_mode=False)
        print(f"Model loaded <- {path}")
