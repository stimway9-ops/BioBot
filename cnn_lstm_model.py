"""
BioBot CNN-LSTM Hybrid Model
Combines CNN for spatial feature extraction with LSTM for temporal patterns
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, 
    Dropout, BatchNormalization, Bidirectional,
    Concatenate, GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class BioBotCNN_LSTM:
    def __init__(self, seq_length=24, n_features=7, verbose=1):
        self.seq_length = seq_length
        self.n_features = n_features
        self.verbose = verbose
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN-LSTM hybrid architecture"""
        inputs = Input(shape=(self.seq_length, self.n_features))
        
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(32, activation='relu')(x)
        
        outputs = Dense(1, activation='linear', name='livable_score')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        if self.verbose:
            self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        predictions = self.model.predict(X_test, verbose=0)
        
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        
        print(f"Test Loss (MSE): {results[0]:.4f}")
        print(f"Test MAE: {results[1]:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        return {'mse': results[0], 'mae': results[1], 'rmse': rmse}
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def plot_training(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        plt.show()
    
    def save_model(self, filepath='models/biobot_cnn_lstm.h5'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/biobot_cnn_lstm.h5'):
        """Load trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
