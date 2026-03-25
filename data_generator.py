"""
BioBot - CNN-LSTM Hybrid Model for Livable Area Prediction
Predicts livable area based on biosense data and human factors
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

class BioSenseDataGenerator:
    def __init__(self, n_samples=10000, seq_length=24):
        self.n_samples = n_samples
        self.seq_length = seq_length
        
    def generate_sequence(self):
        """Generate a single sequence of biosense data"""
        seq = []
        for _ in range(self.seq_length):
            temp = np.random.uniform(15, 40)
            humidity = np.random.uniform(20, 95)
            wind = np.random.uniform(0, 30)
            solar_rad = np.random.uniform(0, 1000)
            
            humidex = self.compute_humidex(temp, humidity)
            activity = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            clothing = np.random.uniform(0.1, 1.0)
            
            livable = self.compute_livable_score(
                temp, humidity, humidex, wind, solar_rad, activity, clothing
            )
            
            seq.append([temp, humidity, humidex, wind, solar_rad, activity, clothing, livable])
        return np.array(seq)
    
    def compute_humidex(self, temp, humidity):
        """Calculate humidex (feels-like temperature)"""
        h = humidity / 100
        e = 6.11 * np.exp(5417.7530 * (1/273.15 - 1/(temp + 273.15)))
        dew_point = (-430.22 + 237.7 * np.log(e)) / (-np.log(e) + 19.08)
        humidex = temp + 0.5555 * (h * 6.11 * np.exp(5417.7530 * (1/273.15 - 1/(dew_point + 273.15))) - 10)
        return max(temp, humidex)
    
    def compute_livable_score(self, temp, humidity, humidex, wind, solar_rad, activity, clothing):
        """Compute livable area score (0-100)"""
        temp_score = 1 - abs(temp - 22) / 30
        humidity_score = 1 - abs(humidity - 50) / 70
        wind_score = 1 - wind / 40
        activity_penalty = (activity - 1) * 0.1
        clothing_penalty = (clothing - 0.3) * 0.15
        
        raw_score = (
            0.25 * temp_score + 
            0.20 * humidity_score + 
            0.25 * wind_score +
            0.30 * (1 - max(0, humidex - 30) / 20)
        )
        
        score = (raw_score - activity_penalty - clothing_penalty) * 100
        return np.clip(score, 0, 100)
    
    def generate_dataset(self):
        """Generate complete dataset"""
        X = np.zeros((self.n_samples, self.seq_length, 7))
        y = np.zeros((self.n_samples, 1))
        
        for i in range(self.n_samples):
            seq = self.generate_sequence()
            X[i, :, :7] = seq[:, :7]
            y[i] = seq[-1, 7]
            
        return X, y
    
    def save_dataset(self, X, y, filepath='data'):
        """Save dataset to files"""
        os.makedirs(filepath, exist_ok=True)
        np.save(f'{filepath}/X_train.npy', X[:int(self.n_samples*0.8)])
        np.save(f'{filepath}/y_train.npy', y[:int(self.n_samples*0.8)])
        np.save(f'{filepath}/X_test.npy', X[int(self.n_samples*0.8):])
        np.save(f'{filepath}/y_test.npy', y[int(self.n_samples*0.8):])
        print(f"Dataset saved: {self.n_samples} samples, {self.seq_length} timesteps, 7 features")
