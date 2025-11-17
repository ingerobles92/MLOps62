"""
Script to verify model reproducibility across environments
Ensures consistent results with fixed random seeds
"""

import numpy as np
import pandas as pd
import random
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import os

# Fix all random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def verify_model_reproducibility():
    """Verify model produces consistent results"""

    print("Verifying Model Reproducibility...")
    print("-" * 50)

    # Load data
    data_path = '../../data/processed/absenteeism_cleaned.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return False

    df = pd.read_csv(data_path)
    X = df.drop('Absenteeism time in hours', axis=1)
    y = df['Absenteeism time in hours']

    # Split with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Load saved model
    model_path = '../../models/best_model_svr.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f"\nModel Performance:")
    print(f"MAE: {mae:.3f} hours")
    print(f"Target: < 4.0 hours")
    print(f"Status: {'PASS' if mae < 4.0 else 'FAIL'}")

    # Verify consistency
    expected_mae = 3.83  # Expected from training
    tolerance = 0.1

    if abs(mae - expected_mae) < tolerance:
        print(f"\nReproducibility verified!")
        print(f"  MAE within tolerance: {mae:.3f} ~= {expected_mae:.3f}")
        return True
    else:
        print(f"\nReproducibility failed!")
        print(f"  MAE deviation: {mae:.3f} vs {expected_mae:.3f}")
        return False

if __name__ == "__main__":
    verify_model_reproducibility()
