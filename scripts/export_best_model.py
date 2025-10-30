"""
Export Best Model for Deployment
Author: Alexis Alduncin (Data Scientist)
Team: MLOps 62

Exports the best performing model from Phase 2 experiments for production deployment.
Reads experiment results, retrains the best model, and saves it with metadata.

Usage:
    python scripts/export_best_model.py

Output:
    - models/best_model_svr.pkl - Trained model pipeline
    - models/model_metadata.json - Model performance metrics
    - models/feature_names.txt - List of input features
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_experiment_results(results_path='experiments/baseline_results.csv'):
    """
    Load experiment results and identify best model.

    Args:
        results_path: Path to experiment results CSV

    Returns:
        dict: Best model information
    """
    logger.info(f"Loading experiment results from {results_path}...")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Experiment results not found: {results_path}")

    results = pd.read_csv(results_path)

    # Best model is the one with lowest Test_MAE
    best_idx = results['Test_MAE'].idxmin()
    best_model = results.iloc[best_idx]

    logger.info(f"\n{'='*60}")
    logger.info(f"Best Model Identified: {best_model['Model']}")
    logger.info(f"{'='*60}")
    logger.info(f"Test MAE:  {best_model['Test_MAE']:.3f} hours")
    logger.info(f"Test RMSE: {best_model['Test_RMSE']:.3f} hours")
    logger.info(f"Test R²:   {best_model['Test_R2']:.3f}")
    logger.info(f"CV MAE:    {best_model['CV_MAE']:.3f} hours")
    logger.info(f"{'='*60}\n")

    return {
        'model_name': best_model['Model'],
        'test_mae': float(best_model['Test_MAE']),
        'test_rmse': float(best_model['Test_RMSE']),
        'test_r2': float(best_model['Test_R2']),
        'cv_mae': float(best_model['CV_MAE']),
        'train_mae': float(best_model['Train_MAE'])
    }


def prepare_data(data_path, test_size=0.2, random_state=42):
    """
    Load and prepare data for training.

    Args:
        data_path: Path to cleaned data
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    logger.info(f"Loading data from {data_path}...")

    # Try multiple possible data paths
    possible_paths = [
        data_path,
        '../mlops-absenteeism-project/data/processed/absenteeism_cleaned.csv',
        'data/processed/work_absenteeism_clean_v1.0.csv',
        '../data/processed/absenteeism_cleaned.csv'
    ]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            logger.info(f"✅ Data loaded from: {path}")
            break

    if df is None:
        raise FileNotFoundError(f"Could not find data file. Tried: {possible_paths}")

    # Target variable
    target = 'Absenteeism time in hours'

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    # Split features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Remove any NaN values
    valid_idx = y.notna() & X.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    feature_names = X.columns.tolist()

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Number of features: {len(feature_names)}")
    logger.info(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}, median={y.median():.2f}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_names


def create_best_model():
    """
    Create the best performing model configuration.
    Based on experiments, this is SVR with RBF kernel.

    Returns:
        Pipeline: Sklearn pipeline with preprocessing and model
    """
    logger.info("Creating SVR model with RBF kernel...")

    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1,
            gamma='scale',
            cache_size=200
        ))
    ])

    return pipeline


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Train model and evaluate performance.

    Args:
        model: Model to train
        X_train, X_test, y_train, y_test: Training and test data

    Returns:
        tuple: (trained_model, metrics_dict)
    """
    logger.info("Training model...")

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Cross-validation MAE
    logger.info("Running cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    cv_mae = -cv_scores.mean()

    metrics = {
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'cv_mae': float(cv_mae),
        'cv_mae_std': float(cv_scores.std())
    }

    logger.info(f"\n{'='*60}")
    logger.info("Model Performance:")
    logger.info(f"{'='*60}")
    logger.info(f"Train MAE:     {train_mae:.3f} hours")
    logger.info(f"Test MAE:      {test_mae:.3f} hours")
    logger.info(f"Test RMSE:     {test_rmse:.3f} hours")
    logger.info(f"Test R²:       {test_r2:.3f}")
    logger.info(f"CV MAE (5-fold): {cv_mae:.3f} ± {cv_scores.std():.3f} hours")
    logger.info(f"{'='*60}\n")

    return model, metrics


def save_model_artifacts(model, metadata, feature_names, output_dir='models'):
    """
    Save model, metadata, and feature information.

    Args:
        model: Trained model pipeline
        metadata: Model metadata dictionary
        feature_names: List of feature names
        output_dir: Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, 'best_model_svr.pkl')
    logger.info(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    logger.info(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save feature names
    features_path = os.path.join(output_dir, 'feature_names.txt')
    logger.info(f"Saving feature names to {features_path}...")
    with open(features_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    logger.info(f"\n{'='*60}")
    logger.info("✅ Model Export Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Model file:     {model_path}")
    logger.info(f"Metadata file:  {metadata_path}")
    logger.info(f"Features file:  {features_path}")
    logger.info(f"{'='*60}\n")


def export_best_model():
    """
    Main function to export best model for deployment.
    """
    try:
        # Load experiment results
        best_model_info = load_experiment_results()

        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = prepare_data(
            data_path='../mlops-absenteeism-project/data/processed/absenteeism_cleaned.csv'
        )

        # Create and train best model
        model = create_best_model()
        trained_model, metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)

        # Prepare metadata
        metadata = {
            'model_type': 'SVR',
            'model_name': 'Support Vector Regression (RBF kernel)',
            'model_config': {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1,
                'gamma': 'scale'
            },
            'preprocessing': 'StandardScaler',
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(feature_names),
            **metrics,
            'experiment_results': best_model_info,
            'target_variable': 'Absenteeism time in hours',
            'unit': 'hours',
            'phase': 'Phase 2',
            'author': 'Alexis Alduncin',
            'team': 'MLOps 62'
        }

        # Save all artifacts
        save_model_artifacts(trained_model, metadata, feature_names)

        logger.info("\n🎉 Model ready for deployment!")
        logger.info(f"Expected MAE in production: {metrics['test_mae']:.2f} ± {metrics['cv_mae_std']:.2f} hours\n")

        return trained_model, metadata

    except Exception as e:
        logger.error(f"\n❌ Error exporting model: {str(e)}")
        raise


if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("EXPORTING BEST MODEL FOR DEPLOYMENT")
    logger.info("="*60 + "\n")

    model, metadata = export_best_model()

    logger.info("Done! You can now run the deployment API:")
    logger.info("  cd deployment")
    logger.info("  python app.py")
