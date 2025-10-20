"""
Baseline Model Experiments with MLflow Tracking
Author: Alexis Alduncin (Data Scientist) - Phase 2

Runs 10+ regression models to establish baseline performance.
Target: Improve MAE from 5.44 (Phase 1) to <4.0 hours

Models tested:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. LightGBM
9. Support Vector Regression (SVR)
10. K-Nearest Neighbors (KNN)

All experiments logged to MLflow with:
- Parameters
- Metrics (MAE, RMSE, R²)
- Model artifacts
- Cross-validation scores
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import logging
import warnings
warnings.filterwarnings('ignore')

from src.data_utils import load_data
from src.models.pipelines import create_full_pipeline, create_pipeline_from_config
from src.config import MLFLOW_EXPERIMENT_NAME

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_data(test_size=0.2, random_state=42):
    """
    Load and split data for experiments.

    Args:
        test_size (float): Test set proportion
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("Loading data...")
    # Use Phase 1 cleaned data (has proper outlier handling - max 120 hours)
    df = pd.read_csv('../mlops-absenteeism-project/data/processed/absenteeism_cleaned.csv')

    # Target variable
    target = 'Absenteeism time in hours'

    # Split features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Drop rows with NaN in target or features
    valid_idx = y.notna() & X.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}, median={y.median():.2f}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=5):
    """
    Evaluate model performance with multiple metrics.

    Args:
        model: Fitted sklearn model/pipeline
        X_train, X_test, y_train, y_test: Train/test data
        cv_folds (int): Number of cross-validation folds

    Returns:
        dict: Dictionary of metrics
    """
    # Test set predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Calculate metrics
    metrics = {
        # Test set metrics (primary)
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),

        # Train set metrics (check for overfitting)
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_r2': r2_score(y_train, y_pred_train),
    }

    # Cross-validation MAE (on training data)
    try:
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        metrics['cv_mae_mean'] = -cv_scores.mean()
        metrics['cv_mae_std'] = cv_scores.std()
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        metrics['cv_mae_mean'] = None
        metrics['cv_mae_std'] = None

    # Overfitting indicator
    metrics['overfit_gap'] = metrics['train_mae'] - metrics['test_mae']

    return metrics


def run_experiment(model_name, model, X_train, X_test, y_train, y_test,
                   pipeline_config='full', log_model=True):
    """
    Run single experiment with MLflow tracking.

    Args:
        model_name (str): Name for MLflow run
        model: Sklearn estimator
        X_train, X_test, y_train, y_test: Data splits
        pipeline_config (str): Pipeline configuration ('full', 'baseline', etc.)
        log_model (bool): Whether to log model artifact

    Returns:
        dict: Metrics dictionary
    """
    with mlflow.start_run(run_name=model_name):
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: {model_name}")
        logger.info(f"Pipeline config: {pipeline_config}")
        logger.info(f"{'='*60}")

        # Log parameters
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("pipeline_config", pipeline_config)

        # Log model hyperparameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param, value in params.items():
                mlflow.log_param(f"model_{param}", value)

        # Create pipeline
        pipeline = create_pipeline_from_config(model, config_name=pipeline_config)

        # Fit model with timing
        logger.info("Fitting model...")
        import time
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Log training time
        mlflow.log_metric("training_time_seconds", training_time)

        # Evaluate
        logger.info("Evaluating model...")
        metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)

        # Log all metrics
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)

        # Log feature importance for tree-based models
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances = pipeline.named_steps['model'].feature_importances_
            # Log top 10 most important features
            top_indices = importances.argsort()[-10:][::-1]
            for i, idx in enumerate(top_indices):
                mlflow.log_metric(f"feature_importance_rank_{i+1}", float(importances[idx]))
            logger.info(f"  Feature importance logged (top 10 features)")

        # Log model artifact
        if log_model:
            mlflow.sklearn.log_model(pipeline, "model")

        # Print results
        logger.info(f"\n{'─'*60}")
        logger.info(f"Results for {model_name}:")
        logger.info(f"  Test MAE:  {metrics['test_mae']:.4f} hours")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f} hours")
        logger.info(f"  Test R²:   {metrics['test_r2']:.4f}")
        logger.info(f"  Train MAE: {metrics['train_mae']:.4f} hours")
        if metrics['cv_mae_mean'] is not None:
            logger.info(f"  CV MAE:    {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}")
        logger.info(f"  Overfit:   {metrics['overfit_gap']:.4f} hours")
        logger.info(f"  Time:      {training_time:.2f} seconds")
        logger.info(f"{'─'*60}")

        # Add training time to metrics
        metrics['training_time'] = training_time

        return metrics


def run_all_baseline_experiments():
    """
    Run all baseline experiments and return results summary.

    Returns:
        pd.DataFrame: Results summary table
    """
    # Setup MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # Define models to test (with anti-overfitting settings)
    models = [
        ('LinearRegression', LinearRegression()),
        ('Ridge_alpha1', Ridge(alpha=1.0, random_state=42)),
        ('Ridge_alpha10', Ridge(alpha=10.0, random_state=42)),
        ('Lasso_alpha0.1', Lasso(alpha=0.1, random_state=42)),
        ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),

        # Random Forest - prevent overfitting with max_depth and min_samples_leaf
        ('RandomForest_depth5', RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=42, n_jobs=-1
        )),
        ('RandomForest_depth7', RandomForestRegressor(
            n_estimators=100, max_depth=7, min_samples_leaf=10, random_state=42, n_jobs=-1
        )),

        # Gradient Boosting - conservative learning rate
        ('GradientBoosting', GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
        )),

        # XGBoost - conservative settings
        ('XGBoost_conservative', xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )),
        ('XGBoost_aggressive', xgb.XGBRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1
        )),

        # LightGBM - conservative settings
        ('LightGBM_conservative', lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=5, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )),
        ('LightGBM_aggressive', lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=6, num_leaves=50,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1, verbose=-1
        )),

        ('SVR_rbf', SVR(kernel='rbf', C=1.0)),
        ('KNN_5', KNeighborsRegressor(n_neighbors=5)),
        ('KNN_10', KNeighborsRegressor(n_neighbors=10))
    ]

    results = []

    # Run experiments
    for model_name, model in models:
        try:
            metrics = run_experiment(
                model_name=model_name,
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                pipeline_config='full',
                log_model=True
            )

            results.append({
                'Model': model_name,
                'Test_MAE': metrics['test_mae'],
                'Test_RMSE': metrics['test_rmse'],
                'Test_R2': metrics['test_r2'],
                'Train_MAE': metrics['train_mae'],
                'CV_MAE': metrics.get('cv_mae_mean'),
                'Overfit_Gap': metrics['overfit_gap'],
                'Time_sec': metrics.get('training_time', 0)
            })

        except Exception as e:
            logger.error(f"Failed to run {model_name}: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_MAE')

    return results_df


def main():
    """Main execution function"""
    logger.info("\n" + "="*80)
    logger.info("BASELINE EXPERIMENTS - PHASE 2")
    logger.info("Goal: Improve MAE from 5.44 (Phase 1) to <4.0 hours")
    logger.info("="*80 + "\n")

    # Run all experiments
    results_df = run_all_baseline_experiments()

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("="*80 + "\n")

    print(results_df.to_string(index=False))

    # Find best model
    best_model = results_df.iloc[0]
    logger.info(f"\n{'='*80}")
    logger.info(f"BEST MODEL: {best_model['Model']}")
    logger.info(f"  Test MAE: {best_model['Test_MAE']:.4f} hours")
    logger.info(f"  Test R²:  {best_model['Test_R2']:.4f}")

    # Check if goal achieved
    if best_model['Test_MAE'] < 4.0:
        logger.info(f"\n✅ GOAL ACHIEVED! MAE = {best_model['Test_MAE']:.4f} < 4.0 hours")
    else:
        logger.info(f"\n⚠️  Goal not yet achieved. MAE = {best_model['Test_MAE']:.4f} >= 4.0 hours")
        logger.info("   Proceed to hyperparameter tuning phase.")

    logger.info(f"{'='*80}\n")

    # Save results
    output_path = 'experiments/baseline_results.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    main()
