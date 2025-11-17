"""
Unit and Integration Tests for MLOps Pipeline
Team 62 - Final Delivery

Tests cover:
- Pipeline creation and structure
- Data loading with DVC
- End-to-end prediction flow
- Model performance validation
- API endpoint functionality

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import sys
import os
sys.path.append(os.path.abspath('../..'))

from src.models.pipelines import create_full_pipeline
from src.data_utils import load_data
from sklearn.svm import SVR
import pandas as pd
import numpy as np


class TestPipelineCreation:
    """Test suite for pipeline creation and structure"""

    def test_pipeline_creation(self):
        """Test that pipeline can be created successfully"""
        model = SVR(kernel='rbf')
        pipeline = create_full_pipeline(model)

        assert pipeline is not None, "Pipeline should not be None"
        assert len(pipeline.steps) == 3, "Pipeline should have 3 steps: features, preprocessor, model"
        assert pipeline.steps[0][0] == 'features', "First step should be features"
        assert pipeline.steps[1][0] == 'preprocessor', "Second step should be preprocessor"
        assert pipeline.steps[2][0] == 'model', "Third step should be model"

    def test_pipeline_components(self):
        """Test that pipeline components are correctly configured"""
        model = SVR(kernel='rbf')
        pipeline = create_full_pipeline(model)

        # Check feature transformer
        feature_transformer = pipeline.steps[0][1]
        assert feature_transformer is not None

        # Check preprocessor
        preprocessor = pipeline.steps[1][1]
        assert preprocessor is not None

        # Check model
        final_model = pipeline.steps[2][1]
        assert isinstance(final_model, SVR)


class TestDataLoading:
    """Test suite for data loading functionality"""

    def test_data_loading(self):
        """Test data loading function with DVC integration"""
        df = load_data('data/processed/absenteeism_cleaned.csv')

        assert df is not None, "DataFrame should not be None"
        assert df.shape[0] > 0, "DataFrame should have rows"
        assert df.shape[1] == 21, "DataFrame should have 21 columns"

    def test_data_columns(self):
        """Test that required columns exist in loaded data"""
        df = load_data('data/processed/absenteeism_cleaned.csv')

        required_columns = [
            'Age', 'Distance from Residence to Work',
            'Service time', 'Work load Average/day',
            'Absenteeism time in hours'
        ]

        for col in required_columns:
            assert col in df.columns, f"Column '{col}' should exist in DataFrame"

    def test_data_quality(self):
        """Test data quality - no missing values in critical columns"""
        df = load_data('data/processed/absenteeism_cleaned.csv')

        # Target should not have missing values
        assert df['Absenteeism time in hours'].isna().sum() == 0, "Target should have no missing values"

        # Check reasonable ranges
        assert df['Age'].min() >= 18, "Age should be at least 18"
        assert df['Age'].max() <= 100, "Age should be reasonable"
        assert df['Absenteeism time in hours'].min() >= 0, "Absenteeism should be non-negative"


class TestPipelinePrediction:
    """Test suite for end-to-end pipeline predictions"""

    def test_pipeline_fit_predict(self):
        """Test end-to-end pipeline fit and predict"""
        # Load data
        df = load_data('data/processed/absenteeism_cleaned.csv')
        X = df.drop('Absenteeism time in hours', axis=1)
        y = df['Absenteeism time in hours']

        # Create and train pipeline
        pipeline = create_full_pipeline(SVR(kernel='rbf'))
        X_sample = X.head(100)
        y_sample = y.head(100)
        pipeline.fit(X_sample, y_sample)

        # Test prediction
        predictions = pipeline.predict(X_sample[:10])

        assert len(predictions) == 10, "Should predict for all 10 samples"
        assert all(pred >= 0 for pred in predictions), "Predictions should be non-negative"
        assert all(pred < 200 for pred in predictions), "Predictions should be reasonable"

    def test_prediction_shape(self):
        """Test that prediction output has correct shape"""
        df = load_data('data/processed/absenteeism_cleaned.csv')
        X = df.drop('Absenteeism time in hours', axis=1)
        y = df['Absenteeism time in hours']

        pipeline = create_full_pipeline(SVR(kernel='rbf'))
        pipeline.fit(X.head(50), y.head(50))

        # Single prediction
        single_pred = pipeline.predict(X.iloc[[0]])
        assert len(single_pred) == 1

        # Multiple predictions
        multi_pred = pipeline.predict(X.iloc[:5])
        assert len(multi_pred) == 5


class TestModelPerformance:
    """Test suite for model performance validation"""

    def test_model_achieves_target_mae(self):
        """Test that model achieves target MAE < 4.0 hours"""
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split

        df = load_data('data/processed/absenteeism_cleaned.csv')
        X = df.drop('Absenteeism time in hours', axis=1)
        y = df['Absenteeism time in hours']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = create_full_pipeline(SVR(kernel='rbf'))
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        assert mae < 4.0, f"MAE should be < 4.0, but got {mae:.2f}"
        print(f"\nAchieved MAE: {mae:.3f} hours")

    def test_model_generalization(self):
        """Test that model generalizes well (train/test gap)"""
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split

        df = load_data('data/processed/absenteeism_cleaned.csv')
        X = df.drop('Absenteeism time in hours', axis=1)
        y = df['Absenteeism time in hours']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = create_full_pipeline(SVR(kernel='rbf'))
        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        gap = abs(test_mae - train_mae)
        assert gap < 2.0, f"Train/test MAE gap should be < 2.0, but got {gap:.2f}"


class TestAPIEndpoints:
    """Test suite for API endpoint validation (requires running API)"""

    @pytest.mark.skipif(True, reason="Requires running API server")
    def test_health_endpoint(self):
        """Test API health endpoint"""
        import requests

        response = requests.get('http://localhost:5000/health')
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] in ['healthy', 'unhealthy']

    @pytest.mark.skipif(True, reason="Requires running API server")
    def test_predict_endpoint(self):
        """Test API predict endpoint"""
        import requests

        sample_data = {
            "ID": 1,
            "Reason for absence": 23,
            "Month of absence": 7,
            "Day of the week": 3,
            "Seasons": 1,
            "Transportation expense": 289,
            "Distance from Residence to Work": 36,
            "Service time": 13,
            "Age": 33,
            "Work load Average/day": 239.554,
            "Hit target": 97,
            "Disciplinary failure": 0,
            "Education": 1,
            "Son": 2,
            "Social drinker": 1,
            "Social smoker": 0,
            "Pet": 1,
            "Weight": 90,
            "Height": 172,
            "Body mass index": 30
        }

        response = requests.post(
            'http://localhost:5000/predict',
            json=sample_data
        )

        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert isinstance(data['prediction'], (int, float))
        assert data['prediction'] >= 0


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring external resources"
    )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
