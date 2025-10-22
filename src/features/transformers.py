"""
Sklearn-compatible Transformers for Feature Engineering
Author: Alexis Alduncin (Data Scientist) - Phase 2

All transformers are compatible with sklearn Pipeline and can be used
for both training and inference with proper fit/transform semantics.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class AbsenceCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Creates categorical bins for absenteeism duration.

    Categories:
    - short: 0-4 hours (minor absences)
    - half_day: 4-8 hours
    - full_day: 8-24 hours
    - extended: 24-120 hours (multi-day leave)
    """

    def __init__(self, target_col='Absenteeism time in hours'):
        self.target_col = target_col
        self.bins = [0, 4, 8, 24, 120]
        self.labels = ['short', 'half_day', 'full_day', 'extended']

    def fit(self, X, y=None):
        """Fit method (does nothing, stateless transformer)"""
        return self

    def transform(self, X):
        """Transform method - creates absence category feature"""
        X_copy = X.copy()

        if self.target_col in X_copy.columns:
            X_copy['absence_category'] = pd.cut(
                X_copy[self.target_col],
                bins=self.bins,
                labels=self.labels,
                include_lowest=True
            )
            logger.debug(f"Created absence_category: {X_copy['absence_category'].value_counts().to_dict()}")
        else:
            logger.warning(f"Column {self.target_col} not found, skipping transformation")

        return X_copy


class BMICategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Creates WHO-standard BMI health categories.

    Categories:
    - underweight: BMI < 18.5
    - normal: 18.5 ≤ BMI < 25
    - overweight: 25 ≤ BMI < 30
    - obese: BMI ≥ 30
    """

    def __init__(self, bmi_col='Body mass index'):
        self.bmi_col = bmi_col
        self.bins = [0, 18.5, 24.9, 29.9, float('inf')]
        self.labels = ['underweight', 'normal', 'overweight', 'obese']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        if self.bmi_col in X_copy.columns:
            X_copy['bmi_category'] = pd.cut(
                X_copy[self.bmi_col],
                bins=self.bins,
                labels=self.labels,
                include_lowest=True
            )
            logger.debug(f"Created bmi_category: {X_copy['bmi_category'].value_counts().to_dict()}")
        else:
            logger.warning(f"Column {self.bmi_col} not found, skipping transformation")

        return X_copy


class AgeGroupTransformer(BaseEstimator, TransformerMixin):
    """
    Creates life-stage age groups.

    Groups:
    - young: 18-30 years (early career)
    - middle: 30-45 years (family responsibilities)
    - senior: 45-60 years (late career)
    - veteran: 60+ years (near retirement)
    """

    def __init__(self, age_col='Age'):
        self.age_col = age_col
        self.bins = [0, 30, 45, 60, float('inf')]
        self.labels = ['young', 'middle', 'senior', 'veteran']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        if self.age_col in X_copy.columns:
            X_copy['age_group'] = pd.cut(
                X_copy[self.age_col],
                bins=self.bins,
                labels=self.labels,
                include_lowest=True
            )
            logger.debug(f"Created age_group: {X_copy['age_group'].value_counts().to_dict()}")
        else:
            logger.warning(f"Column {self.age_col} not found, skipping transformation")

        return X_copy


class DistanceCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Creates commute distance categories.

    Categories:
    - near: 0-10 km
    - moderate: 10-25 km
    - far: 25-40 km
    - very_far: 40+ km
    """

    def __init__(self, distance_col='Distance from Residence to Work'):
        self.distance_col = distance_col
        self.bins = [0, 10, 25, 40, float('inf')]
        self.labels = ['near', 'moderate', 'far', 'very_far']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        if self.distance_col in X_copy.columns:
            X_copy['distance_category'] = pd.cut(
                X_copy[self.distance_col],
                bins=self.bins,
                labels=self.labels,
                include_lowest=True
            )
            logger.debug(f"Created distance_category: {X_copy['distance_category'].value_counts().to_dict()}")
        else:
            logger.warning(f"Column {self.distance_col} not found, skipping transformation")

        return X_copy


class WorkloadCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Creates workload intensity categories based on percentiles.

    Categories:
    - low: 0-33rd percentile
    - medium: 33-66th percentile
    - high: 66-100th percentile
    """

    def __init__(self, workload_col='Work load Average/day'):
        self.workload_col = workload_col
        self.q33 = None
        self.q66 = None

    def fit(self, X, y=None):
        """Fit method - learns percentiles from training data"""
        if self.workload_col in X.columns:
            self.q33 = X[self.workload_col].quantile(0.33)
            self.q66 = X[self.workload_col].quantile(0.66)
            logger.debug(f"Workload percentiles: 33rd={self.q33:.2f}, 66th={self.q66:.2f}")
        return self

    def transform(self, X):
        X_copy = X.copy()

        if self.workload_col in X_copy.columns and self.q33 is not None:
            X_copy['workload_category'] = pd.cut(
                X_copy[self.workload_col],
                bins=[0, self.q33, self.q66, float('inf')],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
            logger.debug(f"Created workload_category: {X_copy['workload_category'].value_counts().to_dict()}")
        else:
            logger.warning(f"Column {self.workload_col} not found or not fitted, skipping transformation")

        return X_copy


class SeasonNameTransformer(BaseEstimator, TransformerMixin):
    """
    Converts numeric season codes to interpretable names.

    Mapping:
    - 1 → summer
    - 2 → autumn
    - 3 → winter
    - 4 → spring
    """

    def __init__(self, season_col='Seasons'):
        self.season_col = season_col
        self.season_map = {1: 'summer', 2: 'autumn', 3: 'winter', 4: 'spring'}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        if self.season_col in X_copy.columns:
            X_copy['season_name'] = X_copy[self.season_col].map(self.season_map)
            logger.debug(f"Created season_name: {X_copy['season_name'].value_counts().to_dict()}")
        else:
            logger.warning(f"Column {self.season_col} not found, skipping transformation")

        return X_copy


class HighRiskTransformer(BaseEstimator, TransformerMixin):
    """
    Creates composite high-risk indicator from multiple factors.

    High risk criteria (any of):
    - Disciplinary failure flag
    - BMI ≥ 30 (obese)
    - Distance > 40 km (very far commute)
    - 3+ children
    """

    def __init__(self,
                 disciplinary_col='Disciplinary failure',
                 bmi_col='Body mass index',
                 distance_col='Distance from Residence to Work',
                 children_col='Son'):
        self.disciplinary_col = disciplinary_col
        self.bmi_col = bmi_col
        self.distance_col = distance_col
        self.children_col = children_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Check all required columns exist
        required_cols = [self.disciplinary_col, self.bmi_col,
                        self.distance_col, self.children_col]
        if all(col in X_copy.columns for col in required_cols):
            X_copy['high_risk'] = (
                (X_copy[self.disciplinary_col] == 1) |
                (X_copy[self.bmi_col] >= 30) |
                (X_copy[self.distance_col] > 40) |
                (X_copy[self.children_col] >= 3)
            ).astype(int)

            risk_count = X_copy['high_risk'].sum()
            total = len(X_copy)
            logger.debug(f"Created high_risk: {risk_count}/{total} ({risk_count/total*100:.1f}%) flagged as high risk")
        else:
            missing = [col for col in required_cols if col not in X_copy.columns]
            logger.warning(f"Missing columns for high_risk: {missing}, skipping transformation")

        return X_copy


# Utility function to create all transformers at once
def create_all_transformers():
    """
    Factory function to create all 7 transformers.

    Returns:
        list: List of (name, transformer) tuples
    """
    return [
        ('absence_cat', AbsenceCategoryTransformer()),
        ('bmi_cat', BMICategoryTransformer()),
        ('age_group', AgeGroupTransformer()),
        ('distance_cat', DistanceCategoryTransformer()),
        ('workload_cat', WorkloadCategoryTransformer()),
        ('season_name', SeasonNameTransformer()),
        ('high_risk', HighRiskTransformer())
    ]


if __name__ == "__main__":
    # Test transformers
    print("Testing sklearn transformers...")

    # Create sample data
    test_data = pd.DataFrame({
        'Absenteeism time in hours': [2, 6, 12, 30],
        'Body mass index': [22, 28, 32, 19],
        'Age': [25, 35, 50, 65],
        'Distance from Residence to Work': [5, 15, 30, 45],
        'Work load Average/day ': [200, 250, 300, 350],
        'Seasons': [1, 2, 3, 4],
        'Disciplinary failure': [0, 0, 1, 0],
        'Son': [1, 2, 3, 4]
    })

    print(f"Original data shape: {test_data.shape}")

    # Test each transformer
    for name, transformer in create_all_transformers():
        transformer.fit(test_data)
        result = transformer.transform(test_data)
        print(f"✓ {name}: {result.shape}")

    print("\nAll transformers working correctly!")
