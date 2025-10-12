"""
Feature Engineering Pipeline for Absenteeism Prediction
Author: Alexis Alduncin (Data Scientist)

This module contains all feature transformation and engineering functions
for the absenteeism prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import logging

from src import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class AbsenteeismFeatureEngine:
    """
    Feature engineering pipeline for absenteeism dataset.

    Handles data cleaning, transformation, and feature creation.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data by removing outliers and handling invalid values.

        Args:
            df: Raw dataframe

        Returns:
            Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()

        # Remove extreme outliers in target variable
        initial_rows = len(df_clean)
        df_clean = df_clean[df_clean[config.TARGET_COLUMN] <= config.OUTLIER_THRESHOLD]
        removed_rows = initial_rows - len(df_clean)
        logger.info(f"Removed {removed_rows} outliers (>{config.OUTLIER_THRESHOLD} hours)")

        # Handle zero values in 'Reason for absence' (0 means no specific reason)
        df_clean['Reason for absence'] = df_clean['Reason for absence'].replace(0, 28)

        # Remove rows with invalid 'Day of the week' values
        df_clean = df_clean[df_clean['Day of the week'].between(2, 6)]

        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        return df_clean

    def create_absence_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical bins for absence duration."""
        df = df.copy()
        df['Absence_Category'] = pd.cut(
            df[config.TARGET_COLUMN],
            bins=config.ABSENCE_BINS,
            labels=config.ABSENCE_LABELS,
            include_lowest=True
        )
        logger.info("Created Absence_Category feature")
        return df

    def create_bmi_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create WHO-standard BMI categories."""
        df = df.copy()
        df['BMI_Category'] = pd.cut(
            df['Body mass index'],
            bins=config.BMI_BINS,
            labels=config.BMI_LABELS,
            include_lowest=True
        )
        logger.info("Created BMI_Category feature")
        return df

    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-based groups."""
        df = df.copy()
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=config.AGE_BINS,
            labels=config.AGE_LABELS,
            include_lowest=True
        )
        logger.info("Created Age_Group feature")
        return df

    def create_distance_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create distance from work categories."""
        df = df.copy()
        df['Distance_Category'] = pd.cut(
            df['Distance from Residence to Work'],
            bins=config.DISTANCE_BINS,
            labels=config.DISTANCE_LABELS,
            include_lowest=True
        )
        logger.info("Created Distance_Category feature")
        return df

    def create_workload_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create workload categories."""
        df = df.copy()
        df['Workload_Category'] = pd.cut(
            df['Work load Average/day'],
            bins=config.WORKLOAD_BINS,
            labels=config.WORKLOAD_LABELS,
            include_lowest=True
        )
        logger.info("Created Workload_Category feature")
        return df

    def create_season_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert season codes to names."""
        df = df.copy()
        season_map = {1: 'Summer', 2: 'Autumn', 3: 'Winter', 4: 'Spring'}
        df['Season_Name'] = df['Seasons'].map(season_map)
        logger.info("Created Season_Name feature")
        return df

    def create_high_risk_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite high-risk indicator based on multiple factors.

        High risk defined as:
        - Disciplinary failure OR
        - BMI > 30 (Obese) OR
        - Distance > 40km OR
        - Multiple children (Son > 2)
        """
        df = df.copy()
        df['High_Risk'] = (
            (df['Disciplinary failure'] == 1) |
            (df['Body mass index'] > 30) |
            (df['Distance from Residence to Work'] > 40) |
            (df['Son'] > 2)
        ).astype(int)
        logger.info("Created High_Risk feature")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.

        Args:
            df: Cleaned dataframe

        Returns:
            Dataframe with engineered features
        """
        logger.info("Starting feature engineering...")

        df = self.create_absence_categories(df)
        df = self.create_bmi_categories(df)
        df = self.create_age_groups(df)
        df = self.create_distance_categories(df)
        df = self.create_workload_categories(df)
        df = self.create_season_names(df)
        df = self.create_high_risk_flag(df)

        initial_features = len([c for c in df.columns if c in config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES])
        new_features = len(df.columns) - initial_features
        logger.info(f"Feature engineering complete. Created {new_features} new features")

        return df

    def prepare_for_modeling(
        self,
        df: pd.DataFrame,
        scale_features: bool = True,
        monthly_model: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning models.

        Args:
            df: Dataframe with engineered features
            scale_features: Whether to apply feature scaling

        Returns:
            Tuple of (features, target)
        """
        logger.info("Preparing data for modeling...")

        # Separate features and target
        if not monthly_model:
            X = df.drop(config.FEATURES_TO_DROP, axis=1, errors='ignore')
            y = df[config.TARGET_COLUMN]
        else:
            X = df.drop(config.FEATURES_TO_DROP_MONTHLY, axis=1, errors='ignore')
            y = df[config.TARGET_COLUMN_MONTHLY]

        # Drop the categorical feature columns (keep encoded versions)
        categorical_feature_cols = [
            'Absence_Category', 'BMI_Category', 'Age_Group',
            'Distance_Category', 'Workload_Category', 'Season_Name'
        ]
        X = X.drop(categorical_feature_cols, axis=1, errors='ignore')

        # Encode remaining categorical variables
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Scale features if requested
        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            logger.info("Applied feature scaling")

        self.feature_names = X.columns.tolist()
        logger.info(f"Data preparation complete. Features: {len(self.feature_names)}")

        return X, y

    def get_feature_importance(self, model, top_n: int = 15) -> pd.DataFrame:
        """
        Extract feature importance from trained model.

        Args:
            model: Trained sklearn model with feature_importances_ attribute
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        logger.info(f"Extracted top {top_n} feature importances")
        return importance_df.head(top_n)


def full_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Execute full feature engineering pipeline.

    Args:
        df: Raw dataframe

    Returns:
        Tuple of (features, target, feature_engineered_df) ready for modeling
    """
    engine = AbsenteeismFeatureEngine()

    # Clean data
    df_clean = engine.clean_data(df)

    # Engineer features
    df_features = engine.engineer_features(df_clean)

    # Prepare for modeling
    X, y = engine.prepare_for_modeling(df_features, scale_features=True)

    return X, y, df_features


if __name__ == "__main__":
    # Example usage
    from src.data_utils import load_data

    print("Feature Engineering Pipeline")
    print("=" * 50)

    # Load data using team's robust DVC approach
    df = load_data()
    print(f"Loaded {len(df)} rows")

    # Run pipeline
    X, y, df_features = full_pipeline(df)
    print(f"Final dataset: {X.shape[1]} features, {len(y)} samples")
    print(f"\nNew features created:")
    new_cols = set(df_features.columns) - set(df.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
