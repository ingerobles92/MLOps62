"""
Sklearn Pipeline Factory Functions
Author: Alexis Alduncin (Data Scientist) - Phase 2

Three-layer pipeline architecture:
1. Preprocessing Pipeline: Handles imputation, scaling, encoding
2. Feature Pipeline: Applies custom transformers for engineered features
3. Full Pipeline: Combines preprocessing → features → model
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
import logging

# Add project root to path (handles both Docker /work and local environments)
import sys
import os
if os.path.exists('/work'):
    sys.path.insert(0, '/work')  # Docker environment
else:
    sys.path.insert(0, os.path.abspath('..'))  # Local environment


from src.features.transformers import (
    AbsenceCategoryTransformer,
    BMICategoryTransformer,
    AgeGroupTransformer,
    DistanceCategoryTransformer,
    WorkloadCategoryTransformer,
    SeasonNameTransformer,
    HighRiskTransformer
)

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(numeric_features=None, categorical_features=None, X=None):
    """
    Creates preprocessing pipeline for numeric and categorical features.

    Numeric features: Impute with median → StandardScaler
    Categorical features: Impute with mode → OneHotEncoder

    Args:
        numeric_features (list): List of numeric column names (if None, auto-detect)
        categorical_features (list): List of categorical column names (if None, auto-detect)
        X (DataFrame, optional): If provided, filter column lists to only include existing columns

    Returns:
        ColumnTransformer: Preprocessing pipeline

    Example:
        >>> numeric = ['Age', 'Body mass index', 'Distance from Residence to Work']
        >>> categorical = ['Education', 'Day of the week']
        >>> preprocessor = create_preprocessing_pipeline(numeric, categorical)
    """
    # Numeric pipeline: impute median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: impute most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Use automatic column detection if column lists not provided
    if numeric_features is None and categorical_features is None:
        # Auto-detect based on dtype
        transformers = [
            ('num', numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
            ('cat', categorical_transformer, make_column_selector(dtype_include=['object', 'category']))
        ]
        logger.info("Created preprocessing pipeline with auto-detected column types")
    else:
        # Use provided column lists (fallback for backward compatibility)
        if numeric_features is None:
            numeric_features = [
                'Age',
                'Body mass index',
                'Distance from Residence to Work',
                'Work load Average/day',
                'Service time',
                'Hit target',
                'Transportation expense',
                'Son',
                'Pet'
            ]

        if categorical_features is None:
            categorical_features = [
                'Education',
                'Day of the week',
                'Month of absence',
                'Reason for absence',
                'Seasons'
            ]

        # Filter to only include columns that exist in X (useful when X is provided)
        if X is not None:
            numeric_features = [col for col in numeric_features if col in X.columns]
            categorical_features = [col for col in categorical_features if col in X.columns]

        transformers = [
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
        logger.info(f"Created preprocessing pipeline: {len(numeric_features)} numeric, {len(categorical_features)} categorical")

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop columns not in numeric or categorical
        verbose_feature_names_out=False
    )

    # Configure to output pandas DataFrames instead of numpy arrays
    preprocessor.set_output(transform="pandas")

    return preprocessor


def create_feature_pipeline():
    """
    Creates feature engineering pipeline with 6 custom transformers.

    Transformers applied in sequence:
    1. BMICategoryTransformer - WHO BMI categories
    2. AgeGroupTransformer - Life-stage groups
    3. DistanceCategoryTransformer - Commute distance bins
    4. WorkloadCategoryTransformer - Workload percentiles (STATEFUL)
    5. SeasonNameTransformer - Season names
    6. HighRiskTransformer - Composite risk flag

    Note:
        - AbsenceCategoryTransformer is excluded (it bins the target variable, which would cause data leakage)
        - WorkloadCategoryTransformer is STATEFUL - it learns percentiles during fit
        - All others are stateless transformers
    """
    feature_pipeline = Pipeline(steps=[
        # Note: AbsenceCategoryTransformer excluded - it transforms the target, not features
        ('bmi_cat', BMICategoryTransformer()),
        ('age_group', AgeGroupTransformer()),
        ('distance_cat', DistanceCategoryTransformer()),
        ('workload_cat', WorkloadCategoryTransformer()),  # STATEFUL
        ('season_name', SeasonNameTransformer()),
        ('high_risk', HighRiskTransformer())
    ])

    logger.info("Created feature engineering pipeline with 6 transformers (excluding AbsenceCategoryTransformer)")

    return feature_pipeline


def create_full_pipeline(model, numeric_features=None, categorical_features=None,
                        apply_feature_engineering=True, apply_preprocessing=True):
    """
    Creates end-to-end ML pipeline: features → preprocessing → model.

    Pipeline flow:
    1. Feature Engineering: Apply custom transformers FIRST (if apply_feature_engineering=True)
    2. Preprocessing: Impute, scale, encode (if apply_preprocessing=True)
    3. Model: Fit and predict

    Note: Feature engineering is applied BEFORE preprocessing because feature transformers
    need access to original column names (e.g., 'Body mass index', 'Age'). After preprocessing,
    column names change due to scaling/encoding.

    Args:
        model: Sklearn-compatible estimator (e.g., LinearRegression, RandomForest)
        numeric_features (list): Numeric columns for preprocessing
        categorical_features (list): Categorical columns for preprocessing
        apply_feature_engineering (bool): Whether to apply custom transformers
        apply_preprocessing (bool): Whether to apply preprocessing

    Returns:
        Pipeline: End-to-end ML pipeline

    Examples:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=100, random_state=42)
        >>> pipeline = create_full_pipeline(model)
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)

        >>> # Pipeline without feature engineering (baseline)
        >>> pipeline_baseline = create_full_pipeline(model, apply_feature_engineering=False)
    """
    steps = []

    # Step 1: Feature Engineering FIRST (optional)
    # Must come before preprocessing so transformers can access original column names
    if apply_feature_engineering:
        feature_pipe = create_feature_pipeline()
        steps.append(('features', feature_pipe))
        logger.debug("Added feature engineering step to pipeline")

    # Step 2: Preprocessing (optional)
    if apply_preprocessing:
        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
        steps.append(('preprocessing', preprocessor))
        logger.debug("Added preprocessing step to pipeline")

    # Step 3: Model (always included)
    steps.append(('model', model))
    logger.debug(f"Added model step: {type(model).__name__}")

    # Create final pipeline
    full_pipeline = Pipeline(steps=steps)

    logger.info(f"Created full pipeline with {len(steps)} steps: {' → '.join([s[0] for s in steps])}")

    return full_pipeline


def get_feature_names_from_pipeline(pipeline, input_feature_names):
    """
    Extracts feature names after preprocessing transformations.

    Useful for:
    - Feature importance analysis
    - Debugging pipeline transformations
    - Understanding what features the model sees

    Args:
        pipeline: Fitted sklearn Pipeline
        input_feature_names (list): Original feature names

    Returns:
        list: Transformed feature names after preprocessing

    Example:
        >>> pipeline.fit(X_train, y_train)
        >>> feature_names = get_feature_names_from_pipeline(pipeline, X_train.columns)
        >>> print(f"Original: {len(X_train.columns)} → Transformed: {len(feature_names)}")
    """
    try:
        # Try to get feature names from preprocessing step
        if 'preprocessing' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessing']
            feature_names = preprocessor.get_feature_names_out(input_feature_names)
            return list(feature_names)
        else:
            # No preprocessing, return original names
            return list(input_feature_names)
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        return list(input_feature_names)


# Pipeline configuration presets
PIPELINE_CONFIGS = {
    'baseline': {
        'apply_feature_engineering': False,
        'apply_preprocessing': True,
        'description': 'Baseline: preprocessing only, no custom features'
    },
    'features_only': {
        'apply_feature_engineering': True,
        'apply_preprocessing': False,
        'description': 'Feature engineering only, no preprocessing'
    },
    'full': {
        'apply_feature_engineering': True,
        'apply_preprocessing': True,
        'description': 'Full pipeline: preprocessing + features'
    },
    'minimal': {
        'apply_feature_engineering': False,
        'apply_preprocessing': False,
        'description': 'Minimal: model only, no transformations'
    }
}


def create_pipeline_from_config(model, config_name='full', **kwargs):
    """
    Creates pipeline using predefined configuration presets.

    Available configs:
    - 'baseline': Preprocessing only, no custom features
    - 'features_only': Custom features only, no preprocessing
    - 'full': Both preprocessing and features (recommended)
    - 'minimal': Model only, no transformations

    Args:
        model: Sklearn estimator
        config_name (str): Config preset name
        **kwargs: Additional arguments for create_full_pipeline

    Returns:
        Pipeline: Configured pipeline

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> model = Ridge(alpha=1.0)
        >>> pipeline = create_pipeline_from_config(model, 'full')
    """
    if config_name not in PIPELINE_CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Available: {list(PIPELINE_CONFIGS.keys())}")

    config = PIPELINE_CONFIGS[config_name].copy()
    description = config.pop('description')

    # Override with any provided kwargs
    config.update(kwargs)

    logger.info(f"Creating pipeline with config '{config_name}': {description}")

    return create_full_pipeline(model, **config)


if __name__ == "__main__":
    """Test pipeline creation"""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    print("Testing Pipeline Factory Functions...\n")

    # Create sample data
    test_data = pd.DataFrame({
        'Age': [25, 35, 50, 65],
        'Body mass index': [22, 28, 32, 19],
        'Distance from Residence to Work': [5, 15, 30, 45],
        'Work load Average/day ': [200, 250, 300, 350],
        'Service time': [2, 5, 10, 15],
        'Hit target': [95, 90, 85, 80],
        'Transportation expense': [100, 200, 150, 120],
        'Son': [1, 2, 3, 4],
        'Pet': [0, 1, 2, 0],
        'Education': [1, 2, 3, 4],
        'Day of the week': [2, 3, 4, 5],
        'Month of absence': [1, 6, 9, 12],
        'Reason for absence': [1, 10, 13, 23],
        'Seasons': [1, 2, 3, 4],
        'Disciplinary failure': [0, 0, 1, 0],
        'Absenteeism time in hours': [2, 6, 12, 30]
    })

    X = test_data.drop('Absenteeism time in hours', axis=1)
    y = test_data['Absenteeism time in hours']

    print(f"Test data shape: {X.shape}")
    print(f"Target shape: {y.shape}\n")

    # Test 1: Preprocessing pipeline
    print("1. Testing preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(X)
    X_preprocessed = preprocessor.transform(X)
    print(f"   OK Preprocessed shape: {X_preprocessed.shape}")

    # Test 2: Feature engineering pipeline
    print("\n2. Testing feature engineering pipeline...")
    feature_pipe = create_feature_pipeline()
    feature_pipe.fit(X)
    X_features = feature_pipe.transform(X)
    print(f"   OK With features shape: {X_features.shape}")
    print(f"   OK New columns: {set(X_features.columns) - set(X.columns)}")

    # Test 3: Full pipeline creation (structure test only)
    print("\n3. Testing full pipeline structure...")
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    full_pipe = create_full_pipeline(model, apply_feature_engineering=True, apply_preprocessing=False)
    print(f"   OK Pipeline created with {len(full_pipe.steps)} steps: {' -> '.join([s[0] for s in full_pipe.steps])}")
    print(f"   (Full integration test with real data will be done in baseline_experiments.py)")

    # Test 4: Config-based pipeline creation (just check structure, don't fit)
    print("\n4. Testing config-based pipeline creation...")
    for config_name in PIPELINE_CONFIGS.keys():
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        pipeline = create_pipeline_from_config(model, config_name)
        print(f"   OK '{config_name}': {len(pipeline.steps)} steps - {' -> '.join([s[0] for s in pipeline.steps])}")

    print("\n[PASS] All pipeline tests passed!")
    print("Note: Full integration test with real data will be done in baseline_experiments.py")
