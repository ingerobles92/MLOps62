"""
Model Pipeline Module for Absenteeism Prediction
Phase 2: Sklearn-compatible pipeline architecture

This module contains pipeline factory functions for creating
end-to-end ML pipelines with preprocessing, feature engineering, and models.
"""

from src.models.pipelines import (
    create_preprocessing_pipeline,
    create_feature_pipeline,
    create_full_pipeline,
    create_pipeline_from_config,
    get_feature_names_from_pipeline,
    PIPELINE_CONFIGS
)

__all__ = [
    'create_preprocessing_pipeline',
    'create_feature_pipeline',
    'create_full_pipeline',
    'create_pipeline_from_config',
    'get_feature_names_from_pipeline',
    'PIPELINE_CONFIGS'
]
