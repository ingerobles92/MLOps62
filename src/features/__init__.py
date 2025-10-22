"""
Feature Engineering Module for Absenteeism Prediction
Phase 2: Sklearn-compatible transformers

This module contains sklearn-compatible transformers for feature engineering.
"""

from src.features.transformers import (
    AbsenceCategoryTransformer,
    BMICategoryTransformer,
    AgeGroupTransformer,
    DistanceCategoryTransformer,
    WorkloadCategoryTransformer,
    SeasonNameTransformer,
    HighRiskTransformer
)

__all__ = [
    'AbsenceCategoryTransformer',
    'BMICategoryTransformer',
    'AgeGroupTransformer',
    'DistanceCategoryTransformer',
    'WorkloadCategoryTransformer',
    'SeasonNameTransformer',
    'HighRiskTransformer'
]
