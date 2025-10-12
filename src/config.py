"""
Configuration for MLOps Team 62 - Absenteeism Prediction Project
Author: Alexis Alduncin (Data Scientist)
"""

import os

# ===== MLflow Configuration =====
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "absenteeism-team62"

# ===== AWS Configuration =====
# AWS credentials are managed via environment variables or .env file
AWS_BUCKET = "s3://mlopsequipo62/mlops/"
AWS_REGION = "us-west-2"

# ===== Data Paths =====
RAW_DATA_PATH = "data/raw/work_absenteeism_modified.csv"
PROCESSED_DATA_PATH = "data/processed/"
CLEAN_DATA_FILE = "work_absenteeism_clean_v1.0.csv"
CLEAN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, CLEAN_DATA_FILE)
INTERIM_DATA_PATH = "data/interim/"

# ===== Model Configuration =====
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# ===== Feature Engineering Parameters =====
# Outlier threshold for absenteeism hours
OUTLIER_THRESHOLD = 120  # hours

# Absence duration categories (in hours)
ABSENCE_BINS = [0, 4, 8, 24, 120]
ABSENCE_LABELS = ['Short', 'Half_Day', 'Full_Day', 'Extended']

# BMI categories (WHO standard)
BMI_BINS = [0, 18.5, 24.9, 29.9, float('inf')]
BMI_LABELS = ['Underweight', 'Normal', 'Overweight', 'Obese']

# Age group categories
AGE_BINS = [0, 30, 40, 50, 100]
AGE_LABELS = ['Young', 'Middle', 'Senior', 'Veteran']

# Distance categories (km)
DISTANCE_BINS = [0, 10, 25, 50, float('inf')]
DISTANCE_LABELS = ['Near', 'Moderate', 'Far', 'Very_Far']

# Workload categories
WORKLOAD_BINS = [0, 250, 300, float('inf')]
WORKLOAD_LABELS = ['Low', 'Medium', 'High']

# ===== Model Training Parameters =====
# Linear Regression
LR_PARAMS = {
    'fit_intercept': True,
    'normalize': False
}

# Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# ===== Visualization Configuration =====
FIGURE_SIZE = (12, 6)
COLOR_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C464E']
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ===== Target Variables =====
TARGET_COLUMN = 'Absenteeism time in hours'
TARGET_COLUMN_MONTHLY = 'MonthlyAbsenceHours'

# ===== Feature Groups =====
CATEGORICAL_FEATURES = [
    'Reason for absence',
    'Month of absence',
    'Day of the week',
    'Seasons',
    'Disciplinary failure',
    'Education',
    'Social drinker',
    'Social smoker'
]

NUMERICAL_FEATURES = [
    'Transportation expense',
    'Distance from Residence to Work',
    'Service time',
    'Age',
    'Work load Average/day',
    'Hit target',
    'Son',
    'Pet',
    'Weight',
    'Height',
    'Body mass index'
]

# Features to drop (non-predictive or target)
FEATURES_TO_DROP = ['ID', TARGET_COLUMN]
FEATURES_TO_DROP_MONTHLY = ['ID', TARGET_COLUMN_MONTHLY]

# ===== Logging Configuration =====
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
