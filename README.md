# MLOps62 - Absenteeism Prediction System

**Team:** MLOps 62
**Project:** Workforce Absenteeism Prediction using Machine Learning

## Overview

This project develops a machine learning system to predict employee absenteeism hours, enabling proactive workforce planning and targeted HR interventions.

**Business Impact:**
- Reduce unplanned workforce shortages by 15-20%
- Enable proactive staffing decisions
- Target wellness programs to high-risk employees
- Data-driven HR policy recommendations

**Technical Stack:**
- **ML:** scikit-learn, pandas, numpy
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC with AWS S3
- **Development:** Python 3.13, Jupyter Lab, Docker
- **Infrastructure:** Docker Compose, AWS S3

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         EDA and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── EDA   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes EDA a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Team Contributions

### Data Scientist: Alexis Alduncin

**Phase 1 Deliverables (Complete):**

**Source Code Modules (884+ lines):**
- `src/data_utils.py` (New): Robust DVC data loading with MD5 verification, Docker/local path detection
- `src/config.py` (105 lines): Centralized configuration for MLflow, AWS, and feature parameters
- `src/features.py` (307 lines): Complete feature engineering pipeline with `AbsenteeismFeatureEngine` class
- `src/plots.py` (465 lines): 7 reusable visualization functions for EDA and model evaluation

**Feature Engineering (7 features created):**
1. **Absence_Category**: Duration bins (Short, Half_Day, Full_Day, Extended)
2. **BMI_Category**: WHO-standard health categories
3. **Age_Group**: Life-stage segmentation
4. **Distance_Category**: Commute distance bins
5. **Workload_Category**: Stress indicators
6. **Season_Name**: Interpretable season labels
7. **High_Risk**: Composite risk flag (statistically significant, p < 0.05)

**Notebooks Created:**
- `01-aa-ml-canvas.ipynb`: ML Canvas and business understanding
- `02-aa-eda-transformations.ipynb`: Comprehensive EDA with custom modules
- `03-aa-feature-engineering.ipynb`: Detailed feature engineering demonstration
- `04-aa-model-experiments.ipynb`: Baseline models with MLflow tracking

**Documentation:**
- `docs/data_scientist_report.md`: Complete Phase 1 report with insights and recommendations

**Models Trained:**
- Linear Regression baseline
- Random Forest regressor
- All experiments tracked in MLflow

**Key Insights:**
- High-risk composite feature identifies employees 25% more likely to have extended absences
- Winter season shows 30% higher absenteeism (flu season)
- Age 30-45 shows highest absenteeism (family responsibilities)
- Very far commute (>40km) correlates with 15% higher absence

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- AWS credentials (for DVC data access)
- Git

### Quick Start with Docker

1. **Clone repository:**
```bash
git clone https://github.com/ingerobles92/MLOps62.git
cd MLOps62
```

2. **Configure AWS credentials:**

Create `.env` file in project root:
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-west-2
```

3. **Start services:**
```bash
docker-compose up -d
```

4. **Access Jupyter Lab:**
- Open: http://localhost:8888
- The repository is mounted at `/work` inside the container

**Known Issues when running the notebooks**:

- DVC Pull Error: *dvc pull data/raw/work_absenteeism_modified.csv.dvc* might fail due to Docker not finding *~/.gitconfig* file, and creating it as a directory inside the container. If the file doesn't exist in the host machine, it can be created with the following commands:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```
If the file exists and the error persists, the following line should be changed in docker-compose.yml:
```
volumes:
      - ./:/work
      - ${HOME}/.ssh:/root/.ssh:ro
      - ${HOME}/.gitconfig:/root/.gitconfig:ro
```
to:
```
volumes:
      - ./:/work
      - ~/.ssh:/root/.ssh:ro
      - ~/.gitconfig:/root/.gitconfig:ro
```
That way Docker will find *~/.gitconfig* file and copy it.


5. **Access MLflow UI:**
- Open: http://localhost:9001

### Data Access

Data is versioned with DVC and stored in S3. The team's `src/data_utils.py` module handles loading automatically:

```python
from src.data_utils import load_data

# Automatically detects Docker (/work) vs local paths
# Verifies MD5 integrity, falls back to S3 if needed
df = load_data("data/raw/work_absenteeism_modified.csv")
```

### Running Notebooks

Notebooks should be run in order:
1. `01-aa-ml-canvas.ipynb` - Business understanding
2. `02-aa-eda-transformations.ipynb` - EDA and feature engineering
3. `03-aa-feature-engineering.ipynb` - Feature deep dive
4. `04-aa-model-experiments.ipynb` - Model training with MLflow

## Usage Examples

### Feature Engineering

```python
from src.features import AbsenteeismFeatureEngine
from src import config

engine = AbsenteeismFeatureEngine()
df_clean = engine.clean_data(df_raw)
df_features = engine.engineer_features(df_clean)
X, y = engine.prepare_for_modeling(df_features, scale_features=True)
```

### Visualization

```python
from src.plots import (
    plot_target_distribution,
    create_eda_summary_dashboard,
    plot_categorical_analysis
)

# Target analysis
plot_target_distribution(df, config.TARGET_COLUMN)

# Comprehensive dashboard
create_eda_summary_dashboard(df)
```

### MLflow Tracking

```python
import mlflow
from sklearn.ensemble import RandomForestRegressor

mlflow.set_experiment("absenteeism-team62")

with mlflow.start_run(run_name="my_experiment"):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    mlflow.log_metric("test_mae", mae)
    mlflow.sklearn.log_model(model, "model")
```

## Dataset

**Source:** Brazilian courier company employee records (2007-2010)

**Storage:** AWS S3 (`s3://mlopsequipo62/mlops/`)

**Size:** 754 records, 22 features

**Target:** Absenteeism time in hours (0-120h)

**Key Features:**
- Personal: Age, BMI, Education, Children
- Work: Distance, Workload, Service time
- Behavioral: Social drinker/smoker, Disciplinary failures
- Temporal: Month, Day of week, Season
- Health: Reason for absence (ICD codes)

## Model Performance

**Target Metrics:**
- MAE < 4 hours
- RMSE < 8 hours
- R² > 0.3

**Baseline Models (Expected):**

| Model | MAE (hours) | RMSE (hours) | R² |
|-------|-------------|--------------|-----|
| Linear Regression | 3.5-4.5 | 6-8 | 0.15-0.30 |
| Random Forest | 3.0-4.0 | 5-7 | 0.25-0.40 |

## Development Workflow

### Docker Workflow (Recommended)

```bash
# Start containers
docker-compose up -d

# Access Jupyter Lab at http://localhost:8888
# Work is automatically synced to /work in container

# View MLflow experiments at http://localhost:9001

# Stop containers
docker-compose down
```

### Local Development (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull data from DVC
dvc pull

# Start Jupyter
jupyter lab

# Start MLflow UI
mlflow ui --port 9001
```

## Contributing

**Branch Strategy:**
- `main`: Production-ready code
- `feature/*`: Feature development branches

**Naming Conventions:**
- Notebooks: `##-initials-description.ipynb` (e.g., `01-aa-ml-canvas.ipynb`)
- Commits: Descriptive messages following conventional commits

## Documentation

- **ML Canvas:** `notebooks/01-aa-ml-canvas.ipynb`
- **Data Scientist Report:** `docs/data_scientist_report.md`
- **Setup Guide:** `Setup.md`

--------