# MLOps62 – Absenteeism Project

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Docker](https://img.shields.io/badge/docker-compose-blue)

**Team:** MLOps 62

**Project:** Workforce Absenteeism Prediction using Machine Learning

## Overview

This project develops a machine learning system to predict employee absenteeism hours, enabling proactive workforce planning and targeted HR interventions.The development of this project include data versioning, EDA, preprocessing, experiment tracking, and model training/deployment.

This repo aims to keep **data reproducibility** (DVC), **artifact traceability** (MLflow), and consistent **runtime** (Docker).


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


---


## Project Organization

```
MLOps62/
├── LICENSE                    <- Open-source license if one is chosen
├── Makefile                   <- Makefile with convenience commands like `make data` or `make train`
├── README.md                  <- The top-level README for developers using this project.
│
├── .dvc                       <- Configuration for DVC and version control.
│
├── .vscode                    <- Python configuration for the environment.
│
├── data                       <- Data from third party sources.
│   ├── external               <- Data from third party sources.
│   ├── interim                <- Intermediate data that has been transformed (DVC pointers only in Git).
│   ├── processed              <- The final, canonical data sets for modeling (DVC pointers only in Git).
│   └── raw                    <- The original, immutable data dump.
│
├── models                     <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks                  <- Jupyter notebooks. Naming e.g.`1.0-jqp-initial-data-exploration`.
│   ├── 1_EDA                  <- Jupyter notebooks of Exploratory Data Analysis (EDA).
│   ├── 2_Feature_Engineering  <- Jupyter notebooks of Feature Engineering.
│   ├── 3_Modeling             <- Jupyter notebooks of ML Modeling experiments.
│   └── Scratch                <- Jupyter notebooks of in-progress experiments.
│                
├── pyproject.toml             <- Project configuration file with package metadata for 
│                                 EDA and configuration for tools like black
│
├── references                 <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                    <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                <- Generated graphics and figures to be used in reporting
│
├── scripts                    <- Helpers to automatically run a series of commands.
│
├── SRC                        <- Source code for use in this project.
│                
├── .env.example               <- Template for env vars for AWS services.
├── .gitignore                 <- Defines which files and folders Git should ignore.
├── Dockerfile                 <- he build recipe for the container image.
├── README.md                  <- The main documentation file describing the project, structure and setup.
├── docker-compose.yml         <- The orchestration file that defines and runs multiple services.
└── requirements.txt           <- The requirements file for reproducing the analysis environment.

```

> **Important:** CSVs live in **S3 via DVC**. Git only stores DVC pointers (`.dvc` files), never raw CSVs.

---

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- AWS credentials (for DVC data access)
- Git


## Environment Variables
Create a local **`.env`** (never commit secrets). Use this template as **`.env.example`**:

```env
# ===== AWS (used by DVC and MLflow) =====
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET
AWS_DEFAULT_REGION=us-east-1

```

Commit `.env.example`, keep `.env` in `.gitignore`.

---

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
## Quickstart (Docker)
```bash
# 1) Clone & enter
git clone git@github.com:ingerobles92/MLOps62.git
cd MLOps62

# 2) Copy and fill your local env
cp .env.example .env
# edit .env and set your AWS_* keys

# 3) Start services
docker compose up -d

# 4) Attach to container shell
docker compose exec mlops-app bash

# 5) Sync datasets (pull pointers -> data)
dvc remote list               # should show s3remote
dvc status -r s3remote -c
dvc pull                      # materialize datasets per .dvc pointers
```

### Access JupyterLab
```bash
# Inside container
jupyter lab --allow-root --ip=0.0.0.0 --port 8888 --NotebookApp.token='' --NotebookApp.password=''

#Open in explorer http://localhost:8888
#Project workspace inside container: `/work`.
```
---

## Access MLflow UI
We run MLflow server via Docker Compose. Default setup:
- **Tracking URI:** `http://localhost:9001`
- **Backend store:** SQLite (mounted at `./.mlflow/`)
- **Artifact store:** S3 bucket `s3://mlopsequipo62/mlops/artifacts`

Access UI: http://localhost:9001

### Minimal tracking example MLFlow
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:9001")
mlflow.set_experiment("absenteeism-team62")

with mlflow.start_run(run_name="quick-check"):
    mlflow.log_param("model", "demo")
    mlflow.log_metric("accuracy", 0.9)
    with open("hello.txt", "w") as f:
        f.write("hello artifact")
    mlflow.log_artifact("hello.txt")
```
Artifacts are stored in S3; runs & params live in the SQLite backend.

---

## Data Versioning with DVC
**Principle:** CSVs are versioned with DVC → **Git stores only `.dvc` pointers**, blobs live in **S3**.

### Add or update a dataset (.csv)
```bash
# Inside container
dvc add data/processed/CSV_name_v1.0.csv
git add data/processed/CSV_name_v1.0.csv.dvc
git commit -m "data: track processed v1.0 via DVC"
dvc push -r s3remote      # upload blob to S3
git push origin <your-branch>
```

### Read helper (from `src/data_utils.py`)
- Verifies MD5 of local file vs pointer.
- If mismatch/missing, **auto-`dvc pull`** and read.

```python
from src.data_utils import dvc_read_csv_verified

df, source = dvc_read_csv_verified("data/raw/work_absenteeism_modified.csv")
print(source)  # "local" or "pulled"
```

---

## Git Workflow (Branches & PRs)
- **Protected `main`**; work on feature branches:
  - `feature/<roll>-<name>`
- Flow:
  1. `git switch -c feature/<something>`
  2. Commit changes (`feat:`, `chore:`, `fix:`, etc.)
  3. `git push -u origin feature/<something>`
  4. Open a Pull Request → code review → merge to `main`

Conventions:
- Notebooks “canónicos”: `notebooks/1_EDA/...`
- Intermediate/processed CSVs via DVC (`data/interim`, `data/processed`).
- No `.csv` in Git — only `.dvc` pointers.

---

## Make It Easy – Helper Script
We include an interactive helper:
```
scripts/publish_data.sh
```
It lets you:
- Select CSV paths to track with DVC.
- Push blobs to S3.
- Commit & push `.dvc` pointers to Git.

Run:
```bash
bash scripts/publish_data.sh
```

---

## Requirements
See `requirements.txt` for the stack used (PyData, DVC S3, MLflow, etc.).
Install inside Docker automatically via Compose.

---

## Security & Secrets
- Never commit `.env` or credentials.
- Limit IAM permissions to the required S3 paths (`mlops/*` is enough).
- Rotate keys when needed.

---

## Contributing
1. Create a feature branch
2. Make changes & tests pass
3. Open a PR to `main`
4. Address review comments and merge

---


---

## Phase 2 Updates - Data Scientist (October 2024)

### 🎯 Achievements
- ✅ **Implemented sklearn pipeline architecture** with modular design
- ✅ **Created 6 reusable feature transformers** (BMI, Age, Distance, Workload, Season, High-Risk)
- ✅ **Achieved MAE: 3.83 hours** (30% improvement over baseline 5.44 hours)
- ✅ **Target met:** MAE < 4.0 hours
- ✅ **Documented all experiments** with MLflow tracking

### 📁 Files Added
| File | Description | Lines |
|------|-------------|-------|
| `src/features/transformers.py` | 6 sklearn-compatible feature engineering transformers | 380 |
| `src/models/pipelines.py` | Modular pipeline factory functions (3-layer architecture) | 300 |
| `experiments/baseline_experiments.py` | Comprehensive model comparison framework (15 models) | 374 |
| `experiments/baseline_results.csv` | Complete model performance comparison | - |
| `notebooks/07-aa-phase2-pipeline-experiments.ipynb` | Results analysis and insights | - |
| `docs/phase2_report.md` | Technical implementation report | - |

### 🏆 Best Model
**Support Vector Regression (RBF kernel)** with sklearn pipeline

| Metric | Value |
|--------|-------|
| Test MAE | **3.83 hours** |
| Test R² | 0.063 |
| CV MAE | 4.61 ± 0.52 hours |
| Status | ✅ Ready for production deployment |

### 🔑 Key Technical Decisions
1. **Pipeline Order:** Features → Preprocessing → Model (prevents column name issues)
2. **Auto-Detection:** Dynamic column type detection for feature-engineered variables
3. **Data Leakage Prevention:** Stateful transformers (WorkloadCategoryTransformer) fit only on training data
4. **Outlier Handling:** Used Phase 1 cleaned data (capped at 120 hours) to resolve data quality issues

### 📊 Model Comparison (Top 5)
| Rank | Model | Test MAE | Improvement vs Baseline |
|------|-------|----------|------------------------|
| 🥇 | SVR_rbf | 3.83 | 30% |
| 🥈 | RandomForest_depth7 | 4.96 | 9% |
| 🥉 | KNN_10 | 4.97 | 9% |
| 4 | LightGBM_conservative | 5.02 | 8% |
| 5 | RandomForest_depth5 | 5.09 | 6% |

### 🔬 Experiment Tracking
All 15 model runs logged to MLflow with:
- Hyperparameters
- Train/test metrics (MAE, RMSE, R²)
- Cross-validation scores (5-fold)
- Overfitting analysis
- Feature importance (tree models)
- Model artifacts

### 📈 Next Steps (Phase 3)
- Deploy SVR model as REST API (Flask/FastAPI)
- Implement model monitoring with Evidently
- Set up automated retraining pipeline
- Hyperparameter optimization with Optuna
- Ensemble methods (combine top 3 models)

**Branch:** `data_scientist_v2`  
**Author:** Alexis Alduncin (Data Scientist)
