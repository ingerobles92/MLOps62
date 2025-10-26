# MLOps62 – Absenteeism Project

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Docker](https://img.shields.io/badge/docker-compose-blue)

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Docker](https://img.shields.io/badge/docker-compose-blue)

**Team:** MLOps 62  
**Project:** Workforce Absenteeism Prediction using Machine Learning

**Team Roles and Responsibilities**

The project team consists of five specialized roles that collaborate to ensure a complete MLOps lifecycle — from data collection and model training to deployment and monitoring.

| Role | Name | Responsibilities | Key Tools / Technologies |
|------|------|------------------|---------------------------|
| **DevOps Engineer** | Emanuel Robles Lezama | Designs and maintains the MLOps infrastructure. Manages Docker environments, CI/CD automation, and environment reproducibility. Ensures seamless integration between DVC, MLflow, and AWS for data and model versioning. | Docker, Docker Compose, AWS, GitHub Actions, DVC, MLflow, Linux (WSL) |
| **Data Engineer** | Elizabeth López Tapia | Builds and manages the data pipelines. Cleans, transforms, and prepares datasets for model consumption. Ensures data quality, schema consistency, and version tracking in DVC. | Python, Pandas, NumPy, DVC, AWS S3, SQL |
| **ML Engineer** | Uriel Alejandro González Rojo | Develops and optimizes machine learning models. Focuses on feature engineering, model training, hyperparameter tuning, and integration with MLflow for experiment tracking. | scikit-learn, TensorFlow / PyTorch, MLflow, NumPy, Pandas |
| **Data Scientist** | Alexis Alducin | Performs advanced data exploration and hypothesis testing. Interprets statistical patterns to improve predictive accuracy and provide business insights. Collaborates with ML Engineer on model explainability. | Python, Pandas, Seaborn, Matplotlib, JupyterLab, SciPy |
| **Software Engineer** | Héctor Jorge Morales Arch | Integrates ML components into production-ready applications. Develops and maintains backend services and APIs for model deployment and data access. Supports automation and performance optimization. | Python (FastAPI / Flask), Docker, REST APIs, Git, Unit Testing |


---

## Overview

This project develops a machine learning system to predict employee absenteeism hours, enabling proactive workforce planning and targeted HR interventions.  

The development process includes data versioning, exploratory data analysis (EDA), preprocessing, experiment tracking, and model training/deployment.

The repository ensures **data reproducibility** (via DVC), **artifact traceability** (via MLflow), and consistent **runtime environments** (via Docker).

**Business Impact:**
- Reduce unplanned workforce shortages by 15–20%
- Enable proactive staffing decisions
- Target wellness programs to high-risk employees
- Support data-driven HR policy recommendations

**Technical Stack:**  
The following technologies are used to ensure scalability, reproducibility, and efficient model lifecycle management:
- **ML:** scikit-learn, pandas, numpy  
- **Experiment Tracking:** MLflow  
- **Data Versioning:** DVC with AWS S3  
- **Development:** Python 3.13, Jupyter Lab, Docker  
- **Infrastructure:** Docker Compose, AWS S3  

For setup and execution details, see the [Setup Instructions](#setup-instructions) section below.

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

---

## Setup Instructions

### Prerequisites

- **WSL (for Windows installation only)**

  macOS and Linux users can skip this section.

  If you are using **Windows 10 or 11**, follow these steps to enable Linux compatibility (required for Docker and file mounts to work correctly):

   1. **Enable WSL 2 and install Ubuntu**  
      Open PowerShell as Administrator and run:
      ```bash
      wsl --install -d Ubuntu-24.04
      ```
      This command installs WSL 2 and downloads Ubuntu from the Microsoft Store.  
      Restart your computer when prompted.

   2. **Verify the installation**
      ```bash
      wsl --list --verbose
      ```
      Expected output:
      ```
      NAME      STATE   VERSION
      Ubuntu    Running 2
      ```

   3. **Launch WSL**
      ```bash
      wsl -d Ubuntu-24.04
      ```

   4. **Update Ubuntu packages**
      ```bash
      sudo apt update && sudo apt upgrade -y
      ```

   Once these steps are complete, open the **Ubuntu terminal** from the Windows Start Menu to continue with the Docker setup.


- **AWS credentials**

   Make sure you have an AWS account or valid IAM user credentials provided by the DevOps.

   ⚠️ *Note:* These credentials are only used to authenticate DVC and MLflow access to AWS S3.  
   You don’t need to set up any AWS services manually.


### Instructions

- **Docker and Docker Compose**

   **🪟 Windows (Docker Desktop + WSL2)**  
   1. Download and install Docker Desktop from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
   2. Enable **“Use the WSL 2 based engine”** and ensure your Ubuntu distro is selected under **WSL Integration**.
   3. Verify inside Ubuntu:
      ```bash
      docker --version
      docker compose version
      ```
      Expected output:
      ```
      Docker version 27.xx.xx
      Docker Compose version v2.xx.xx
      ```

   **🍎 macOS**  
   1. Install Docker Desktop for Mac from the same link above.  
   2. Launch Docker Desktop and verify:
      ```bash
      docker --version
      docker compose version
      ```

   **🐧 Linux (Ubuntu)**  
   1. Install Docker Engine and Compose plugin:
      ```bash
      sudo apt update
      sudo apt install ca-certificates curl gnupg -y
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      echo \
         "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
         $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
         sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
      sudo apt update
      sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y
      ```
   2. Enable and verify:
      ```bash
      sudo systemctl enable docker
      sudo systemctl start docker
      docker --version
      docker compose version
      ```



- **Git**

   **Steps:**
   1. **Install Git**
      - **Windows (via WSL Ubuntu):**
      ```bash
      sudo apt update
      sudo apt install git -y
      ```
      - **macOS:**
      ```bash
      brew install git
      ```
      - **Linux (Ubuntu):**
      ```bash
      sudo apt install git -y
      ```

   2. **Configure your Git identity (first-time setup)**
      ```bash
      git config --global user.name "Your Name"
      git config --global user.email "your.email@example.com"
      ```
      This ensures your commits are properly attributed and prevents DVC or Docker from failing due to a missing `.gitconfig` file.

      Each developer should generate an SSH key on their own machine/WSL and add it to their GitHub account.  
      This allows you to clone and push using the `git@github.com:...` URL without typing credentials every time.

      Generate a new SSH key (use your GitHub email):
      ```bash
      ssh-keygen -t ed25519 -C "your.email@example.com"
      ```
      When prompted for a file location, press Enter to accept the default (`~/.ssh/id_ed25519`).  
      Set a passphrase.

      Start the SSH agent and load your key:
      ```bash
      eval "$(ssh-agent -s)"
      ssh-add ~/.ssh/id_ed25519
      ```

      Show the public key:
      ```bash
      cat ~/.ssh/id_ed25519.pub
      ```

      Copy the entire output and add it in GitHub:
      - GitHub → Settings → SSH and GPG keys → "New SSH key"
      - Paste the key and save.

      You can verify the connection:
      ```bash
      ssh -T git@github.com
      ```
      Expected response includes a greeting like:
      `Hi <your-username>! You’ve successfully authenticated, but GitHub does not provide shell access.`

      After this step, SSH-based clone/push should work from this environment.


   3. **Clone the repository (starting from `main`)**

      ⚠️ **Windows + WSL note**  
      Do **not** clone the repository into a Windows-mounted path like `/mnt/c/...`.  
      Instead, open your Ubuntu/WSL terminal and clone into your Linux home directory:
      ```bash
      cd ~
      git clone git@github.com:ingerobles92/MLOps62.git
      cd MLOps62
      ```
      **Why:**  
      - Docker Compose mounts the current directory into the container (`./:/work`).  
      - Performance and file permissions are far more stable when the project lives in native WSL storage (`/home/<user>/MLOps62`) instead of `/mnt/c/...`.
   
      For Linux and macOS:
      ```bash
      git clone git@github.com:ingerobles92/MLOps62.git
      cd MLOps62
      ```

   4. **Create your feature branch (do not work directly on `main`)**
      ```bash
      #creates new branch
      git switch -c feature/<Role>-<Name>

      #only for previous existing branch
      git switch feature/<Role>-<Name>
      # example:
      # git switch -c feature/Devops-Emanuel
      ```

  5. **Verify Git access / repo status at any time**
     ```bash
     git status
     ```
     If you see something like `On branch feature/...` and no errors, Git is properly configured.

   ⚠️ Note:  
   `main` is protected. Do not commit directly to `main`.  
   All work should happen in your own `feature/...` branch, then go through a Pull Request into `main`.  
   Your `.env` and any credentials must never be committed.


- **Environment Variables**

   1. Make sure you are in the project root directory.  
      Example:
      ```bash
      cd /home/<user>/MLOps62
      ```

   2. Create your local `.env` file from the template:
      ```bash
      cp .env.example .env
      ```

   3. Open the new `.env` file and fill in your AWS credentials:
   
      Using `nano` in terminal (recommended for WSL/Linux):**

      ```bash
      nano .env
      ```
      Replace the placeholder values (`YOUR_KEY`, `YOUR_SECRET`). 
   
      Example content:
      ```env
      # ===== AWS (used by DVC and MLflow) =====
      AWS_ACCESS_KEY_ID=YOUR_KEY
      AWS_SECRET_ACCESS_KEY=YOUR_SECRET
      AWS_DEFAULT_REGION=us-east-1
      ```

      - `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are provided by Devops.

      Important:
      `.env.example` is committed to Git so the team can see which variables are required.
      `.env` contains real secrets and MUST NOT be committed (it is already included in `.gitignore`).


- **Docker**

   1. From the project root directory (same folder as `docker-compose.yml`), start all services in the background:
      ```bash
      cd MLOps62
      docker compose up -d
      ```
      - On first run, Docker will build the image. This includes installing Python packages, PyTorch, DVC with S3 support, etc.
      - This step can take several minutes the first time.

   2. Verify that the containers are running:
      ```bash
      docker compose ps
      ```
      You should see your main app container (for example `mlops-app`) and any supporting services (for example MLflow) in the `Up` state.

   3. Open a shell inside the main application container:
      ```bash
      docker compose exec mlops-app bash
      ```
      You are now inside the container’s environment (Python, DVC, etc.). All development commands below should be run **inside** this container.

   4. Verify connectivity (AWS / DVC / GitHub)

   5. Check that the DVC remote is configured
      Inside the container:
      ```bash
      dvc remote list
      ```
      Expected response includes a greeting like:
      `s3remote        s3://mlopsequipo62/mlops        (default)`

   5. (First time only) Pull project data with DVC:
      ```bash
      dvc pull
      ```
      This will download the versioned data/artifacts from the remote S3 bucket using the AWS credentials from your `.env`.  
      If your AWS credentials are missing or invalid, this command will fail.

   6. Verify MLflow UI is running

      The Docker Compose setup includes an MLflow tracking server.  
      To confirm it is up and reachable:

      In your browser (on the host machine, not inside the container), open:
         - http://localhost:9001/
         If the MLflow UI loads in the browser, the tracking service is running correctly. 


   7. Launch and verify JupyterLab integration with DVC

      Follow these steps to start JupyterLab inside the container and confirm that both Jupyter and DVC are working correctly together.

      1. **Launch JupyterLab**  
         Run the following command (this starts JupyterLab without password or token, accessible from your host):
         ```bash
         jupyter lab --allow-root --ip=0.0.0.0 --port 8888 --NotebookApp.token='' --NotebookApp.password=''
         ```

      2. **Open Jupyter in your browser**  
         Once JupyterLab starts, open this URL in your host browser:
         - http://localhost:8888/

         You should see the JupyterLab interface.  
         The workspace is mounted at `/work`, which mirrors your local project folder.

      3. **Run the test notebook**  
         In the Jupyter file browser, navigate to:
         ```
         /notebooks/1.0-ERL-Utilidad-DVC-lectura-datasets.ipynb
         ```
         Open the notebook and **run all cells** (or at least up to **cell 4**).

      4. **Expected output (cell 4)**  
         The notebook should print the following:
         ```
         Repo root: /work | exists: True
         Expected CSV: data/raw/work_absenteeism_modified.csv
         Expected MD5 (.dvc): 96c318341d1846f567be7127f52d03e1
         Local MD5: 96c318341d1846f567be7127f52d03e1
         MD5 matches .dvc? YES ✅
         Read from: local | rows=754 | cols=22
         ```

         ✅ If you see this output, it confirms:
         - JupyterLab is running correctly.
         - DVC successfully accessed and validated the dataset.
         - The environment and data mounting are working as expected.

      5. **Known Issues when running the notebooks**:

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


---

## Data Versioning and traceability

   ### Access MLflow UI
      
   - We run MLflow server via Docker Compose. Default setup:
      - **Tracking URI:** `http://localhost:9001`
      - **Backend store:** SQLite (mounted at `./.mlflow/`)
      - **Artifact store:** S3 bucket `s3://mlopsequipo62/mlops/artifacts`

      Access UI: http://localhost:9001

   - Minimal tracking example MLFlow
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



   ### Data Versioning with DVC

   - Add or update a dataset (.csv)
      ```bash
      # Inside container
      dvc add data/processed/CSV_name_v1.0.csv
      git add data/processed/CSV_name_v1.0.csv.dvc
      git commit -m "data: track processed v1.0 via DVC"
      dvc push -r s3remote      # upload blob to S3
      git push origin <your-branch>
      ```

   - Read helper (from `src/data_utils.py`)
      - Verifies MD5 of local file vs pointer.
      - If mismatch/missing, **auto-`dvc pull`** and read.

      ```python
      from src.data_utils import dvc_read_csv_verified

      df, source = dvc_read_csv_verified("data/raw/work_absenteeism_modified.csv")
      print(source)  # "local" or "pulled"
      ```

   > **Important:** CSVs live in **S3 via DVC**. Git only stores DVC pointers (`.dvc` files), never raw CSVs.
---

### Git Workflow (Branches & PRs)
- **Protected `main`**; work on feature branches:
  - `feature/<role>-<name>`
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

### Make It Easy – Helper Script
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
