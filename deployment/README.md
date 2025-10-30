# Model Deployment - Absenteeism Prediction API

**Author:** Alexis Alduncin (Data Scientist)
**Team:** MLOps 62
**Model:** SVR with RBF kernel (MAE: 3.83 hours)

---

## Overview

This directory contains the Flask REST API for deploying the absenteeism prediction model in production. The API provides endpoints for single and batch predictions with proper error handling and monitoring.

### Architecture
- **Framework:** Flask 2.3.3
- **Production Server:** Gunicorn with 4 workers
- **Model:** Best performing model from Phase 2 (SVR with RBF kernel)
- **Features:** 19 input features from Phase 1 cleaned data
- **Performance:** MAE = 3.83 hours on test set

---

## Quick Start

### 1. Export the Best Model

First, ensure the model is exported from MLflow experiments:

```bash
cd ..
python scripts/export_best_model.py
```

This will create:
- `models/best_model_svr.pkl` - Trained model pipeline
- `models/model_metadata.json` - Model performance metrics
- `models/feature_names.txt` - Feature list

### 2. Install Dependencies

```bash
cd deployment
pip install -r requirements.txt
```

### 3. Run Locally (Development)

```bash
python app.py
```

The API will start on `http://localhost:5000`

### 4. Test the API

In a new terminal:

```bash
python test_api.py
```

This will run a comprehensive test suite checking all endpoints.

---

## API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-10-30T10:30:00",
  "model_loaded": true,
  "model_type": "SVR",
  "test_mae": 3.83,
  "training_date": "2024-10-20 15:17:07"
}
```

### 2. Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "Age": 35,
  "Distance from Residence to Work": 15,
  "Service time": 10,
  "Body mass index": 25.5,
  "Work load Average/day": 250,
  "Hit target": 95,
  "Transportation expense": 200,
  "Disciplinary failure": 0,
  "Education": 1,
  "Son": 1,
  "Social drinker": 1,
  "Social smoker": 0,
  "Pet": 1,
  "Weight": 75,
  "Height": 175,
  "Reason for absence": 0,
  "Month of absence": 7,
  "Day of the week": 3,
  "Seasons": 1
}
```

**Response:**
```json
{
  "prediction": 4.2,
  "unit": "hours",
  "model": "SVR_RBF",
  "timestamp": "2024-10-30T10:30:00",
  "confidence": "MAE ± 3.83"
}
```

### 3. Batch Prediction
```bash
POST /batch_predict
Content-Type: application/json

{
  "instances": [
    {"Age": 35, "Distance from Residence to Work": 15, ...},
    {"Age": 42, "Distance from Residence to Work": 25, ...},
    ...
  ]
}
```

**Response:**
```json
{
  "predictions": [4.2, 6.1, 3.5],
  "count": 3,
  "model": "SVR_RBF",
  "timestamp": "2024-10-30T10:30:00",
  "confidence": "MAE ± 3.83"
}
```

### 4. Model Information
```bash
GET /model_info
```

**Response:**
```json
{
  "status": "loaded",
  "metadata": {
    "model_type": "SVR",
    "test_mae": 3.83,
    "test_rmse": 10.105,
    "test_r2": 0.063,
    "train_date": "2024-10-20 15:17:07",
    "features_count": 19,
    "training_samples": 592,
    "test_samples": 148
  },
  "timestamp": "2024-10-30T10:30:00"
}
```

---

## Docker Deployment

### Build Docker Image
```bash
docker build -t absenteeism-api:latest .
```

### Run Container
```bash
docker run -p 5000:5000 absenteeism-api:latest
```

### Test Dockerized API
```bash
curl http://localhost:5000/health
```

---

## Production Deployment (TODO for Team)

### 1. Cloud Deployment Options

#### AWS (ECS/Fargate)
```bash
# Build and push to ECR
aws ecr create-repository --repository-name absenteeism-api
docker tag absenteeism-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/absenteeism-api:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/absenteeism-api:latest

# Create ECS task definition and service
# Configure ALB for load balancing
# Set up auto-scaling based on CPU/memory
```

#### Azure (App Service)
```bash
# Login to Azure
az login

# Create resource group
az group create --name MLOps62 --location eastus

# Create App Service plan
az appservice plan create --name MLOpsPlan --resource-group MLOps62 --sku B1 --is-linux

# Deploy container
az webapp create --resource-group MLOps62 --plan MLOpsPlan --name absenteeism-api --deployment-container-image absenteeism-api:latest
```

#### GCP (Cloud Run)
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/absenteeism-api

# Deploy to Cloud Run
gcloud run deploy absenteeism-api --image gcr.io/<project-id>/absenteeism-api --platform managed --region us-central1 --allow-unauthenticated
```

### 2. Authentication Setup

Add API key authentication:

```python
# In app.py, add before each endpoint:
API_KEY = os.environ.get('API_KEY')

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-Key') != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ...
```

### 3. Load Balancer Configuration

Set up health check endpoint and configure load balancer:
- Health check path: `/health`
- Health check interval: 30 seconds
- Unhealthy threshold: 2 consecutive failures
- Healthy threshold: 2 consecutive successes

### 4. Monitoring and Logging

Configure CloudWatch (AWS), Application Insights (Azure), or Stackdriver (GCP):
- API request/response times
- Prediction latency
- Error rates
- Model performance metrics

### 5. CI/CD Pipeline

Set up GitHub Actions or Jenkins:
```yaml
# .github/workflows/deploy.yml
name: Deploy API

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and test
        run: |
          cd deployment
          pip install -r requirements.txt
          python test_api.py
      - name: Build Docker image
        run: docker build -t absenteeism-api:latest deployment/
      - name: Deploy to cloud
        run: # Cloud-specific deployment commands
```

---

## Error Handling

The API handles the following error cases:

1. **Model not loaded** (503 Service Unavailable)
2. **Missing required features** (400 Bad Request)
3. **Invalid JSON format** (400 Bad Request)
4. **Invalid endpoint** (404 Not Found)
5. **Internal errors** (500 Internal Server Error)

All errors return JSON with:
```json
{
  "error": "Error message",
  "type": "ErrorType"
}
```

---

## Performance Optimization

### Current Performance
- Single prediction: ~50ms
- Batch prediction (10 instances): ~100ms
- Model load time: <1 second

### Optimization Strategies (if needed)
1. **Caching:** Cache predictions for identical inputs (Redis)
2. **Batch processing:** Optimize batch size for throughput
3. **Model quantization:** Reduce model size if memory-constrained
4. **Async processing:** Use Celery for long-running predictions

---

## Security Considerations

1. **Input Validation:** All inputs validated before prediction
2. **Rate Limiting:** TODO - Add Flask-Limiter
3. **HTTPS:** Use TLS/SSL in production
4. **API Keys:** Implement authentication (see Production Deployment section)
5. **CORS:** Configure allowed origins for web clients

---

## Troubleshooting

### Model Not Found Error
```
FileNotFoundError: Model file not found: ../models/best_model_svr.pkl
```

**Solution:** Run the model export script first:
```bash
python ../scripts/export_best_model.py
```

### Import Errors
```
ModuleNotFoundError: No module named 'src'
```

**Solution:** Make sure you're running from the deployment directory and the parent src/ folder exists.

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```

**Solution:** Kill the process using port 5000 or use a different port:
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9

# Or use different port
python app.py --port 5001
```

---

## Next Steps

1. **Deploy to cloud platform** (AWS/Azure/GCP)
2. **Set up authentication** (API keys or OAuth)
3. **Configure monitoring** (CloudWatch/Application Insights)
4. **Implement request logging** for audit trail
5. **Add rate limiting** to prevent abuse
6. **Set up automated retraining** triggers
7. **Create A/B testing** framework for model updates

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Flask application with API endpoints |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build configuration |
| `test_api.py` | Comprehensive API test suite |
| `README.md` | This file |

---

## Contact

**Data Scientist:** Alexis Alduncin
**Team:** MLOps 62
**Repository:** https://github.com/ingerobles92/MLOps62

For issues or questions, please open a GitHub issue or contact the team.
