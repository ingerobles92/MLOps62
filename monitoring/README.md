# Model Monitoring with Evidently

**Author:** Alexis Alduncin (Data Scientist)
**Team:** MLOps 62

---

## Overview

This directory contains monitoring infrastructure for the absenteeism prediction model using Evidently. Monitoring tracks data drift, data quality, and model performance in production.

### What is Monitored

1. **Data Drift** - Changes in feature distributions over time
2. **Data Quality** - Missing values, duplicates, outliers
3. **Model Performance** - MAE, RMSE, R² on new data
4. **Prediction Drift** - Changes in prediction distributions

---

## Quick Start

### 1. Install Dependencies

```bash
cd monitoring
pip install -r requirements.txt
```

### 2. Run Example Monitoring

```bash
python monitor.py
```

This will generate:
- `monitoring_report_<timestamp>.html` - Interactive HTML report
- `monitoring_summary_<timestamp>.json` - Metrics summary
- `test_results_<timestamp>.html` - Test suite results
- `test_summary_<timestamp>.json` - Test results summary
- `monitoring_config.json` - Configuration file

### 3. View Reports

Open the generated HTML files in your browser to see interactive dashboards.

---

## Usage in Production

### Basic Monitoring

```python
from monitor import AbsenteeismMonitor
import pandas as pd

# Initialize monitor with reference (training) data
monitor = AbsenteeismMonitor(
    reference_data_path='../data/processed/absenteeism_cleaned.csv'
)

# Load current production data
current_data = pd.read_csv('production_data.csv')

# Generate monitoring report
report_summary = monitor.create_monitoring_report(
    current_data=current_data,
    output_path='monitoring_report.html'
)

# Check if drift detected
if report_summary['drift_detected']:
    print(f"⚠️  Data drift detected: {report_summary['drift_share']:.1%} of features")
    # Trigger retraining or investigation
```

### Automated Testing (CI/CD)

```python
# Run test suite for automated checks
test_results = monitor.create_test_suite(
    current_data=current_data,
    output_path='test_results.html'
)

# Fail CI/CD if tests don't pass
if test_results['failed_tests'] > 0:
    raise ValueError(f"Monitoring tests failed: {test_results['failed_tests']} failures")
```

### Data Quality Checks

```python
# Quick quality check
quality = monitor.check_data_quality(current_data)

print(f"Missing values: {quality['missing_percentage']:.2f}%")
print(f"Duplicate rows: {quality['duplicate_rows']}")
print(f"Out of range values: {quality['out_of_range_values']}")

# Alert if quality issues
if quality['missing_percentage'] > 5.0:
    send_alert("High percentage of missing values detected")
```

---

## Monitoring Reports

### 1. Data Drift Report

Detects changes in feature distributions between reference and current data.

**Key Metrics:**
- **Drift Score:** Statistical measure of distribution change
- **Drift detected:** Boolean flag per feature
- **Drift share:** Percentage of features with drift

**Drift Detection Methods:**
- Kolmogorov-Smirnov test (numeric features)
- Chi-squared test (categorical features)

**Visualization:**
- Feature distribution comparisons
- Drift score heatmap
- Feature importance vs drift

### 2. Data Quality Report

Monitors data quality issues that could impact model performance.

**Checks:**
- Missing values count and percentage
- Duplicate rows
- Constant columns (no variance)
- Out-of-range values (Age, BMI, etc.)
- Type mismatches

**Visualization:**
- Missing values heatmap
- Quality metrics dashboard
- Feature completeness over time

### 3. Model Performance Report

Tracks model prediction quality on new data (requires ground truth labels).

**Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Error distribution
- Residual plots

**Visualization:**
- Actual vs predicted scatter plots
- Error distribution histograms
- Performance over time trends

---

## Alert Configuration

### monitoring_config.json

```json
{
  "alerts": {
    "drift_threshold": 0.1,
    "missing_data_threshold": 0.05,
    "mae_threshold": 4.5,
    "rmse_threshold": 12.0
  },
  "check_interval": "daily",
  "metrics": [
    "data_drift",
    "data_quality",
    "prediction_drift",
    "model_performance"
  ],
  "notification": {
    "enabled": true,
    "channels": ["email", "slack"],
    "email": "team@mlops62.com",
    "slack_webhook": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  },
  "storage": {
    "type": "s3",
    "bucket": "mlopsequipo62",
    "prefix": "monitoring_reports/"
  }
}
```

### Alert Triggers

1. **Data Drift Alert:** >10% of features showing drift
2. **Quality Alert:** >5% missing values or >10 duplicates
3. **Performance Alert:** MAE > 4.5 hours or RMSE > 12 hours
4. **Outlier Alert:** Out-of-range values detected

---

## Integration with Production API

### Add Monitoring to API

```python
# In deployment/app.py

from monitoring.monitor import AbsenteeismMonitor

# Initialize monitor
monitor = AbsenteeismMonitor(
    reference_data_path='../data/processed/absenteeism_cleaned.csv'
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)[0]

    # Log prediction for monitoring
    log_prediction(df, prediction)

    return jsonify({'prediction': float(prediction)})

def log_prediction(features, prediction):
    """Log predictions for monitoring"""
    # Append to production data file
    features['prediction'] = prediction
    features['timestamp'] = datetime.now()
    features.to_csv('production_predictions.csv', mode='a', header=False)
```

### Scheduled Monitoring Job

```python
# monitoring/scheduled_check.py

import schedule
import time
from monitor import AbsenteeismMonitor
import pandas as pd

def daily_monitoring_check():
    """Run daily monitoring checks"""
    print(f"Running daily monitoring check: {datetime.now()}")

    monitor = AbsenteeismMonitor(
        reference_data_path='../data/processed/absenteeism_cleaned.csv'
    )

    # Load last 24 hours of production data
    current_data = pd.read_csv('production_predictions.csv')
    current_data = current_data[current_data['timestamp'] > (datetime.now() - timedelta(days=1))]

    # Generate report
    report_summary = monitor.create_monitoring_report(current_data)

    # Check for alerts
    if report_summary['drift_detected']:
        send_alert(f"Data drift detected: {report_summary['drift_share']:.1%}")

    # Run tests
    test_results = monitor.create_test_suite(current_data)

    if test_results['failed_tests'] > 0:
        send_alert(f"Monitoring tests failed: {test_results['failed_tests']} failures")

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(daily_monitoring_check)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Test Suite

The test suite includes:

### Data Drift Tests
- `TestNumberOfDriftedColumns`: <5 columns with drift
- `TestShareOfDriftedColumns`: <30% columns with drift

### Data Quality Tests
- `TestNumberOfMissingValues`: <10 missing values
- `TestShareOfMissingValues`: <5% missing
- `TestNumberOfDuplicatedRows`: 0 duplicates
- `TestNumberOfConstantColumns`: 0 constant columns

### Custom Tests (add as needed)
```python
from evidently.tests import TestColumnValueMean, TestColumnValueMin

tests = TestSuite(tests=[
    TestColumnValueMean(column_name='Age', gte=25, lte=45),
    TestColumnValueMin(column_name='Age', gte=18),
    # Add more custom tests
])
```

---

## Dashboard Setup (TODO for Team)

### Option 1: Evidently UI

Start Evidently monitoring UI:

```bash
evidently ui --workspace ./workspace
```

Access at: http://localhost:8000

### Option 2: Grafana Integration

1. Export metrics to Prometheus format
2. Configure Grafana dashboard
3. Set up alerts in Grafana

### Option 3: Custom Dashboard

Create custom Flask dashboard to display reports:

```python
# monitoring/dashboard.py

from flask import Flask, render_template
import glob

app = Flask(__name__)

@app.route('/')
def dashboard():
    # Get all monitoring reports
    reports = glob.glob('monitoring_report_*.html')
    reports.sort(reverse=True)  # Most recent first

    return render_template('dashboard.html', reports=reports)

if __name__ == '__main__':
    app.run(port=8080)
```

---

## Troubleshooting

### No Reference Data

```
ValueError: Reference data not loaded. Cannot generate drift report.
```

**Solution:** Provide path to reference (training) data:
```python
monitor = AbsenteeismMonitor(
    reference_data_path='../data/processed/absenteeism_cleaned.csv'
)
```

### Column Mismatch

```
KeyError: Column 'XYZ' not found in current data
```

**Solution:** Ensure current data has the same features as reference data. Update `column_mapping` if needed.

### Memory Issues with Large Data

For large production datasets, sample before monitoring:

```python
# Sample 10,000 recent records
current_data = current_data.tail(10000)
report_summary = monitor.create_monitoring_report(current_data)
```

---

## Best Practices

1. **Run monitoring regularly** (daily or weekly)
2. **Store historical reports** for trend analysis
3. **Set appropriate thresholds** based on business needs
4. **Automate alerts** for critical issues
5. **Review reports manually** to catch subtle issues
6. **Retrain model** when significant drift detected
7. **Document investigation results** for future reference

---

## Next Steps

1. **Set up automated monitoring** (cron job or scheduler)
2. **Configure alerts** (Slack, email, PagerDuty)
3. **Create monitoring dashboard** (Grafana or custom)
4. **Define retraining triggers** (drift > 10%, MAE > 4.5)
5. **Implement A/B testing** for model updates
6. **Track business metrics** (actual absenteeism impact)

---

## Files

| File | Description |
|------|-------------|
| `monitor.py` | Main monitoring class and functions |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
| `monitoring_config.json` | Configuration (generated) |

---

## Contact

**Data Scientist:** Alexis Alduncin
**Team:** MLOps 62
**Repository:** https://github.com/ingerobles92/MLOps62

For monitoring issues or questions, please open a GitHub issue.
