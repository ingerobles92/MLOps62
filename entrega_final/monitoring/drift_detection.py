"""
Data Drift Detection and Simulation
Team 62 - Final Delivery

Demonstrates:
- Data drift simulation (Age, Distance, Workload)
- Drift detection using Evidently
- Performance degradation monitoring
- Automated alert system

Run with: python monitoring/drift_detection.py
"""

import sys
import os
sys.path.append(os.path.abspath('../..'))

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def simulate_data_drift(original_data, drift_factor=0.2):
    """
    Simulate realistic data drift scenarios

    Args:
        original_data: Reference DataFrame
        drift_factor: Magnitude of drift (0-1)

    Returns:
        DataFrame with simulated drift
    """
    print(f"\nSimulating data drift (factor: {drift_factor})...")

    drifted_data = original_data.copy()

    # Scenario 1: Population aging (Age drift)
    print("  - Applying age drift (population getting older)")
    age_shift = np.random.normal(5, 2, len(drifted_data))
    drifted_data['Age'] = drifted_data['Age'] + age_shift
    drifted_data['Age'] = drifted_data['Age'].clip(18, 65)  # Keep reasonable bounds

    # Scenario 2: Remote work increase (Distance drift)
    print("  - Applying distance drift (more remote work)")
    drifted_data['Distance from Residence to Work'] *= (1 + drift_factor)

    # Scenario 3: Workload intensification
    print("  - Applying workload drift (increased workload)")
    drifted_data['Work load Average/day'] *= 1.3

    # Scenario 4: Service time changes
    print("  - Applying service time variability")
    service_noise = np.random.normal(0, 2, len(drifted_data))
    drifted_data['Service time'] = drifted_data['Service time'] + service_noise
    drifted_data['Service time'] = drifted_data['Service time'].clip(0, 30)

    # Scenario 5: BMI distribution change
    print("  - Applying BMI distribution shift")
    drifted_data['Body mass index'] *= (1 + drift_factor * 0.5)

    # Scenario 6: Introduce missing values (data quality degradation)
    print("  - Introducing missing values (5% of Service time)")
    missing_indices = np.random.choice(
        len(drifted_data),
        size=int(0.05 * len(drifted_data)),
        replace=False
    )
    drifted_data.loc[missing_indices, 'Service time'] = np.nan

    # Scenario 7: Transportation expense increase (inflation)
    print("  - Applying transportation cost inflation")
    drifted_data['Transportation expense'] *= 1.15

    print(f"✅ Drift simulation complete ({len(drifted_data)} records)")
    return drifted_data


def detect_drift(reference_data, current_data, target_col='Absenteeism time in hours'):
    """
    Detect data drift using Evidently framework

    Args:
        reference_data: Original/reference DataFrame
        current_data: Current/drifted DataFrame
        target_col: Target column name

    Returns:
        Dictionary with drift detection results and alerts
    """
    print("\n" + "="*60)
    print("DRIFT DETECTION ANALYSIS")
    print("="*60)

    # Define column mapping
    numerical_features = [
        'Age', 'Distance from Residence to Work',
        'Service time', 'Work load Average/day',
        'Transportation expense', 'Height', 'Weight',
        'Body mass index', 'Son', 'Pet', 'Hit target'
    ]

    categorical_features = [
        'Disciplinary failure', 'Education',
        'Social drinker', 'Social smoker'
    ]

    column_mapping = ColumnMapping(
        target=target_col,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    # Create drift report
    print("\nGenerating Evidently drift report...")
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )

    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'drift_report_{timestamp}.html'
    drift_report.save_html(report_path)
    print(f"✅ HTML report saved: {report_path}")

    # Extract metrics
    report_dict = drift_report.as_dict()

    # Parse drift results
    dataset_drift = report_dict['metrics'][2]['result']['dataset_drift']
    n_drifted = report_dict['metrics'][2]['result']['number_of_drifted_columns']
    drift_share = report_dict['metrics'][2]['result']['share_of_drifted_columns']

    # Get drifted columns
    drifted_columns = []
    drift_by_columns = report_dict['metrics'][2]['result']['drift_by_columns']
    for col, info in drift_by_columns.items():
        if info['drift_detected']:
            drifted_columns.append({
                'column': col,
                'drift_score': info['drift_score'],
                'threshold': info.get('threshold', 0.1)
            })

    results = {
        'timestamp': timestamp,
        'dataset_drift_detected': dataset_drift,
        'report_path': report_path,
        'n_drifted_features': n_drifted,
        'drift_share': drift_share,
        'drifted_columns': drifted_columns
    }

    # Performance degradation check
    print("\nEvaluating model performance on drifted data...")
    try:
        from sklearn.metrics import mean_absolute_error
        import pickle

        # Load trained model
        model_path = '../../models/best_model_svr.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Evaluate on current data (with drift)
            X_current = current_data.drop(target_col, axis=1)
            y_current = current_data[target_col]

            # Handle missing values for prediction
            X_current_clean = X_current.fillna(X_current.mean())

            predictions = model.predict(X_current_clean)
            mae_current = mean_absolute_error(y_current, predictions)

            # Compare with baseline
            mae_baseline = 3.83
            performance_degradation = ((mae_current - mae_baseline) / mae_baseline) * 100

            results['mae_baseline'] = mae_baseline
            results['mae_current'] = round(mae_current, 2)
            results['performance_degradation_pct'] = round(performance_degradation, 2)

            print(f"✅ Performance evaluation complete")
            print(f"   Baseline MAE: {mae_baseline:.2f}h")
            print(f"   Current MAE: {mae_current:.2f}h")
            print(f"   Degradation: {performance_degradation:.1f}%")
        else:
            print(f"⚠️  Model not found at {model_path}, skipping performance check")
            results['mae_baseline'] = 3.83
            results['mae_current'] = None
            results['performance_degradation_pct'] = None

    except Exception as e:
        print(f"⚠️  Error during performance evaluation: {e}")
        results['mae_baseline'] = 3.83
        results['mae_current'] = None
        results['performance_degradation_pct'] = None

    # Determine alert level
    print("\nDetermining alert level...")

    if dataset_drift and results.get('performance_degradation_pct'):
        if results['performance_degradation_pct'] > 15:
            results['alert_level'] = 'CRITICAL'
            results['action'] = 'Immediate model retraining required'
            results['priority'] = 1
        elif results['performance_degradation_pct'] > 10:
            results['alert_level'] = 'HIGH'
            results['action'] = 'Model retraining recommended within 1 week'
            results['priority'] = 2
        else:
            results['alert_level'] = 'MEDIUM'
            results['action'] = 'Schedule model retraining within 2 weeks'
            results['priority'] = 3
    elif dataset_drift:
        results['alert_level'] = 'MEDIUM'
        results['action'] = 'Monitor closely, consider retraining'
        results['priority'] = 3
    else:
        results['alert_level'] = 'LOW'
        results['action'] = 'Continue normal monitoring'
        results['priority'] = 4

    return results


def print_results(results):
    """Print formatted drift detection results"""
    print("\n" + "="*60)
    print("DRIFT DETECTION RESULTS")
    print("="*60)

    print(f"\n📊 Dataset Drift: {'✅ DETECTED' if results['dataset_drift_detected'] else '❌ NOT DETECTED'}")
    print(f"📈 Drifted Features: {results['n_drifted_features']}")
    print(f"📉 Drift Share: {results['drift_share']:.1%}")

    if results['drifted_columns']:
        print(f"\n🔍 Drifted Columns (Top 5):")
        for col_info in sorted(results['drifted_columns'],
                              key=lambda x: x['drift_score'],
                              reverse=True)[:5]:
            print(f"   • {col_info['column']}: {col_info['drift_score']:.3f}")

    if results.get('mae_current'):
        print(f"\n📉 Performance Metrics:")
        print(f"   Baseline MAE: {results['mae_baseline']:.2f}h")
        print(f"   Current MAE:  {results['mae_current']:.2f}h")
        print(f"   Degradation:  {results['performance_degradation_pct']:.1f}%")

    print(f"\n🚨 Alert Level: {results['alert_level']}")
    print(f"💡 Recommended Action: {results['action']}")
    print(f"📄 Report: {results['report_path']}")

    print("\n" + "="*60)


def main():
    """Main execution function"""
    from src.data_utils import load_data

    print("\n" + "="*60)
    print("DATA DRIFT DETECTION DEMONSTRATION")
    print("Team 62 - MLOps Final Delivery")
    print("="*60)

    # Load reference data
    print("\nStep 1: Loading reference data...")
    try:
        reference_data = load_data('data/processed/absenteeism_cleaned.csv')
        print(f"✅ Reference data loaded: {reference_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Make sure you're running from the project root")
        return

    # Simulate drift
    print("\nStep 2: Simulating data drift...")
    drifted_data = simulate_data_drift(reference_data, drift_factor=0.3)

    # Detect drift
    print("\nStep 3: Running drift detection...")
    results = detect_drift(reference_data, drifted_data)

    # Display results
    print_results(results)

    # Save results to JSON
    results_file = 'drift_results.json'
    # Convert to JSON-serializable format
    json_results = {k: v for k, v in results.items()
                   if k not in ['drifted_columns']}  # Simplified for JSON
    json_results['drifted_columns_count'] = len(results['drifted_columns'])

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

    print("\n✅ Drift detection demonstration complete!")
    print("\nNext steps:")
    print("  1. Review HTML report for detailed visualizations")
    print("  2. Check drift_results.json for programmatic access")
    print("  3. Implement alerts in production monitoring")
    print("  4. Schedule retraining based on alert level")


if __name__ == "__main__":
    main()
