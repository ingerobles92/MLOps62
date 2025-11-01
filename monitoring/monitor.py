"""
Production Model Monitoring with Evidently
Author: Alexis Alduncin (Data Scientist)
Team: MLOps 62

Monitors model performance, data drift, and data quality in production.
"""

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.tests import *
import pandas as pd
import json
import os
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

from src import config

class AbsenteeismMonitor:
    """Monitor for absenteeism prediction model"""

    def __init__(self, reference_data_path=None, target_col='Absenteeism time in hours'):
        """
        Initialize monitor

        Args:
            reference_data_path: Path to reference (training) data
            target_col: Name of target column
        """
        self.target_col = target_col
        self.reference_data = None

        if reference_data_path and os.path.exists(reference_data_path):
            self.reference_data = pd.read_csv(reference_data_path)
            print(f"✅ Reference data loaded: {self.reference_data.shape}")
        else:
            print("⚠️  No reference data provided - drift detection will be limited")

        # Define column mapping
        self.column_mapping = ColumnMapping(
            target=self.target_col,
            numerical_features=[
                'Age', 'Distance from Residence to Work',
                'Service time', 'Work load Average/day',
                'Hit target', 'Transportation expense',
                'Body mass index', 'Weight', 'Height',
                'Son', 'Pet'
            ],
            categorical_features=[
                'Disciplinary failure', 'Education',
                'Social drinker', 'Social smoker',
                'Reason for absence', 'Month of absence',
                'Day of the week', 'Seasons'
            ]
        )

    def create_monitoring_report(self, current_data, output_path='monitoring_report.html'):
        """
        Generate comprehensive monitoring report

        Args:
            current_data: Current production data (DataFrame)
            output_path: Path to save HTML report

        Returns:
            dict: Report metrics summary
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded. Cannot generate drift report.")

        print("Generating monitoring report...")

        # Create report with multiple presets
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            RegressionPreset()
        ])

        # Run report
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'monitoring_report_{timestamp}.html'

        # Save report
        report.save_html(output_path)
        print(f"✅ Report saved to: {output_path}")

        # Extract metrics
        report_dict = report.as_dict()

        # Check for drift
        drift_detected = False
        drift_share = 0.0

        for metric in report_dict.get('metrics', []):
            if 'result' in metric:
                result = metric['result']
                if isinstance(result, dict):
                    if 'drift_detected' in result:
                        drift_detected = drift_detected or result['drift_detected']
                    if 'drift_share' in result:
                        drift_share = max(drift_share, result['drift_share'])

        summary = {
            'report_path': output_path,
            'timestamp': datetime.now().isoformat(),
            'reference_size': len(self.reference_data),
            'current_size': len(current_data),
            'drift_detected': drift_detected,
            'drift_share': drift_share
        }

        # Save summary as JSON
        summary_path = f'monitoring_summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Summary saved to: {summary_path}")
        print(f"\nDrift Detection Results:")
        print(f"  Drift detected: {drift_detected}")
        print(f"  Drift share: {drift_share:.2%}")

        return summary

    def create_test_suite(self, current_data, output_path='test_results.html'):
        """
        Create test suite for CI/CD integration

        Args:
            current_data: Current production data (DataFrame)
            output_path: Path to save HTML report

        Returns:
            dict: Test results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded. Cannot run tests.")

        print("Running test suite...")

        # Create test suite with specific tests
        tests = TestSuite(tests=[
            # Data drift tests
            TestNumberOfDriftedColumns(lt=5),  # Less than 5 columns with drift
            TestShareOfDriftedColumns(lt=0.3),  # Less than 30% columns with drift

            # Data quality tests
            TestNumberOfMissingValues(lt=10),  # Less than 10 missing values
            TestShareOfMissingValues(lt=0.05),  # Less than 5% missing
            TestNumberOfDuplicatedRows(eq=0),  # No duplicates
            TestNumberOfConstantColumns(eq=0),  # No constant columns

            # Regression tests (if target is available)
            # TestValueMeanError(gte=-1, lte=1),  # Mean error within range
        ])

        # Run tests
        tests.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'test_results_{timestamp}.html'

        # Save results
        tests.save_html(output_path)
        print(f"✅ Test results saved to: {output_path}")

        # Extract test results
        test_results = tests.as_dict()

        # Count passed/failed tests
        total_tests = len(test_results.get('tests', []))
        passed_tests = sum(1 for test in test_results.get('tests', [])
                          if test.get('status') == 'SUCCESS')
        failed_tests = total_tests - passed_tests

        results_summary = {
            'test_results_path': output_path,
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }

        # Save summary
        summary_path = f'test_summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"\nTest Results:")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success rate: {results_summary['success_rate']:.1%}")

        return results_summary

    def check_data_quality(self, data):
        """
        Quick data quality check

        Args:
            data: DataFrame to check

        Returns:
            dict: Quality metrics
        """
        quality = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': int(data.isnull().sum().sum()),
            'missing_percentage': float(data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100),
            'duplicate_rows': int(data.duplicated().sum()),
            'constant_columns': int((data.nunique() == 1).sum())
        }

        # Check numeric ranges
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        out_of_range = {}

        if 'Age' in data.columns:
            out_of_range['Age'] = int(((data['Age'] < 18) | (data['Age'] > 70)).sum())

        if 'Body mass index' in data.columns:
            out_of_range['BMI'] = int(((data['Body mass index'] < 10) | (data['Body mass index'] > 50)).sum())

        quality['out_of_range_values'] = out_of_range

        return quality


def setup_monitoring_config():
    """Create monitoring configuration file"""
    config = {
        'alerts': {
            'drift_threshold': 0.1,
            'missing_data_threshold': 0.05,
            'mae_threshold': 4.5,
            'rmse_threshold': 12.0
        },
        'check_interval': 'daily',
        'metrics': [
            'data_drift',
            'data_quality',
            'prediction_drift',
            'model_performance'
        ],
        'notification': {
            'enabled': True,
            'channels': ['email', 'slack'],
            'email': 'team@mlops62.com',
            'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        },
        'storage': {
            'type': 's3',
            'bucket': 'mlopsequipo62',
            'prefix': 'monitoring_reports/'
        }
    }

    with open('monitoring_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ Monitoring configuration created: monitoring_config.json")
    return config


def example_usage():
    """Example usage of monitoring functions"""
    print("="*60)
    print("Example: Model Monitoring Usage")
    print("="*60)

    # Initialize monitor
    monitor = AbsenteeismMonitor(
        reference_data_path='../data/processed/absenteeism_cleaned.csv'
    )

    # Simulate current data (in production, this would be recent predictions)
    current_data = monitor.reference_data.sample(n=100, random_state=42)

    # Generate monitoring report
    print("\n1. Generating monitoring report...")
    report_summary = monitor.create_monitoring_report(current_data)

    # Run test suite
    print("\n2. Running test suite...")
    test_summary = monitor.create_test_suite(current_data)

    # Quick data quality check
    print("\n3. Data quality check...")
    quality = monitor.check_data_quality(current_data)
    print(json.dumps(quality, indent=2))

    # Setup monitoring config
    print("\n4. Setting up monitoring configuration...")
    config = setup_monitoring_config()

    print("\n" + "="*60)
    print("✅ Example completed successfully!")
    print("="*60)


if __name__ == '__main__':
    example_usage()
