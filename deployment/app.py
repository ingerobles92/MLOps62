"""
Flask API for Absenteeism Prediction Model
Author: Alexis Alduncin (Data Scientist)
Team: MLOps 62

Provides REST API endpoints for the production absenteeism prediction model.
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

from src.models.pipelines import create_full_pipeline

app = Flask(__name__)

# Global model variable (loaded on startup)
model = None
model_metadata = {}

def load_model():
    """Load the trained model from disk"""
    global model, model_metadata

    model_path = '../models/best_model_svr.pkl'

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please run the model export script first: python scripts/export_best_model.py"
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load metadata if available
    metadata_path = '../models/model_metadata.json'
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)

    print(f"✅ Model loaded successfully from {model_path}")
    if model_metadata:
        print(f"   Model Type: {model_metadata.get('model_type', 'Unknown')}")
        print(f"   Test MAE: {model_metadata.get('test_mae', 'Unknown')}")
        print(f"   Training Date: {model_metadata.get('train_date', 'Unknown')}")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    }

    if model is not None and model_metadata:
        status.update({
            'model_type': model_metadata.get('model_type', 'Unknown'),
            'test_mae': model_metadata.get('test_mae', 'Unknown'),
            'training_date': model_metadata.get('train_date', 'Unknown')
        })

    return jsonify(status), 200 if model is not None else 503

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint

    Expected JSON format:
    {
        "Age": 35,
        "Distance from Residence to Work": 15,
        "Service time": 10,
        "Body mass index": 25.5,
        ...
    }

    Returns:
    {
        "prediction": 4.2,
        "unit": "hours",
        "model": "SVR_RBF",
        "timestamp": "2024-10-30T10:30:00"
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        # Get input data
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Validate required features
        required_features = [
            'Age', 'Body mass index', 'Distance from Residence to Work',
            'Work load Average/day', 'Service time', 'Hit target'
        ]

        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': missing_features
            }), 400

        # Make prediction
        prediction = model.predict(df)[0]

        # Return result
        return jsonify({
            'prediction': float(prediction),
            'unit': 'hours',
            'model': model_metadata.get('model_type', 'SVR_RBF'),
            'timestamp': datetime.now().isoformat(),
            'confidence': 'MAE ± {}'.format(model_metadata.get('test_mae', 'Unknown'))
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint

    Expected JSON format:
    {
        "instances": [
            {"Age": 35, "Distance from Residence to Work": 15, ...},
            {"Age": 42, "Distance from Residence to Work": 25, ...},
            ...
        ]
    }

    Returns:
    {
        "predictions": [4.2, 6.1, ...],
        "count": 10,
        "model": "SVR_RBF",
        "timestamp": "2024-10-30T10:30:00"
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        # Get input data
        data = request.json
        if not data or 'instances' not in data:
            return jsonify({'error': 'No instances provided. Expected format: {"instances": [...]}'}), 400

        instances = data['instances']
        if not isinstance(instances, list) or len(instances) == 0:
            return jsonify({'error': 'Instances must be a non-empty list'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(instances)

        # Validate required features
        required_features = [
            'Age', 'Body mass index', 'Distance from Residence to Work',
            'Work load Average/day', 'Service time', 'Hit target'
        ]

        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': missing_features
            }), 400

        # Make predictions
        predictions = model.predict(df)

        # Return results
        return jsonify({
            'predictions': predictions.tolist(),
            'count': len(predictions),
            'model': model_metadata.get('model_type', 'SVR_RBF'),
            'timestamp': datetime.now().isoformat(),
            'confidence': 'MAE ± {}'.format(model_metadata.get('test_mae', 'Unknown'))
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    info = {
        'status': 'loaded',
        'metadata': model_metadata,
        'timestamp': datetime.now().isoformat()
    }

    return jsonify(info), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health',
            'POST /predict',
            'POST /batch_predict',
            'GET /model_info'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

if __name__ == '__main__':
    print("="*60)
    print("Absenteeism Prediction API")
    print("="*60)

    # Load model on startup
    try:
        load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("⚠️  API will start but predictions will not be available")

    print("\nStarting Flask server...")
    print("API endpoints:")
    print("  - GET  /health           : Health check")
    print("  - POST /predict          : Single prediction")
    print("  - POST /batch_predict    : Batch predictions")
    print("  - GET  /model_info       : Model metadata")
    print("="*60)

    app.run(host='0.0.0.0', port=5000, debug=True)
