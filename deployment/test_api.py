"""
Test script for the Absenteeism Prediction API
Author: Alexis Alduncin
Team: MLOps 62
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Health check failed"
    print("✅ Health check passed")

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("Testing /predict endpoint")
    print("="*60)

    # Sample employee data
    data = {
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

    print(f"Input data: {json.dumps(data, indent=2)}")

    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Single prediction failed"
    assert 'prediction' in response.json(), "Prediction not in response"
    print(f"✅ Single prediction passed: {response.json()['prediction']:.2f} hours")

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing /batch_predict endpoint")
    print("="*60)

    # Sample batch data (3 employees)
    data = {
        "instances": [
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
            },
            {
                "Age": 42,
                "Distance from Residence to Work": 25,
                "Service time": 15,
                "Body mass index": 28.0,
                "Work load Average/day": 280,
                "Hit target": 90,
                "Transportation expense": 250,
                "Disciplinary failure": 0,
                "Education": 2,
                "Son": 2,
                "Social drinker": 1,
                "Social smoker": 1,
                "Pet": 0,
                "Weight": 85,
                "Height": 180,
                "Reason for absence": 10,
                "Month of absence": 12,
                "Day of the week": 2,
                "Seasons": 3
            },
            {
                "Age": 28,
                "Distance from Residence to Work": 5,
                "Service time": 3,
                "Body mass index": 22.0,
                "Work load Average/day": 220,
                "Hit target": 98,
                "Transportation expense": 150,
                "Disciplinary failure": 0,
                "Education": 1,
                "Son": 0,
                "Social drinker": 0,
                "Social smoker": 0,
                "Pet": 1,
                "Weight": 65,
                "Height": 170,
                "Reason for absence": 0,
                "Month of absence": 3,
                "Day of the week": 4,
                "Seasons": 4
            }
        ]
    }

    print(f"Batch size: {len(data['instances'])}")

    response = requests.post(f"{BASE_URL}/batch_predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Batch prediction failed"
    assert 'predictions' in response.json(), "Predictions not in response"
    assert len(response.json()['predictions']) == len(data['instances']), "Wrong number of predictions"
    print(f"✅ Batch prediction passed: {len(response.json()['predictions'])} predictions returned")

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing /model_info endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Model info failed"
    print("✅ Model info passed")

def test_error_handling():
    """Test error handling"""
    print("\n" + "="*60)
    print("Testing error handling")
    print("="*60)

    # Test missing features
    print("\n1. Testing missing features...")
    data = {"Age": 35}  # Missing required features
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 400, "Should return 400 for missing features"
    print("✅ Missing features handled correctly")

    # Test invalid endpoint
    print("\n2. Testing invalid endpoint...")
    response = requests.get(f"{BASE_URL}/invalid_endpoint")
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 404, "Should return 404 for invalid endpoint"
    print("✅ Invalid endpoint handled correctly")

if __name__ == '__main__':
    print("="*60)
    print("API Test Suite")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")
    print("\nMake sure the API is running: python app.py")
    print("="*60)

    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        test_error_handling()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running: python app.py")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
