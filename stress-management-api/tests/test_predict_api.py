import pytest
from fastapi.testclient import TestClient
from API.main import app

client = TestClient(app)

def test_predict_stress_level():
    # Example input data for prediction
    input_data = {
        "feature1": 1.0,
        "feature2": 2.0,
        "feature3": "example_value",
        "gpa": None  # simulate missing value
    }
    
    response = client.post("/predict", json=input_data)
    
    assert response.status_code == 200
    assert "predicted_stress_level" in response.json()

def test_health_check():
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}