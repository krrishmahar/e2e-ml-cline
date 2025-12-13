import requests
import json
import numpy as np
from fastapi.testclient import TestClient
from src.app import app

def test_api_functionality():
    """
    Test the FastAPI application functionality.
    """
    # Create test client
    client = TestClient(app)

    # Manually trigger model loading for testing
    from src.app import startup_event
    import asyncio

    # Run the startup event to load the model
    asyncio.run(startup_event())

    print("Testing FastAPI California Housing Price Prediction API...")

    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    response = client.get("/")
    print(f"Root endpoint response: {response.json()}")
    assert response.status_code == 200

    # Test 2: Health check endpoint
    print("\n2. Testing health check endpoint...")
    response = client.get("/health")
    print(f"Health check response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] == True

    # Test 3: Prediction endpoint with sample data
    print("\n3. Testing prediction endpoint...")

    # Create sample housing data (8 features)
    # These are typical values for California housing dataset
    sample_data = {
        "features": [
            8.3252,    # MedInc - median income
            41.0,      # HouseAge - median house age
            6.98412698, # AveRooms - average rooms
            1.02380952, # AveBedrms - average bedrooms
            322.0,     # Population
            2.55555556, # AveOccup - average occupancy
            37.88,     # Latitude
            -122.23    # Longitude
        ]
    }

    response = client.post("/predict", json=sample_data)
    print(f"Prediction response: {response.json()}")
    assert response.status_code == 200

    prediction_result = response.json()
    assert "prediction" in prediction_result
    assert isinstance(prediction_result["prediction"], float)
    assert prediction_result["unit"] == "USD (in hundreds of thousands)"

    print(f"\n✅ All tests passed!")
    print(f"🏠 Sample prediction: ${prediction_result['prediction'] * 100000:,.2f}")
    print(f"📊 API is working correctly and ready for deployment!")

if __name__ == "__main__":
    test_api_functionality()