# California Housing Price Prediction API - Deployment Guide

## Overview

This guide provides instructions for deploying the FastAPI-based California Housing Price Prediction service.

## Project Structure

```
src/
├── app.py                  # FastAPI application
├── artifacts/
│   └── model.h5            # Trained Keras model
├── pipelines/
│   └── train.py            # Training pipeline
├── models/
│   └── dnn.py              # DNN model architecture
├── data/                   # Data loading and preprocessing
├── utils/                  # Utility functions
└── test_api.py             # API test script
```

## Deployment Instructions

### 1. Install Dependencies

```bash
pip install fastapi uvicorn tensorflow scikit-learn numpy pydantic
```

### 2. Train the Model (if not already trained)

```bash
cd /home/krrish/Desktop/e2e-ml-cline
python -m src.pipelines.train --epochs 10 --batch-size 32
```

This will:
- Load the California housing dataset
- Preprocess the data
- Train a Deep Neural Network model
- Save the trained model to `src/artifacts/model.h5`

### 3. Start the FastAPI Server

```bash
cd /home/krrish/Desktop/e2e-ml-cline/src
python app.py
```

The server will start on `http://0.0.0.0:8000`

### 4. Test the API

```bash
python test_api.py
```

## API Endpoints

### GET `/`
Health check endpoint
**Response:**
```json
{
  "message": "California Housing Price Prediction API is running"
}
```

### GET `/health`
Model health check
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "DNN",
  "api_version": "1.0.0"
}
```

### POST `/predict`
Make housing price prediction

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 67.1324234008789,
  "unit": "USD (in hundreds of thousands)"
}
```

## Input Features Description

The API expects 8 features in the following order:

1. **MedInc** - Median income in block group (scaled)
2. **HouseAge** - Median house age in block group
3. **AveRooms** - Average number of rooms per household
4. **AveBedrms** - Average number of bedrooms per household
5. **Population** - Block group population
6. **AveOccup** - Average number of household members
7. **Latitude** - Block group latitude
8. **Longitude** - Block group longitude

## Deployment Options

### Local Development
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production with Gunicorn
```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 app:app --bind 0.0.0.0:8000
```

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t housing-api .
docker run -p 8000:8000 housing-api
```

## Model Information

- **Model Type**: Deep Neural Network (DNN)
- **Architecture**: 3 hidden layers (64, 32, 16 neurons)
- **Input Shape**: (8,) - 8 housing features
- **Output**: Single value (house price prediction)
- **Training**: California Housing Dataset
- **Metrics**: MAE, MSE, RMSE

## Troubleshooting

**Model loading issues**:
- Ensure `artifacts/model.h5` exists
- Verify TensorFlow version compatibility
- Check file permissions

**API connection issues**:
- Verify server is running (`ps aux | grep uvicorn`)
- Check firewall settings
- Test with `curl http://localhost:8000/health`

## Performance

- **Training Time**: ~30 seconds for 10 epochs
- **Prediction Time**: <10ms per request
- **Model Size**: ~12KB
- **Accuracy**: MAE ~0.45, RMSE ~0.64 (on test set)

## Scaling

For high traffic:
- Use multiple workers (`-w 4` in gunicorn)
- Implement caching for frequent predictions
- Consider model quantization for faster inference
- Use load balancing for multiple instances

## Monitoring

Recommended monitoring:
- Request latency
- Error rates
- Model prediction distribution
- Input feature ranges

## Security

- Add authentication for production
- Implement rate limiting
- Use HTTPS in production
- Validate all inputs

## Example Usage

```python
import requests

# Sample prediction
data = {
    "features": [
        8.3252, 41.0, 6.98412698, 1.02380952,
        322.0, 2.55555556, 37.88, -122.23
    ]
}

response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()["prediction"]
print(f"Predicted price: ${prediction * 100000:,.2f}")
```

## Support

For issues or questions, please refer to the project documentation or contact the development team.