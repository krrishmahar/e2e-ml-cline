import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define input data model
class HousingFeatures(BaseModel):
    """
    Input features for California housing price prediction.

    Features represent:
    0. MedInc - median income in block group
    1. HouseAge - median house age in block group
    2. AveRooms - average number of rooms per household
    3. AveBedrms - average number of bedrooms per household
    4. Population - block group population
    5. AveOccup - average number of household members
    6. Latitude - block group latitude
    7. Longitude - block group longitude
    """
    features: List[float] = Field(
        ...,
        min_items=8,
        max_items=8,
        description="List of 8 housing features for prediction"
    )

class PredictionResponse(BaseModel):
    prediction: float
    unit: str = "USD (in hundreds of thousands)"

# Initialize FastAPI app
app = FastAPI(
    title="California Housing Price Prediction API",
    description="API for predicting California housing prices using a trained DNN model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded model
model = None

def load_model(model_path: str) -> keras.Model:
    """
    Load the trained Keras model from the specified path.

    Args:
        model_path: Path to the .h5 model file

    Returns:
        Loaded Keras model

    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info(f"Loading model from {model_path}")

        # Try loading with custom objects for compatibility
        try:
            loaded_model = keras.models.load_model(model_path)
        except ValueError as e:
            if "Could not deserialize" in str(e):
                # Handle compatibility issues with custom objects
                from tensorflow.keras import losses, metrics

                custom_objects = {
                    'mse': losses.MeanSquaredError(),
                    'mae': metrics.MeanAbsoluteError(),
                    'MeanSquaredError': losses.MeanSquaredError,
                    'MeanAbsoluteError': metrics.MeanAbsoluteError
                }

                loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)
            else:
                raise

        logger.info("Model loaded successfully")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Could not load model: {str(e)}")

def get_model_path() -> str:
    """
    Get the absolute path to the model file.

    Returns:
        Absolute path to model.h5 file
    """
    # Construct path relative to this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "artifact", "model.h5")
    return model_path

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler to load the model when the app starts.
    """
    global model
    model_path = get_model_path()

    # Check if model file exists
    if not os.path.exists(model_path):
        error_msg = f"Model file not found at {model_path}. Please train the model first."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Load the model
    model = load_model(model_path)

@app.get("/")
async def root():
    """
    Root endpoint for health check.
    """
    return {"message": "California Housing Price Prediction API is running"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and model status.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "DNN",
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(housing_data: HousingFeatures):
    """
    Predict housing price based on input features.

    Args:
        housing_data: Input features for prediction

    Returns:
        Prediction response with predicted price

    Raises:
        HTTPException: If prediction fails or input is invalid
    """
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to numpy array and reshape for prediction
        input_array = np.array(housing_data.features, dtype=np.float32)
        input_array = input_array.reshape(1, -1)  # Reshape to (1, 8)

        # Make prediction
        prediction = model.predict(input_array, verbose=0)
        predicted_price = float(prediction[0][0])

        logger.info(f"Prediction successful: {predicted_price}")

        return PredictionResponse(prediction=predicted_price)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    # Check if model exists before starting
    model_path = get_model_path()
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training pipeline first:")
        print("python src/pipelines/train.py")
        exit(1)

    # Start the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )