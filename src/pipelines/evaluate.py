from tensorflow import keras
import logging
import os
import numpy as np
from typing import Union, List

def predict(data: Union[List, np.ndarray], model_path: str = "models/final_model.h5") -> List[float]:
    """
    Make predictions using a trained model.

    Args:
        data: Input data for prediction (list or numpy array)
        model_path: Path to the trained model file

    Returns:
        List of predictions as floats

    Raises:
        ValueError: If input data is None, empty, or invalid
        FileNotFoundError: If model file doesn't exist
        Exception: For other prediction errors

    Example:
        >>> predictions = predict([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        >>> print(f"Predictions: {predictions}")
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Input validation
        if data is None:
            raise ValueError("Input data cannot be None")

        if isinstance(data, list) and len(data) == 0:
            raise ValueError("Input data list cannot be empty")

        if isinstance(data, np.ndarray) and data.size == 0:
            raise ValueError("Input numpy array cannot be empty")

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        logging.info(f"Loading model from {model_path}")

        # Load model and make predictions
        model = keras.models.load_model(model_path)

        # Convert input to numpy array if it's a list
        if isinstance(data, list):
            data_array = np.array(data, dtype=np.float32)
        else:
            data_array = data

        logging.info(f"Making predictions on {len(data_array)} samples")
        predictions = model.predict(data_array).tolist()

        logging.info("Predictions completed successfully")
        return predictions

    except ValueError as ve:
        logging.error(f"Input validation error: {str(ve)}")
        raise
    except FileNotFoundError as fe:
        logging.error(f"Model file error: {str(fe)}")
        raise
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise ValueError(f"Failed to make predictions: {str(e)}")