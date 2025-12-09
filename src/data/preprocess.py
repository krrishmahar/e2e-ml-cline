from sklearn.preprocessing import StandardScaler
from typing import Tuple
import numpy as np
import logging

def preprocess(X_train: np.ndarray,
               X_valid: np.ndarray,
               X_test: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Preprocess input features using StandardScaler for normalization.

    Args:
        X_train: Training features
        X_valid: Validation features
        X_test: Test features

    Returns:
        Tuple of scaled arrays (X_train_scaled, X_valid_scaled, X_test_scaled)

    Raises:
        ValueError: If input arrays have inconsistent shapes or are empty

    Example:
        >>> X_train_scaled, X_valid_scaled, X_test_scaled = preprocess(X_train, X_valid, X_test)
    """
    # Input validation
    if len(X_train) == 0 or len(X_valid) == 0 or len(X_test) == 0:
        raise ValueError("Input arrays cannot be empty")

    if X_train.shape[1] != X_valid.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("All input arrays must have the same number of features")

    logging.info(f"Preprocessing data with {X_train.shape[1]} features")

    try:
        scaler = StandardScaler()

        # Fit on training data only to avoid data leakage
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        logging.info("Data preprocessing completed successfully")
        logging.debug(f"Feature means: {scaler.mean_}")
        logging.debug(f"Feature stds: {scaler.scale_}")

        return X_train_scaled, X_valid_scaled, X_test_scaled

    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise