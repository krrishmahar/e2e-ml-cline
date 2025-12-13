from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import numpy as np
import logging
import pickle
import os

# Global scaler cache for performance optimization
_scaler_cache = None
_scaler_cache_path = "data/scaler_cache.pkl"

def _validate_input_arrays(X_train: np.ndarray,
                          X_valid: np.ndarray,
                          X_test: np.ndarray) -> None:
    """
    Validate input arrays for preprocessing.

    Args:
        X_train: Training features
        X_valid: Validation features
        X_test: Test features

    Raises:
        ValueError: If input arrays have inconsistent shapes or are empty
    """
    if len(X_train) == 0 or len(X_valid) == 0 or len(X_test) == 0:
        raise ValueError("Input arrays cannot be empty")

    if X_train.shape[1] != X_valid.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("All input arrays must have the same number of features")

def _load_cached_scaler() -> Optional[StandardScaler]:
    """
    Load scaler from cache file if available.

    Returns:
        Cached StandardScaler instance if available, None otherwise
    """
    try:
        if os.path.exists(_scaler_cache_path):
            with open(_scaler_cache_path, 'rb') as f:
                scaler = pickle.load(f)
                logging.info("Loaded scaler from cache file")
                return scaler
    except Exception as e:
        logging.warning(f"Failed to load scaler cache: {str(e)}")
    return None

def _save_scaler_cache(scaler: StandardScaler) -> None:
    """
    Save scaler to cache file for persistence.

    Args:
        scaler: StandardScaler instance to cache
    """
    try:
        os.makedirs(os.path.dirname(_scaler_cache_path), exist_ok=True)
        with open(_scaler_cache_path, 'wb') as f:
            pickle.dump(scaler, f)
        logging.debug("Scaler saved to cache file")
    except Exception as cache_error:
        logging.warning(f"Failed to save scaler cache: {str(cache_error)}")

def _get_scaler(use_cache: bool = True) -> StandardScaler:
    """
    Get scaler instance, either from cache or create new one.

    Args:
        use_cache: Whether to use cached scaler

    Returns:
        StandardScaler instance
    """
    global _scaler_cache

    if use_cache:
        # Try to load from memory cache first
        if _scaler_cache is not None:
            logging.info("Using cached scaler from memory")
            return _scaler_cache

        # Try to load from file cache
        cached_scaler = _load_cached_scaler()
        if cached_scaler is not None:
            _scaler_cache = cached_scaler
            return cached_scaler

    # Create new scaler if no cache available or caching disabled
    logging.debug("Creating new scaler instance")
    return StandardScaler()

def preprocess(X_train: np.ndarray,
               X_valid: np.ndarray,
               X_test: np.ndarray,
               use_cache: bool = True) -> Tuple[np.ndarray, ...]:
    """
    Preprocess input features using StandardScaler for normalization.

    Args:
        X_train: Training features
        X_valid: Validation features
        X_test: Test features
        use_cache: Whether to use cached scaler for performance optimization

    Returns:
        Tuple of scaled arrays (X_train_scaled, X_valid_scaled, X_test_scaled)

    Raises:
        ValueError: If input arrays have inconsistent shapes or are empty

    Example:
        >>> X_train_scaled, X_valid_scaled, X_test_scaled = preprocess(X_train, X_valid, X_test)
    """
    # Input validation
    _validate_input_arrays(X_train, X_valid, X_test)
    logging.info(f"Preprocessing data with {X_train.shape[1]} features")

    try:
        # Get scaler (cached or new)
        scaler = _get_scaler(use_cache)

        # Fit on training data only to avoid data leakage
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        # Cache the scaler for future use
        if use_cache:
            _scaler_cache = scaler
            _save_scaler_cache(scaler)

        logging.info("Data preprocessing completed successfully")
        logging.debug(f"Feature means: {scaler.mean_}")
        logging.debug(f"Feature stds: {scaler.scale_}")

        return X_train_scaled, X_valid_scaled, X_test_scaled

    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise