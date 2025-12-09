from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np
import logging

def load(test_size: float = 0.2,
         validation_split: float = 0.3,
         random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Load and split the California housing dataset into training, validation, and test sets.

    Args:
        test_size: Proportion of data to use for testing (0.0-1.0)
        validation_split: Proportion of training data to use for validation (0.0-1.0)
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing (X_train, X_valid, X_test, y_train, y_valid, y_test)

    Raises:
        ValueError: If test_size or validation_split are not in valid range [0, 1]

    Example:
        >>> X_train, X_valid, X_test, y_train, y_valid, y_test = load()
        >>> print(f"Training samples: {X_train.shape[0]}")
    """
    # Validate input parameters
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if not (0 < validation_split < 1):
        raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}")

    # Set random seed for reproducibility
    np.random.seed(random_state)

    try:
        # Load dataset
        housing = fetch_california_housing()
        logging.info(f"Loaded California housing dataset with {housing.data.shape[0]} samples")

        # First split: training + validation vs test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target,
            test_size=test_size,
            random_state=random_state
        )

        # Second split: training vs validation
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full,
            test_size=validation_split,
            random_state=random_state
        )

        logging.info(f"Data split - Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}, Test: {X_test.shape[0]}")

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise