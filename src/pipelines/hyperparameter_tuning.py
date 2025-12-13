import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
from datetime import datetime
from src.models.dnn import build

def tune_hyperparameters(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_valid: np.ndarray,
                        y_valid: np.ndarray,
                        param_grid: Dict,
                        max_trials: int = 10,
                        epochs: int = 50,
                        batch_size: int = 32,
                        random_state: int = 42) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform hyperparameter tuning using random search.

    Args:
        X_train: Training features
        y_train: Training target values
        X_valid: Validation features
        y_valid: Validation target values
        param_grid: Dictionary of hyperparameter distributions to search
        max_trials: Maximum number of trials to run
        epochs: Number of training epochs per trial
        batch_size: Batch size for training
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (best_params, results_df) where best_params is the best hyperparameter
        combination and results_df is a DataFrame with all trial results

    Example:
        >>> param_grid = {
        ...     'learning_rate': [0.001, 0.0001, 0.01],
        ...     'layer_sizes': [(64, 32), (128, 64, 32), (256, 128, 64)],
        ...     'activation': ['relu', 'selu'],
        ...     'dropout_rate': [0.1, 0.2, 0.3]
        ... }
        >>> best_params, results = tune_hyperparameters(X_train, y_train, X_valid, y_valid, param_grid)
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Set random seeds for reproducibility
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    try:
        logging.info("Starting hyperparameter tuning")

        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"tuning_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        # Initialize results storage
        results = []

        # Set up random search
        np.random.seed(random_state)

        best_val_loss = float('inf')
        best_params = None
        best_model = None

        # Run trials
        for trial in range(max_trials):
            logging.info(f"Starting trial {trial + 1}/{max_trials}")

            # Sample hyperparameters
            trial_params = {}
            for param_name, param_options in param_grid.items():
                if isinstance(param_options, list):
                    # Random choice from list
                    trial_params[param_name] = np.random.choice(param_options)
                elif isinstance(param_options, dict):
                    # Sample from distribution
                    if param_options['type'] == 'uniform':
                        trial_params[param_name] = np.random.uniform(
                            param_options['min'], param_options['max']
                        )
                    elif param_options['type'] == 'loguniform':
                        trial_params[param_name] = np.exp(np.random.uniform(
                            np.log(param_options['min']), np.log(param_options['max'])
                        ))
                    elif param_options['type'] == 'int':
                        trial_params[param_name] = np.random.randint(
                            param_options['min'], param_options['max'] + 1
                        )
                else:
                    trial_params[param_name] = param_options

            logging.info(f"Trial {trial + 1} parameters: {trial_params}")

            # Build model with sampled parameters using unified build function
            model = build(
                input_shape=X_train.shape[1:],
                **trial_params
            )

            # Train model
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_valid, y_valid),
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True,
                        verbose=0
                    )
                ]
            )

            # Evaluate on validation set
            val_loss = min(history.history['val_loss'])
            trial_params['val_loss'] = val_loss
            trial_params['trial'] = trial + 1

            results.append(trial_params)

            # Save trial results
            trial_results_path = os.path.join(results_dir, f"trial_{trial + 1}_results.json")
            with open(trial_results_path, 'w') as f:
                json.dump(trial_params, f, indent=2)

            logging.info(f"Trial {trial + 1} completed - Validation Loss: {val_loss:.6f}")

            # Check if this is the best trial so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = trial_params.copy()
                best_model = model

                # Save best model
                best_model_path = os.path.join(results_dir, "best_model.h5")
                best_model.save(best_model_path)
                logging.info(f"New best model found! Validation Loss: {val_loss:.6f}")

        # Save all results
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(results_dir, "all_results.csv")
        results_df.to_csv(results_csv_path, index=False)

        # Save best parameters
        best_params_path = os.path.join(results_dir, "best_params.json")
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)

        logging.info(f"Hyperparameter tuning completed")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best validation loss: {best_val_loss:.6f}")
        logging.info(f"Results saved to: {results_dir}")

        return best_params, results_df

    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {str(e)}")
        raise ValueError(f"Failed to tune hyperparameters: {str(e)}")


def create_default_param_grid() -> Dict:
    """
    Create a default parameter grid for hyperparameter tuning.

    Returns:
        Dictionary with default parameter distributions
    """
    return {
        'learning_rate': {
            'type': 'loguniform',
            'min': 0.0001,
            'max': 0.01
        },
        'layer_sizes': [
            (64, 32),
            (128, 64, 32),
            (256, 128, 64),
            (128, 64, 32, 16)
        ],
        'activation': ['relu', 'selu', 'elu'],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4],
        'use_batch_norm': [True, False],
        'l2_reg': [0.0, 0.001, 0.01],
        'optimizer_type': ['adam', 'rmsprop', 'nadam'],
        'batch_size': [16, 32, 64]
    }

if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load
    from src.data.preprocess import preprocess

    # Load and preprocess data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load()
    X_train, X_valid, X_test = preprocess(X_train, X_valid, X_test)

    # Create parameter grid
    param_grid = create_default_param_grid()

    # Run hyperparameter tuning
    best_params, results_df = tune_hyperparameters(
        X_train, y_train, X_valid, y_valid,
        param_grid=param_grid,
        max_trials=5,
        epochs=20,
        batch_size=32
    )

    print("Hyperparameter tuning completed!")
    print(f"Best parameters: {best_params}")
    print(f"Results shape: {results_df.shape}")