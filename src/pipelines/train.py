import argparse
import logging
import os
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data.load_data import load
from src.data.preprocess import preprocess
from src.models.dnn import build
from src.utils.callbacks import get_callbacks
from src.utils.learning_rate import create_learning_rate_schedule, apply_learning_rate_schedule


def train(dry_run: bool = False,
          epochs: int = 100,
          batch_size: int = 32,
          learning_rate: float = 0.001,
          layer_sizes: Optional[tuple] = None,
          test_size: float = 0.2,
          validation_split: float = 0.3,
          random_state: int = 42,
          activation: str = "relu",
          use_batch_norm: bool = False,
          l2_reg: float = 0.0,
          optimizer_type: str = "adam",
          use_lr_schedule: bool = False,
          lr_schedule_type: str = "cosine") -> Optional[tf.keras.Model]:
    """
    Train a neural network model on the California housing dataset.

    Args:
        dry_run: If True, only build the model without training
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        layer_sizes: Tuple specifying hidden layer sizes
        test_size: Proportion of data for testing
        validation_split: Proportion of training data for validation
        random_state: Random seed for reproducibility
        activation: Activation function ('relu', 'selu', 'leaky_relu', 'elu')
        use_batch_norm: Whether to use batch normalization
        l2_reg: L2 regularization factor
        optimizer_type: Type of optimizer ('adam', 'rmsprop', 'nadam', 'sgd')
        use_lr_schedule: Whether to use learning rate scheduling
        lr_schedule_type: Type of learning rate schedule ('cosine', 'exponential')

    Returns:
        Trained model if not dry_run, None otherwise

    Example:
        >>> model = train(epochs=50, learning_rate=0.001, activation="selu", use_batch_norm=True)
        >>> model.summary()
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Set random seeds for reproducibility
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    logging.info("Starting training pipeline")

    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        X_train, X_valid, X_test, y_train, y_valid, y_test = load(
            test_size=test_size,
            validation_split=validation_split,
            random_state=random_state
        )

        X_train, X_valid, X_test = preprocess(X_train, X_valid, X_test)

        logging.info(f"Data shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

        # Build model
        logging.info("Building model...")
        model = build(
            input_shape=X_train.shape[1:],
            learning_rate=learning_rate,
            layer_sizes=layer_sizes,
            activation=activation,
            use_batch_norm=use_batch_norm,
            l2_reg=l2_reg,
            optimizer_type=optimizer_type
        )

        model.summary()

        if dry_run:
            logging.info("Dry run completed. Model built successfully.")
            return None

        # Prepare callbacks
        callbacks = get_callbacks(
            model_name="california_housing",
            patience=10,
            log_dir="logs/training_logs"
        )

        # Add learning rate scheduling if enabled
        if use_lr_schedule:
            decay_steps = epochs * (X_train.shape[0] // batch_size)
            lr_schedule = create_learning_rate_schedule(
                lr_schedule_type, learning_rate, decay_steps
            )

            apply_learning_rate_schedule(model, optimizer_type, lr_schedule)
            logging.info(f"Using {lr_schedule_type} learning rate schedule")

        # Train model
        logging.info("Starting model training...")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        os.makedirs("artifact", exist_ok=True)
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "model.h5"))
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")

        # Evaluate on test set
        logging.info("Evaluating model on test set...")
        test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)

        logging.info(f"Test Results - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {np.sqrt(test_mse):.4f}")

        # Log training summary
        final_epoch = len(history.history['loss'])
        logging.info(f"Training completed in {final_epoch} epochs")
        logging.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")

        return model

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train California Housing Price Prediction Model")
    parser.add_argument("--dry-run", action="store_true", help="Build model without training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer sizes (e.g., '64,32,16')")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--validation-split", type=float, default=0.3, help="Validation set proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--activation", type=str, default="relu",
                       help="Activation function ('relu', 'selu', 'leaky_relu', 'elu')")
    parser.add_argument("--batch-norm", action="store_true", help="Use batch normalization")
    parser.add_argument("--l2-reg", type=float, default=0.0, help="L2 regularization factor")
    parser.add_argument("--optimizer", type=str, default="adam",
                       help="Optimizer type ('adam', 'rmsprop', 'nadam', 'sgd')")
    parser.add_argument("--lr-schedule", action="store_true", help="Use learning rate scheduling")
    parser.add_argument("--lr-schedule-type", type=str, default="cosine",
                       help="Learning rate schedule type ('cosine', 'exponential')")

    args = parser.parse_args()

    # Parse layer sizes if provided
    layer_sizes = None
    if args.layers:
        layer_sizes = tuple(int(size) for size in args.layers.split(','))

    train(
        dry_run=args.dry_run,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        layer_sizes=layer_sizes,
        test_size=args.test_size,
        validation_split=args.validation_split,
        random_state=args.random_state,
        activation=args.activation,
        use_batch_norm=args.batch_norm,
        l2_reg=args.l2_reg,
        optimizer_type=args.optimizer,
        use_lr_schedule=args.lr_schedule,
        lr_schedule_type=args.lr_schedule_type
    )
