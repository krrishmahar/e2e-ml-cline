from tensorflow import keras
from typing import List
import os
import datetime
import logging

def get_callbacks(model_name: str = "california_housing",
                  patience: int = 5,
                  log_dir: str = "logs") -> List[keras.callbacks.Callback]:
    """
    Create a comprehensive set of callbacks for model training.

    Args:
        model_name: Name for the model (used in callbacks)
        patience: Number of epochs to wait before early stopping
        log_dir: Directory to save training logs and checkpoints

    Returns:
        List of Keras callbacks for training

    Example:
        >>> callbacks = get_callbacks(model_name="experiment_1")
        >>> model.fit(X_train, y_train, callbacks=callbacks)
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp for unique run identification
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_name}_{timestamp}"

    callbacks = []

    # Early Stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Model Checkpoint
    checkpoint_path = os.path.join(log_dir, f"{run_name}_best_model.h5")
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(model_checkpoint)

    # TensorBoard Logging
    tensorboard_log_dir = os.path.join(log_dir, f"{run_name}_logs")
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq="epoch"
    )
    callbacks.append(tensorboard)

    # CSV Logger
    csv_path = os.path.join(log_dir, f"{run_name}_training.log")
    csv_logger = keras.callbacks.CSVLogger(
        filename=csv_path,
        separator=",",
        append=False
    )
    callbacks.append(csv_logger)

    # Reduce Learning Rate on Plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=patience//2,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)

    logging.info(f"Created callbacks: {', '.join([type(cb).__name__ for cb in callbacks])}")

    try:
        return callbacks
    except Exception as e:
        logging.error(f"Error creating callbacks: {str(e)}")
        raise ValueError(f"Failed to create callbacks: {str(e)}")