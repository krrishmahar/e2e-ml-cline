from tensorflow import keras
from typing import Tuple, Optional
import tensorflow as tf

def build(input_shape: Tuple[int, ...],
          learning_rate: float = 0.001,
          layer_sizes: Optional[Tuple[int, ...]] = None,
          dropout_rate: float = 0.2) -> keras.Model:
    """
    Build and compile a Deep Neural Network model for regression tasks.

    Args:
        input_shape: Shape of the input data (features)
        learning_rate: Learning rate for the Adam optimizer
        layer_sizes: Tuple specifying the number of neurons in each hidden layer.
                   If None, uses default [64, 32, 16]
        dropout_rate: Dropout rate for regularization

    Returns:
        Compiled Keras model ready for training

    Example:
        >>> model = build(input_shape=(8,), learning_rate=0.001)
        >>> model.summary()
    """
    # Set default layer sizes if not provided
    if layer_sizes is None:
        layer_sizes = (64, 32, 16)

    # Set random seeds for reproducibility
    tf.random.set_seed(42)

    model = keras.models.Sequential()

    # Input layer
    model.add(keras.layers.Dense(
        layer_sizes[0],
        activation="relu",
        input_shape=input_shape,
        kernel_initializer='he_normal'
    ))
    model.add(keras.layers.Dropout(dropout_rate))

    # Hidden layers
    for size in layer_sizes[1:]:
        model.add(keras.layers.Dense(
            size,
            activation="relu",
            kernel_initializer='he_normal'
        ))
        model.add(keras.layers.Dropout(dropout_rate))

    # Output layer
    model.add(keras.layers.Dense(1))

    # Custom optimizer with specified learning rate
    try:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae", "mse"]
        )

        return model
    except Exception as e:
        logging.error(f"Error building model: {str(e)}")
        raise ValueError(f"Failed to build model: {str(e)}")