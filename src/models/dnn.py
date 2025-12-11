from tensorflow import keras
from typing import Tuple, Optional
import tensorflow as tf
import logging

def build(input_shape: Tuple[int, ...],
          learning_rate: float = 0.001,
          layer_sizes: Optional[Tuple[int, ...]] = None,
          dropout_rate: float = 0.2,
          activation: str = "relu",
          use_batch_norm: bool = False,
          l2_reg: float = 0.0,
          optimizer_type: str = "adam") -> keras.Model:
    """
    Build and compile a Deep Neural Network model for regression tasks.

    Args:
        input_shape: Shape of the input data (features)
        learning_rate: Learning rate for the optimizer
        layer_sizes: Tuple specifying the number of neurons in each hidden layer.
                   If None, uses default [64, 32, 16]
        dropout_rate: Dropout rate for regularization
        activation: Activation function ('relu', 'selu', 'leaky_relu', 'elu')
        use_batch_norm: Whether to use batch normalization
        l2_reg: L2 regularization factor (0.0 for no regularization)
        optimizer_type: Type of optimizer ('adam', 'rmsprop', 'nadam', 'sgd')

    Returns:
        Compiled Keras model ready for training

    Example:
        >>> model = build(input_shape=(8,), learning_rate=0.001, activation="selu")
        >>> model.summary()
    """
    # Set default layer sizes if not provided
    if layer_sizes is None:
        layer_sizes = (64, 32, 16)

    # Set random seeds for reproducibility
    tf.random.set_seed(42)

    model = keras.models.Sequential()

    # Determine appropriate initializer based on activation function
    if activation == "selu":
        kernel_initializer = 'lecun_normal'
    else:
        kernel_initializer = 'he_normal'

    # Input layer
    model.add(keras.layers.Dense(
        layer_sizes[0],
        activation=activation,
        input_shape=input_shape,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    ))

    if use_batch_norm and activation != "selu":  # SELU doesn't work well with BatchNorm
        model.add(keras.layers.BatchNormalization())

    if dropout_rate > 0:
        # Use AlphaDropout for SELU, regular Dropout for others
        if activation == "selu":
            model.add(keras.layers.AlphaDropout(dropout_rate))
        else:
            model.add(keras.layers.Dropout(dropout_rate))

    # Hidden layers
    for size in layer_sizes[1:]:
        model.add(keras.layers.Dense(
            size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
        ))

        if use_batch_norm and activation != "selu":
            model.add(keras.layers.BatchNormalization())

        if dropout_rate > 0:
            if activation == "selu":
                model.add(keras.layers.AlphaDropout(dropout_rate))
            else:
                model.add(keras.layers.Dropout(dropout_rate))

    # Output layer
    model.add(keras.layers.Dense(1))

    # Select optimizer based on type
    try:
        if optimizer_type == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_type == "nadam":
            optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
        elif optimizer_type == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
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