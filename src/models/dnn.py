from tensorflow import keras
from typing import Tuple, Optional, Union
import tensorflow as tf
import logging

def _validate_build_parameters(input_shape: Tuple[int, ...],
                             layer_sizes: Optional[Tuple[int, ...]],
                             dropout_rate: float,
                             l2_reg: float,
                             learning_rate: float) -> Tuple[Tuple[int, ...], str]:
    """
    Validate and normalize model building parameters.

    Args:
        input_shape: Shape of the input data
        layer_sizes: Tuple specifying hidden layer sizes
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for optimizer

    Returns:
        Tuple of (normalized_layer_sizes, kernel_initializer)

    Raises:
        ValueError: If any parameter is invalid
    """
    # Set default layer sizes if not provided
    if layer_sizes is None:
        layer_sizes = (64, 32, 16)

    # Validate input parameters
    if not isinstance(input_shape, tuple) or len(input_shape) == 0:
        raise ValueError(f"Invalid input_shape: {input_shape}")

    if not isinstance(layer_sizes, tuple) or len(layer_sizes) == 0:
        raise ValueError(f"Invalid layer_sizes: {layer_sizes}")

    if not (0 <= dropout_rate <= 1):
        raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

    if not (0 <= l2_reg):
        raise ValueError(f"l2_reg must be non-negative, got {l2_reg}")

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    return layer_sizes

def _get_optimizer(optimizer_type: str, learning_rate: float) -> keras.optimizers.Optimizer:
    """
    Create and return the appropriate optimizer based on type.

    Args:
        optimizer_type: Type of optimizer
        learning_rate: Learning rate for optimizer

    Returns:
        Configured Keras optimizer

    Raises:
        ValueError: If optimizer type is invalid
    """
    optimizer_map = {
        "adam": keras.optimizers.Adam,
        "rmsprop": keras.optimizers.RMSprop,
        "nadam": keras.optimizers.Nadam,
        "sgd": lambda lr: keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    }

    if optimizer_type not in optimizer_map:
        logging.warning(f"Unknown optimizer type '{optimizer_type}', defaulting to Adam")
        return keras.optimizers.Adam(learning_rate=learning_rate)

    optimizer_factory = optimizer_map[optimizer_type]
    return optimizer_factory(learning_rate=learning_rate)

def _add_regularization_layers(model: keras.Sequential,
                              activation: str,
                              dropout_rate: float,
                              use_batch_norm: bool) -> None:
    """
    Add batch normalization and dropout layers based on activation function.

    Args:
        model: Keras Sequential model
        activation: Activation function name
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch normalization
    """
    if use_batch_norm and activation != "selu":
        model.add(keras.layers.BatchNormalization())

    if dropout_rate > 0:
        dropout_layer = keras.layers.AlphaDropout if activation == "selu" else keras.layers.Dropout
        model.add(dropout_layer(dropout_rate))

def _get_kernel_initializer(activation: str) -> str:
    """
    Get appropriate kernel initializer based on activation function.

    Args:
        activation: Activation function name

    Returns:
        Kernel initializer name
    """
    return 'lecun_normal' if activation == "selu" else 'he_normal'

def build(input_shape: Tuple[int, ...],
          learning_rate: float = 0.001,
          layer_sizes: Optional[Tuple[int, ...]] = None,
          dropout_rate: float = 0.2,
          activation: str = "relu",
          use_batch_norm: bool = False,
          l2_reg: float = 0.0,
          optimizer_type: str = "adam",
          model_name: str = "dnn_model") -> keras.Model:
    """
    Build and compile a Deep Neural Network model for regression tasks.

    This function creates a flexible neural network architecture with configurable
    hyperparameters for regression problems. It supports multiple activation functions,
    regularization techniques, and optimizer types.

    Args:
        input_shape: Shape of the input data (features), e.g., (8,) for 8 features
        learning_rate: Learning rate for the optimizer (default: 0.001)
        layer_sizes: Tuple specifying the number of neurons in each hidden layer.
                   If None, uses default [64, 32, 16]. Example: (128, 64, 32)
        dropout_rate: Dropout rate for regularization (0.0-1.0, default: 0.2)
        activation: Activation function ('relu', 'selu', 'leaky_relu', 'elu', default: 'relu')
        use_batch_norm: Whether to use batch normalization (default: False)
        l2_reg: L2 regularization factor (0.0 for no regularization, default: 0.0)
        optimizer_type: Type of optimizer ('adam', 'rmsprop', 'nadam', 'sgd', default: 'adam')
        model_name: Name for the model (used for identification, default: 'dnn_model')

    Returns:
        Compiled Keras model ready for training

    Raises:
        ValueError: If model building fails due to invalid parameters

    Examples:
        >>> # Basic usage with default parameters
        >>> model = build(input_shape=(8,))
        >>> model.summary()

        >>> # Advanced usage with custom architecture
        >>> model = build(
        ...     input_shape=(8,),
        ...     learning_rate=0.0001,
        ...     layer_sizes=(128, 64, 32),
        ...     activation="selu",
        ...     use_batch_norm=True,
        ...     l2_reg=0.001,
        ...     optimizer_type="nadam"
        ... )
        >>> model.summary()

    Notes:
        - SELU activation uses 'lecun_normal' initializer and AlphaDropout
        - Other activations use 'he_normal' initializer and regular Dropout
        - Batch normalization is automatically disabled for SELU activation
        - L2 regularization is only applied if l2_reg > 0
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(42)

    # Validate and normalize parameters
    layer_sizes = _validate_build_parameters(
        input_shape, layer_sizes, dropout_rate, l2_reg, learning_rate
    )

    model = keras.models.Sequential()
    kernel_initializer = _get_kernel_initializer(activation)

    # Input layer
    model.add(keras.layers.Dense(
        layer_sizes[0],
        activation=activation,
        input_shape=input_shape,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    ))

    _add_regularization_layers(model, activation, dropout_rate, use_batch_norm)

    # Hidden layers
    for size in layer_sizes[1:]:
        model.add(keras.layers.Dense(
            size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
        ))

        _add_regularization_layers(model, activation, dropout_rate, use_batch_norm)

    # Output layer
    model.add(keras.layers.Dense(1))

    # Create optimizer and compile model
    try:
        optimizer = _get_optimizer(optimizer_type, learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae", "mse"]
        )

        logging.info(f"Successfully built model with {len(layer_sizes)} hidden layers")
        return model
    except Exception as e:
        logging.error(f"Error building model: {str(e)}")
        raise ValueError(f"Failed to build model: {str(e)}")