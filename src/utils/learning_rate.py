"""
Learning rate scheduling utilities for model training.

This module provides functions for creating and managing learning rate schedules
to optimize model training performance.
"""

from tensorflow import keras
from typing import Optional, Union
import logging

def create_learning_rate_schedule(schedule_type: str = "cosine",
                                 initial_learning_rate: float = 0.001,
                                 decay_steps: int = 1000,
                                 decay_rate: float = 0.96,
                                 staircase: bool = True) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Create a learning rate schedule based on the specified type.

    Args:
        schedule_type: Type of learning rate schedule ('cosine', 'exponential', 'piecewise')
        initial_learning_rate: Initial learning rate
        decay_steps: Number of steps for decay
        decay_rate: Decay rate for exponential schedule
        staircase: Whether to apply decay in discrete intervals

    Returns:
        Learning rate schedule object

    Raises:
        ValueError: If schedule_type is invalid
    """
    schedule_type = schedule_type.lower()

    try:
        if schedule_type == "cosine":
            return keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps
            )
        elif schedule_type == "exponential":
            return keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=staircase
            )
        elif schedule_type == "piecewise":
            # Define piecewise constant decay boundaries and values
            boundaries = [decay_steps // 2, decay_steps * 3 // 4, decay_steps]
            values = [
                initial_learning_rate,
                initial_learning_rate * 0.5,
                initial_learning_rate * 0.1,
                initial_learning_rate * 0.01
            ]
            return keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=boundaries,
                values=values
            )
        else:
            logging.warning(f"Unknown schedule type '{schedule_type}', defaulting to cosine decay")
            return keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps
            )
    except Exception as e:
        logging.error(f"Error creating learning rate schedule: {str(e)}")
        raise ValueError(f"Failed to create learning rate schedule: {str(e)}")

def apply_learning_rate_schedule(model: keras.Model,
                               optimizer_type: str,
                               lr_schedule: keras.optimizers.schedules.LearningRateSchedule) -> None:
    """
    Apply learning rate schedule to the model's optimizer.

    Args:
        model: Keras model
        optimizer_type: Type of optimizer
        lr_schedule: Learning rate schedule to apply

    Raises:
        ValueError: If optimizer type is invalid or schedule application fails
    """
    try:
        optimizer = model.optimizer
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate = lr_schedule
        else:
            # Create new optimizer with learning rate schedule
            optimizer_map = {
                "adam": keras.optimizers.Adam,
                "rmsprop": keras.optimizers.RMSprop,
                "nadam": keras.optimizers.Nadam,
                "sgd": lambda lr: keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            }

            optimizer_factory = optimizer_map.get(optimizer_type.lower(), keras.optimizers.Adam)
            new_optimizer = optimizer_factory(learning_rate=lr_schedule)

            model.compile(
                optimizer=new_optimizer,
                loss="mse",
                metrics=["mae", "mse"]
            )

        logging.info(f"Successfully applied learning rate schedule to {optimizer_type} optimizer")

    except Exception as e:
        logging.error(f"Error applying learning rate schedule: {str(e)}")
        raise ValueError(f"Failed to apply learning rate schedule: {str(e)}")

def get_learning_rate_schedule_config(epochs: int,
                                    batch_size: int,
                                    dataset_size: int,
                                    initial_lr: float = 0.001,
                                    schedule_type: str = "cosine") -> dict:
    """
    Get recommended learning rate schedule configuration based on training parameters.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        dataset_size: Size of training dataset
        initial_lr: Initial learning rate
        schedule_type: Type of learning rate schedule

    Returns:
        Dictionary containing schedule configuration
    """
    steps_per_epoch = dataset_size // batch_size
    total_steps = epochs * steps_per_epoch

    return {
        "schedule_type": schedule_type,
        "initial_learning_rate": initial_lr,
        "decay_steps": total_steps,
        "decay_rate": 0.96 if schedule_type == "exponential" else None,
        "staircase": True if schedule_type == "exponential" else None
    }