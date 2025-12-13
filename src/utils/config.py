"""
Configuration module for centralized settings and constants.
"""

import os
from typing import Dict, Any
import json
import logging

class Config:
    """
    Centralized configuration manager for the ML pipeline.

    Attributes:
        DEFAULT_MODEL_DIR: Default directory for saving models
        DEFAULT_LOG_DIR: Default directory for saving logs
        DEFAULT_PLOT_DIR: Default directory for saving plots
        DEFAULT_TUNING_DIR: Default directory for hyperparameter tuning results
        DEFAULT_RANDOM_SEED: Default random seed for reproducibility
        DEFAULT_BATCH_SIZE: Default batch size for training and prediction
        DEFAULT_EPOCHS: Default number of training epochs
        DEFAULT_LEARNING_RATE: Default learning rate
        DEFAULT_LAYER_SIZES: Default layer sizes for neural networks
        DEFAULT_ACTIVATION: Default activation function
        DEFAULT_DROPOUT_RATE: Default dropout rate
        DEFAULT_L2_REG: Default L2 regularization factor
        DEFAULT_OPTIMIZER: Default optimizer type
    """

    # Default directories
    DEFAULT_MODEL_DIR = "models"
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_PLOT_DIR = "plots"
    DEFAULT_TUNING_DIR = "tuning_results"

    # Default training parameters
    DEFAULT_RANDOM_SEED = 42
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_EPOCHS = 100
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_LAYER_SIZES = (64, 32, 16)
    DEFAULT_ACTIVATION = "relu"
    DEFAULT_DROPOUT_RATE = 0.2
    DEFAULT_L2_REG = 0.0
    DEFAULT_OPTIMIZER = "adam"

    # Default evaluation parameters
    DEFAULT_EVAL_BATCH_SIZE = 32

    def __init__(self, config_file: str = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Optional path to JSON configuration file
        """
        self._config = self._get_default_config()

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration dictionary."""
        return {
            'model_dir': self.DEFAULT_MODEL_DIR,
            'log_dir': self.DEFAULT_LOG_DIR,
            'plot_dir': self.DEFAULT_PLOT_DIR,
            'tuning_dir': self.DEFAULT_TUNING_DIR,
            'random_seed': self.DEFAULT_RANDOM_SEED,
            'batch_size': self.DEFAULT_BATCH_SIZE,
            'epochs': self.DEFAULT_EPOCHS,
            'learning_rate': self.DEFAULT_LEARNING_RATE,
            'layer_sizes': self.DEFAULT_LAYER_SIZES,
            'activation': self.DEFAULT_ACTIVATION,
            'dropout_rate': self.DEFAULT_DROPOUT_RATE,
            'l2_reg': self.DEFAULT_L2_REG,
            'optimizer': self.DEFAULT_OPTIMIZER,
            'eval_batch_size': self.DEFAULT_EVAL_BATCH_SIZE
        }

    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from JSON file.

        Args:
            config_file: Path to JSON configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is not valid JSON
            ValueError: If config file has invalid structure
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)

            # Update configuration with file values
            for key, value in file_config.items():
                if key in self._config:
                    self._config[key] = value
                else:
                    logging.warning(f"Unknown configuration key: {key}")

        except FileNotFoundError:
            logging.error(f"Config file not found: {config_file}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in config file: {config_file}")
            raise
        except Exception as e:
            logging.error(f"Error loading config file: {str(e)}")
            raise ValueError(f"Failed to load config: {str(e)}")

    def save_to_file(self, config_file: str) -> None:
        """
        Save current configuration to JSON file.

        Args:
            config_file: Path to save configuration file

        Raises:
            IOError: If unable to write to file
        """
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving config file: {str(e)}")
            raise IOError(f"Failed to save config: {str(e)}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default if not found
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Value to set

        Raises:
            ValueError: If key is not valid
        """
        if key in self._config:
            self._config[key] = value
        else:
            raise ValueError(f"Invalid configuration key: {key}")

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.

        Returns:
            Dictionary of all configuration values
        """
        return self._config.copy()

    def create_directories(self) -> None:
        """
        Create all necessary directories based on current configuration.

        Creates:
            - Model directory
            - Log directory
            - Plot directory
            - Tuning directory
        """
        try:
            os.makedirs(self.get('model_dir'), exist_ok=True)
            os.makedirs(self.get('log_dir'), exist_ok=True)
            os.makedirs(self.get('plot_dir'), exist_ok=True)
            os.makedirs(self.get('tuning_dir'), exist_ok=True)
            logging.info("Created all necessary directories")
        except Exception as e:
            logging.error(f"Error creating directories: {str(e)}")
            raise IOError(f"Failed to create directories: {str(e)}")

# Global configuration instance
config = Config()

def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Global Config instance
    """
    return config