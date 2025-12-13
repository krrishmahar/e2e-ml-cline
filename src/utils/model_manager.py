"""
Model management utility for saving, loading, and versioning ML models.
"""

import os
import logging
from typing import Dict, List, Optional, Union
from tensorflow import keras
import json
import datetime
import hashlib

class ModelManager:
    """
    Model management utility for handling model persistence, versioning, and metadata.

    Features:
        - Model saving with versioning
        - Model loading with version selection
        - Model metadata tracking
        - Model history and provenance
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize ModelManager.

        Args:
            model_dir: Directory to store models (default: "models")
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Create metadata directory
        self.metadata_dir = os.path.join(self.model_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def save_model(self,
                  model: keras.Model,
                  model_name: str = "model",
                  version: Optional[str] = None,
                  metadata: Optional[Dict] = None,
                  overwrite: bool = False) -> str:
        """
        Save a model with versioning and metadata.

        Args:
            model: Keras model to save
            model_name: Base name for the model
            version: Optional version identifier (default: timestamp)
            metadata: Optional dictionary of metadata
            overwrite: Whether to overwrite existing model

        Returns:
            Path to the saved model file

        Raises:
            IOError: If model saving fails
        """
        try:
            # Generate version if not provided
            if version is None:
                version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create model-specific directory
            model_version_dir = os.path.join(self.model_dir, f"{model_name}_{version}")
            os.makedirs(model_version_dir, exist_ok=True)

            # Save model using Keras format (not HDF5)
            model_path = os.path.join(model_version_dir, f"{model_name}.keras")
            model.save(model_path)

            # Save metadata
            if metadata is None:
                metadata = {}

            # Add automatic metadata
            metadata.update({
                'model_name': model_name,
                'version': version,
                'save_timestamp': datetime.datetime.now().isoformat(),
                'model_type': str(type(model)),
                'model_summary': self._get_model_summary(model)
            })

            metadata_path = os.path.join(model_version_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Update model registry
            self._update_model_registry(model_name, version, metadata)

            logging.info(f"Saved model {model_name} version {version} to {model_path}")
            return model_path

        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise IOError(f"Failed to save model: {str(e)}")

    def load_model(self,
                  model_name: str,
                  version: Optional[str] = None,
                  return_metadata: bool = False) -> Union[keras.Model, tuple]:
        """
        Load a model by name and optional version.

        Args:
            model_name: Name of the model to load
            version: Optional version identifier (default: latest)
            return_metadata: Whether to return metadata along with model

        Returns:
            Loaded Keras model, or tuple of (model, metadata) if return_metadata=True

        Raises:
            FileNotFoundError: If model not found
            IOError: If model loading fails
        """
        try:
            # Get available versions
            versions = self.get_model_versions(model_name)
            if not versions:
                raise FileNotFoundError(f"No models found with name: {model_name}")

            # Select version
            if version is None:
                version = versions[-1]  # Use latest version
            elif version not in versions:
                raise FileNotFoundError(f"Version {version} not found for model {model_name}")

            # Load model
            model_path = os.path.join(self.model_dir, f"{model_name}_{version}", f"{model_name}.keras")
            model = keras.models.load_model(model_path)

            # Load metadata
            metadata_path = os.path.join(self.model_dir, f"{model_name}_{version}", "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            logging.info(f"Loaded model {model_name} version {version} from {model_path}")

            if return_metadata:
                return model, metadata
            return model

        except FileNotFoundError:
            logging.error(f"Model {model_name} version {version} not found")
            raise
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise IOError(f"Failed to load model: {str(e)}")

    def get_model_versions(self, model_name: str) -> List[str]:
        """
        Get available versions for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of available versions, sorted chronologically
        """
        versions = []
        prefix = f"{model_name}_"

        try:
            for item in os.listdir(self.model_dir):
                if item.startswith(prefix) and os.path.isdir(os.path.join(self.model_dir, item)):
                    version = item[len(prefix):]
                    versions.append(version)

            # Sort versions chronologically if they are timestamps
            versions.sort()
            return versions

        except Exception as e:
            logging.error(f"Error getting model versions: {str(e)}")
            return []

    def get_latest_model(self,
                        model_name: str,
                        return_metadata: bool = False) -> Union[keras.Model, tuple]:
        """
        Get the latest version of a model.

        Args:
            model_name: Name of the model
            return_metadata: Whether to return metadata along with model

        Returns:
            Latest version of the model, or tuple of (model, metadata) if return_metadata=True

        Raises:
            FileNotFoundError: If no models found
        """
        versions = self.get_model_versions(model_name)
        if not versions:
            raise FileNotFoundError(f"No models found with name: {model_name}")

        return self.load_model(model_name, versions[-1], return_metadata)

    def delete_model(self, model_name: str, version: str) -> bool:
        """
        Delete a specific model version.

        Args:
            model_name: Name of the model
            version: Version to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            model_dir = os.path.join(self.model_dir, f"{model_name}_{version}")
            if os.path.exists(model_dir):
                import shutil
                shutil.rmtree(model_dir)
                self._update_model_registry(model_name, version, None)  # Remove from registry
                logging.info(f"Deleted model {model_name} version {version}")
                return True
            return False
        except Exception as e:
            logging.error(f"Error deleting model: {str(e)}")
            return False

    def _get_model_summary(self, model: keras.Model) -> str:
        """Get model summary as string."""
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()

    def _update_model_registry(self, model_name: str, version: str, metadata: Optional[Dict]) -> None:
        """Update the model registry file."""
        try:
            registry_path = os.path.join(self.metadata_dir, "model_registry.json")

            # Load existing registry
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {}

            # Update registry
            if model_name not in registry:
                registry[model_name] = {}

            if metadata is None:
                # Remove version from registry
                if version in registry[model_name]:
                    del registry[model_name][version]
            else:
                # Add/update version in registry
                registry[model_name][version] = {
                    'timestamp': metadata.get('save_timestamp', datetime.datetime.now().isoformat()),
                    'metadata_path': os.path.join(self.model_dir, f"{model_name}_{version}", "metadata.json")
                }

            # Save registry
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)

        except Exception as e:
            logging.error(f"Error updating model registry: {str(e)}")

    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict:
        """
        Get information about a model.

        Args:
            model_name: Name of the model
            version: Optional version (default: latest)

        Returns:
            Dictionary containing model information

        Raises:
            FileNotFoundError: If model not found
        """
        try:
            if version is None:
                versions = self.get_model_versions(model_name)
                if not versions:
                    raise FileNotFoundError(f"No models found with name: {model_name}")
                version = versions[-1]

            metadata_path = os.path.join(self.model_dir, f"{model_name}_{version}", "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return {
                'model_name': model_name,
                'version': version,
                'save_timestamp': metadata.get('save_timestamp'),
                'model_type': metadata.get('model_type'),
                'additional_metadata': {k: v for k, v in metadata.items()
                                      if k not in ['model_name', 'version', 'save_timestamp', 'model_type', 'model_summary']}
            }

        except FileNotFoundError:
            logging.error(f"Model metadata not found: {model_name} version {version}")
            raise
        except Exception as e:
            logging.error(f"Error getting model info: {str(e)}")
            raise IOError(f"Failed to get model info: {str(e)}")

    def list_models(self) -> Dict[str, List[str]]:
        """
        List all available models and their versions.

        Returns:
            Dictionary mapping model names to lists of versions
        """
        try:
            models = {}
            for item in os.listdir(self.model_dir):
                if os.path.isdir(os.path.join(self.model_dir, item)) and "_" in item:
                    parts = item.split("_", 1)
                    if len(parts) == 2:
                        model_name, version = parts
                        if model_name not in models:
                            models[model_name] = []
                        models[model_name].append(version)

            # Sort versions for each model
            for model_name in models:
                models[model_name].sort()

            return models

        except Exception as e:
            logging.error(f"Error listing models: {str(e)}")
            return {}

# Global model manager instance
model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    """
    Get the global model manager instance.

    Returns:
        Global ModelManager instance
    """
    return model_manager