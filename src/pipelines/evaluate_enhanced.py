from tensorflow import keras
import logging
import os
import numpy as np
from typing import Union, List, Dict, Optional
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def evaluate_model_comprehensive(model: keras.Model,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               model_name: str = "model",
                               save_plots: bool = True,
                               plot_dir: str = "evaluation_plots") -> Dict[str, float]:
    """
    Comprehensive model evaluation with multiple metrics and visualizations.

    This function performs a thorough evaluation of a trained model, calculating multiple
    performance metrics and generating diagnostic visualizations. It supports batch processing
    for efficient prediction on large datasets.

    Args:
        model: Trained Keras model
        X_test: Test features (numpy array)
        y_test: True target values (numpy array)
        model_name: Name for the model (used in plots and logging, default: "model")
        save_plots: Whether to save evaluation plots (default: True)
        plot_dir: Directory to save plots (default: "evaluation_plots")

    Returns:
        Dictionary containing all evaluation metrics:
        - 'mae': Mean Absolute Error
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'r2': R-squared score
        - 'explained_variance': Explained Variance Score
        - 'mape': Mean Absolute Percentage Error

    Raises:
        ValueError: If evaluation fails due to input or prediction errors

    Examples:
        >>> # Basic usage
        >>> metrics = evaluate_model_comprehensive(model, X_test, y_test, "california_housing")
        >>> print(f"R2 Score: {metrics['r2']:.4f}")

        >>> # Custom directory
        >>> metrics = evaluate_model_comprehensive(
        ...     model, X_test, y_test, "experiment_1",
        ...     save_plots=True, plot_dir="custom_plots"
        ... )

    Notes:
        - Uses batch processing with default batch size of 32 for efficient prediction
        - Generates 4 diagnostic plots: Actual vs Predicted, Residual Plot, Error Distribution, Feature Importance
        - Saves metrics as CSV and plots as PNG files when save_plots=True
        - Feature importance is only available for models with accessible weights
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Create plot directory if it doesn't exist
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)

        logging.info(f"Starting comprehensive evaluation for {model_name}")

        # Make predictions with batch processing for better performance
        batch_size = 32
        y_pred = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size]
            batch_pred = model.predict(batch, verbose=0).flatten()
            y_pred.extend(batch_pred)
        y_pred = np.array(y_pred)

        # Calculate comprehensive metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'explained_variance': explained_variance_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
        }

        # Log all metrics
        logging.info(f"Evaluation Metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"  {metric_name.upper()}: {metric_value:.4f}")

        # Create visualizations
        plt.figure(figsize=(15, 10))

        # 1. Actual vs Predicted plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Actual vs Predicted')
        plt.grid(True)

        # 2. Residual plot
        plt.subplot(2, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} - Residual Plot')
        plt.grid(True)

        # 3. Error distribution
        plt.subplot(2, 2, 3)
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - Error Distribution')
        plt.grid(True)

        # 4. Feature importance (if model supports it)
        plt.subplot(2, 2, 4)
        try:
            # For simple dense models, we can extract weights
            if hasattr(model, 'layers') and len(model.layers) > 0:
                weights = []
                for layer in model.layers:
                    if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                        layer_weights = layer.get_weights()[0]
                        if len(layer_weights.shape) == 2:  # Dense layer weights
                            # Average absolute weights for each input feature
                            avg_weights = np.mean(np.abs(layer_weights), axis=1)
                            weights.extend(avg_weights)

                if weights:
                    feature_names = [f"Feature {i+1}" for i in range(len(weights))]
                    plt.bar(range(len(weights)), weights)
                    plt.xticks(range(len(weights)), feature_names, rotation=45)
                    plt.ylabel('Average Absolute Weight')
                    plt.title(f'{model_name} - Feature Importance')
                    plt.grid(True, axis='y')
        except Exception as e:
            plt.text(0.5, 0.5, 'Feature importance\nnot available',
                    ha='center', va='center')
            plt.title(f'{model_name} - Feature Importance')

        plt.tight_layout()

        # Save or show plot
        if save_plots:
            plot_path = os.path.join(plot_dir, f"{model_name}_evaluation.png")
            plt.savefig(plot_path)
            logging.info(f"Saved evaluation plot to {plot_path}")
        else:
            plt.show()

        plt.close()

        # Create metrics DataFrame and save as CSV
        metrics_df = pd.DataFrame([metrics])
        if save_plots:
            csv_path = os.path.join(plot_dir, f"{model_name}_metrics.csv")
            metrics_df.to_csv(csv_path, index=False)
            logging.info(f"Saved metrics to {csv_path}")

        logging.info(f"Comprehensive evaluation completed for {model_name}")
        return metrics

    except Exception as e:
        logging.error(f"Error during comprehensive evaluation: {str(e)}")
        raise ValueError(f"Failed to evaluate model: {str(e)}")

def compare_models(models: Dict[str, keras.Model],
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  plot_dir: str = "model_comparison") -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Args:
        models: Dictionary of model names to trained models
        X_test: Test features
        y_test: True target values
        plot_dir: Directory to save comparison plots

    Returns:
        DataFrame with comparison metrics for all models
    """
    try:
        os.makedirs(plot_dir, exist_ok=True)

        comparison_results = []

        plt.figure(figsize=(15, 8))

        for i, (model_name, model) in enumerate(models.items(), 1):
            # Evaluate model
            metrics = evaluate_model_comprehensive(
                model, X_test, y_test,
                model_name=model_name,
                save_plots=False
            )

            # Add to comparison results
            metrics['model_name'] = model_name
            comparison_results.append(metrics)

            # Plot actual vs predicted for each model with batch processing
            batch_size = 32
            y_pred = []
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i + batch_size]
                batch_pred = model.predict(batch, verbose=0).flatten()
                y_pred.extend(batch_pred)
            y_pred = np.array(y_pred)
            plt.subplot(2, 3, i)
            plt.scatter(y_test, y_pred, alpha=0.5, label=model_name)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{model_name} (R2: {metrics["r2"]:.3f})')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        comparison_plot_path = os.path.join(plot_dir, "model_comparison.png")
        plt.savefig(comparison_plot_path)
        plt.close()

        logging.info(f"Saved model comparison plot to {comparison_plot_path}")

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_csv_path = os.path.join(plot_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_csv_path, index=False)

        logging.info(f"Saved model comparison metrics to {comparison_csv_path}")

        return comparison_df

    except Exception as e:
        logging.error(f"Error during model comparison: {str(e)}")
        raise ValueError(f"Failed to compare models: {str(e)}")

if __name__ == "__main__":
    # Example usage
    from src.pipelines.train import train
    from src.data.load_data import load
    from src.data.preprocess import preprocess

    # Load and preprocess data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load()
    X_train, X_valid, X_test = preprocess(X_train, X_valid, X_test)

    # Train a model
    model = train(epochs=10, batch_size=32, learning_rate=0.001, dry_run=False)

    # Evaluate the model
    metrics = evaluate_model_comprehensive(model, X_test, y_test, "california_housing")

    print("Evaluation completed successfully!")
    print(f"R2 Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")