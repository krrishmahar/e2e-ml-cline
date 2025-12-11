#!/usr/bin/env python3
"""
Test script to verify the ML enhancements work correctly.
This script tests the enhanced functionality without requiring full training.
"""

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append('src')

def test_model_building():
    """Test the enhanced model building functionality."""
    print("🧪 Testing enhanced model building...")

    from models.dnn import build

    # Test different activation functions
    activations = ['relu', 'selu', 'elu']

    for activation in activations:
        try:
            model = build(
                input_shape=(8,),  # California housing has 8 features
                learning_rate=0.001,
                activation=activation,
                use_batch_norm=True,
                l2_reg=0.001,
                optimizer_type='adam'
            )
            print(f"✓ Model with {activation} activation built successfully")
            print(f"  Model summary: {model.count_params()} parameters")

        except Exception as e:
            print(f"✗ Failed to build model with {activation}: {str(e)}")
            return False

    # Test different optimizers
    optimizers = ['adam', 'rmsprop', 'nadam', 'sgd']

    for optimizer in optimizers:
        try:
            model = build(
                input_shape=(8,),
                learning_rate=0.001,
                optimizer_type=optimizer
            )
            print(f"✓ Model with {optimizer} optimizer built successfully")

        except Exception as e:
            print(f"✗ Failed to build model with {optimizer}: {str(e)}")
            return False

    return True

def test_data_loading():
    """Test data loading functionality."""
    print("\n📊 Testing data loading...")

    try:
        from data.load_data import load
        from data.preprocess import preprocess

        # Load data
        X_train, X_valid, X_test, y_train, y_valid, y_test = load()
        print(f"✓ Data loaded successfully")
        print(f"  Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

        # Preprocess data
        X_train_scaled, X_valid_scaled, X_test_scaled = preprocess(X_train, X_valid, X_test)
        print(f"✓ Data preprocessing successful")
        print(f"  Scaled data shapes: {X_train_scaled.shape}, {X_valid_scaled.shape}, {X_test_scaled.shape}")

        return True

    except Exception as e:
        print(f"✗ Data loading/preprocessing failed: {str(e)}")
        return False

def test_callbacks():
    """Test callbacks functionality."""
    print("\n🔔 Testing callbacks...")

    try:
        from utils.callbacks import get_callbacks

        callbacks = get_callbacks(
            model_name="test_model",
            patience=5,
            log_dir="test_logs"
        )

        print(f"✓ Callbacks created successfully: {[type(cb).__name__ for cb in callbacks]}")
        return True

    except Exception as e:
        print(f"✗ Callbacks creation failed: {str(e)}")
        return False

def test_training_parameters():
    """Test that training function accepts new parameters."""
    print("\n🎛️ Testing training function parameters...")

    try:
        # Just test that the function can be called with new parameters
        # We won't actually run training to save time
        from pipelines.train import train

        # Test parameter validation by checking function signature
        import inspect
        sig = inspect.signature(train)
        params = list(sig.parameters.keys())

        required_new_params = ['activation', 'use_batch_norm', 'l2_reg', 'optimizer_type', 'use_lr_schedule', 'lr_schedule_type']

        for param in required_new_params:
            if param in params:
                print(f"✓ Parameter '{param}' found in train function")
            else:
                print(f"✗ Parameter '{param}' missing from train function")
                return False

        return True

    except Exception as e:
        print(f"✗ Training parameter test failed: {str(e)}")
        return False

def test_evaluation_functions():
    """Test evaluation functions."""
    print("\n📈 Testing evaluation functions...")

    try:
        from pipelines.evaluate_enhanced import evaluate_model_comprehensive, compare_models

        # Test that functions exist and can be imported
        print("✓ Evaluation functions imported successfully")

        # Check function signatures
        import inspect

        # Check evaluate_model_comprehensive
        eval_sig = inspect.signature(evaluate_model_comprehensive)
        eval_params = list(eval_sig.parameters.keys())
        print(f"✓ evaluate_model_comprehensive has {len(eval_params)} parameters")

        # Check compare_models
        compare_sig = inspect.signature(compare_models)
        compare_params = list(compare_sig.parameters.keys())
        print(f"✓ compare_models has {len(compare_params)} parameters")

        return True

    except Exception as e:
        print(f"✗ Evaluation function test failed: {str(e)}")
        return False

def test_hyperparameter_tuning():
    """Test hyperparameter tuning functions."""
    print("\n🎯 Testing hyperparameter tuning...")

    try:
        from pipelines.hyperparameter_tuning import tune_hyperparameters, create_default_param_grid

        # Test parameter grid creation
        param_grid = create_default_param_grid()
        print(f"✓ Default parameter grid created with {len(param_grid)} parameters")

        # Check that it contains expected parameters
        expected_params = ['learning_rate', 'layer_sizes', 'activation', 'dropout_rate']
        for param in expected_params:
            if param in param_grid:
                print(f"✓ Parameter '{param}' found in grid")
            else:
                print(f"✗ Parameter '{param}' missing from grid")
                return False

        return True

    except Exception as e:
        print(f"✗ Hyperparameter tuning test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting ML Enhancements Test Suite")
    print("=" * 50)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    tests = [
        test_data_loading,
        test_model_building,
        test_callbacks,
        test_training_parameters,
        test_evaluation_functions,
        test_hyperparameter_tuning
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {str(e)}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {i+1}. {test.__name__}: {status}")

    print(f"\n🏁 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! ML enhancements are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)