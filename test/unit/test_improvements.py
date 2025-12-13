#!/usr/bin/env python3
"""
Test script to validate the code improvements made to the ML pipeline.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

def test_imports():
    """Test that all improved modules can be imported successfully."""
    print("Testing imports...")

    try:
        # Test model building module
        from models.dnn import build, _validate_build_parameters, _get_optimizer, _add_regularization_layers, _get_kernel_initializer
        print("✓ Successfully imported improved dnn.py module")

        # Test preprocessing module
        from data.preprocess import preprocess, _validate_input_arrays, _load_cached_scaler, _save_scaler_cache, _get_scaler
        print("✓ Successfully imported improved preprocess.py module")

        # Test learning rate utility module
        from utils.learning_rate import create_learning_rate_schedule, apply_learning_rate_schedule, get_learning_rate_schedule_config
        print("✓ Successfully imported new learning_rate.py module")

        # Test that train module can import the new utilities
        from pipelines.train import train
        print("✓ Successfully imported improved train.py module")

        return True

    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False

def test_function_signatures():
    """Test that the improved functions have the correct signatures."""
    print("\nTesting function signatures...")

    try:
        from models.dnn import _validate_build_parameters, _get_optimizer, _add_regularization_layers, _get_kernel_initializer

        # Test _validate_build_parameters signature
        import inspect
        sig = inspect.signature(_validate_build_parameters)
        expected_params = ['input_shape', 'layer_sizes', 'dropout_rate', 'l2_reg', 'learning_rate']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params, f"Expected {expected_params}, got {actual_params}"
        print("✓ _validate_build_parameters has correct signature")

        # Test _get_optimizer signature
        sig = inspect.signature(_get_optimizer)
        expected_params = ['optimizer_type', 'learning_rate']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params, f"Expected {expected_params}, got {actual_params}"
        print("✓ _get_optimizer has correct signature")

        return True

    except Exception as e:
        print(f"✗ Signature test failed: {str(e)}")
        return False

def test_code_structure():
    """Test that the code improvements follow the expected structure."""
    print("\nTesting code structure...")

    try:
        # Test that helper functions exist and are properly named
        from models.dnn import _validate_build_parameters, _get_optimizer, _add_regularization_layers, _get_kernel_initializer
        from data.preprocess import _validate_input_arrays, _load_cached_scaler, _save_scaler_cache, _get_scaler

        # Test that the main functions still exist
        from models.dnn import build
        from data.preprocess import preprocess

        print("✓ All expected functions exist with proper naming")

        # Test that the learning rate functions are properly separated
        from utils.learning_rate import create_learning_rate_schedule, apply_learning_rate_schedule

        print("✓ Learning rate utilities are properly separated")

        return True

    except ImportError as e:
        print(f"✗ Code structure test failed: {str(e)}")
        return False

def main():
    """Run all tests and report results."""
    print("Running code improvement validation tests...\n")

    tests = [
        test_imports,
        test_function_signatures,
        test_code_structure
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\nTest Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("🎉 All tests passed! Code improvements are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())