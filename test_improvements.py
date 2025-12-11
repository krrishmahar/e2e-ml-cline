#!/usr/bin/env python3
"""
Test script to verify the code improvements work correctly.
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dnn_import():
    """Test that dnn.py can be imported without errors."""
    try:
        from models.dnn import build
        print("✓ dnn.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ dnn.py import failed: {e}")
        return False

def test_evaluate_import():
    """Test that evaluate.py can be imported without errors."""
    try:
        from pipelines.evaluate import predict
        print("✓ evaluate.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ evaluate.py import failed: {e}")
        return False

def test_evaluate_functionality():
    """Test evaluate.py functionality with mock data."""
    try:
        from pipelines.evaluate import predict

        # Test with invalid input
        try:
            predict(None)
            print("✗ predict() should raise ValueError for None input")
            return False
        except ValueError:
            print("✓ predict() correctly validates None input")

        # Test with empty input
        try:
            predict([])
            print("✗ predict() should raise ValueError for empty input")
            return False
        except ValueError:
            print("✓ predict() correctly validates empty input")

        # Test with valid input (should fail on missing model file, which is expected)
        try:
            predict([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
            print("✗ predict() should raise FileNotFoundError for missing model")
            return False
        except FileNotFoundError:
            print("✓ predict() correctly handles missing model file")
            return True
        except Exception as e:
            print(f"✗ predict() unexpected error: {e}")
            return False

    except Exception as e:
        print(f"✗ evaluate.py functionality test failed: {e}")
        return False

def test_train_import():
    """Test that train.py can be imported without errors."""
    try:
        from pipelines.train import train
        print("✓ train.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ train.py import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing code improvements...")
    print("=" * 50)

    tests = [
        test_dnn_import,
        test_evaluate_import,
        test_evaluate_functionality,
        test_train_import
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Code improvements are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)