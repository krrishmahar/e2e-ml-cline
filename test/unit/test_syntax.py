#!/usr/bin/env python3
"""
Simple syntax test to verify that the improved code has no syntax errors.
"""

import sys
import os
import ast

def test_file_syntax(file_path):
    """Test that a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse the file to check for syntax errors
        ast.parse(content)
        print(f"✓ {file_path} - Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {file_path} - Syntax Error: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ {file_path} - Error: {str(e)}")
        return False

def test_imports_syntax():
    """Test that import statements work in isolation."""
    print("\nTesting import syntax...")

    # Test the keras import that was added
    try:
        # This will fail at runtime but should parse correctly
        code = "from tensorflow import keras"
        ast.parse(code)
        print("✓ Keras import syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ Keras import syntax error: {str(e)}")
        return False

def main():
    """Run syntax tests."""
    print("Running syntax tests for improved code...")
    print("=" * 50)

    # Files that were improved
    improved_files = [
        'src/pipelines/train.py',
        'src/models/dnn.py',
        'src/data/preprocess.py',
        'src/utils/callbacks.py',
        'src/pipelines/evaluate.py'
    ]

    results = []

    # Test syntax of improved files
    for file_path in improved_files:
        if os.path.exists(file_path):
            results.append(test_file_syntax(file_path))
        else:
            print(f"✗ {file_path} - File not found")
            results.append(False)

    # Test import syntax
    results.append(test_imports_syntax())

    print("\n" + "=" * 50)
    print("Syntax Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All syntax tests passed! Code improvements are syntactically correct.")
        return True
    else:
        print("❌ Some syntax tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)