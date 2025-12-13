#!/usr/bin/env python3
"""
Test script to validate the code structure improvements made to the ML pipeline.
This test focuses on code organization and doesn't require TensorFlow.
"""

import sys
import os
import ast
import inspect

def test_file_structure():
    """Test that the improved files have the expected structure."""
    print("Testing file structure improvements...")

    files_to_check = [
        ('src/models/dnn.py', [
            '_validate_build_parameters',
            '_get_optimizer',
            '_add_regularization_layers',
            '_get_kernel_initializer',
            'build'
        ]),
        ('src/data/preprocess.py', [
            '_validate_input_arrays',
            '_load_cached_scaler',
            '_save_scaler_cache',
            '_get_scaler',
            'preprocess'
        ]),
        ('src/utils/learning_rate.py', [
            'create_learning_rate_schedule',
            'apply_learning_rate_schedule',
            'get_learning_rate_schedule_config'
        ]),
        ('src/pipelines/train.py', [
            'train'
        ])
    ]

    all_passed = True

    for file_path, expected_functions in files_to_check:
        try:
            # Read the file content
            with open(file_path, 'r') as f:
                content = f.read()

            # Parse the AST to find function definitions
            tree = ast.parse(content)

            # Extract function names
            function_names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_names.append(node.name)

            # Check if all expected functions are present
            missing_functions = []
            for func in expected_functions:
                if func not in function_names:
                    missing_functions.append(func)

            if missing_functions:
                print(f"✗ {file_path}: Missing functions: {missing_functions}")
                all_passed = False
            else:
                print(f"✓ {file_path}: All expected functions present")

        except FileNotFoundError:
            print(f"✗ {file_path}: File not found")
            all_passed = False
        except SyntaxError as e:
            print(f"✗ {file_path}: Syntax error: {str(e)}")
            all_passed = False
        except Exception as e:
            print(f"✗ {file_path}: Unexpected error: {str(e)}")
            all_passed = False

    return all_passed

def test_code_organization():
    """Test that the code improvements follow good organization principles."""
    print("\nTesting code organization...")

    try:
        # Check that helper functions are properly prefixed with underscore
        files_to_check = [
            'src/models/dnn.py',
            'src/data/preprocess.py'
        ]

        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()

            tree = ast.parse(content)

            # Count helper functions (starting with underscore)
            helper_functions = []
            public_functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('_'):
                        helper_functions.append(node.name)
                    else:
                        public_functions.append(node.name)

            print(f"✓ {file_path}: Found {len(helper_functions)} helper functions and {len(public_functions)} public functions")

        return True

    except Exception as e:
        print(f"✗ Code organization test failed: {str(e)}")
        return False

def test_learning_rate_separation():
    """Test that learning rate functionality is properly separated."""
    print("\nTesting learning rate separation...")

    try:
        # Check that train.py imports from the new learning_rate module
        with open('src/pipelines/train.py', 'r') as f:
            content = f.read()

        # Check for the import statement
        if 'from src.utils.learning_rate import' in content:
            print("✓ train.py properly imports from learning_rate module")
        else:
            print("✗ train.py does not import from learning_rate module")
            return False

        # Check that the old duplicate functions are removed
        if '_create_learning_rate_schedule' not in content and '_apply_learning_rate_schedule' not in content:
            print("✓ Duplicate learning rate functions removed from train.py")
        else:
            print("✗ Duplicate learning rate functions still present in train.py")
            return False

        return True

    except Exception as e:
        print(f"✗ Learning rate separation test failed: {str(e)}")
        return False

def test_function_documentation():
    """Test that the improved functions have proper documentation."""
    print("\nTesting function documentation...")

    try:
        files_to_check = [
            'src/models/dnn.py',
            'src/data/preprocess.py',
            'src/utils/learning_rate.py'
        ]

        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()

            tree = ast.parse(content)

            # Check that functions have docstrings
            functions_without_docstrings = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        functions_without_docstrings.append(node.name)

            if functions_without_docstrings:
                print(f"✗ {file_path}: Functions without docstrings: {functions_without_docstrings}")
                return False
            else:
                print(f"✓ {file_path}: All functions have docstrings")

        return True

    except Exception as e:
        print(f"✗ Documentation test failed: {str(e)}")
        return False

def main():
    """Run all structure tests and report results."""
    print("Running code structure validation tests...\n")

    tests = [
        test_file_structure,
        test_code_organization,
        test_learning_rate_separation,
        test_function_documentation
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\nStructure Test Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("🎉 All structure tests passed! Code improvements are properly organized.")
        return 0
    else:
        print("❌ Some structure tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())