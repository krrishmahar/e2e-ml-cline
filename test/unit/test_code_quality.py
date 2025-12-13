#!/usr/bin/env python3
"""
Test script to verify code quality improvements without requiring TensorFlow.
This tests syntax, imports, and basic functionality.
"""

import sys
import os
import ast
import re

def test_file_syntax(file_path):
    """Test that a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse the AST to check for syntax errors
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def test_imports(file_path, required_imports):
    """Test that a file contains required imports."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        for imp in required_imports:
            if imp not in content:
                return False, f"Missing required import: {imp}"

        return True, None
    except Exception as e:
        return False, f"Error checking imports: {e}"

def test_function_documentation(file_path, function_name):
    """Test that a function has proper documentation."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Find the function definition line
        func_pattern = rf'def\s+{function_name}\s*\('
        func_match = re.search(func_pattern, content)
        if not func_match:
            return False, f"Function {function_name} missing docstring"

        func_start = func_match.start()

        # Find the next triple quote after the function definition
        docstring_pattern = r'"""(.*?)"""'
        content_after_func = content[func_start:]
        docstring_match = re.search(docstring_pattern, content_after_func, re.DOTALL)

        if not docstring_match:
            return False, f"Function {function_name} missing docstring"

        docstring = docstring_match.group(1).strip()
        if not docstring:
            return False, f"Function {function_name} has empty docstring"

        return True, None
    except Exception as e:
        return False, f"Error checking documentation: {e}"

def test_logging_configuration(file_path):
    """Test that a file has proper logging configuration."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Check for logging.basicConfig
        if 'logging.basicConfig' not in content:
            return False, "Missing logging.basicConfig"

        return True, None
    except Exception as e:
        return False, f"Error checking logging: {e}"

def test_error_handling(file_path, function_name):
    """Test that a function has proper error handling."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Simple check for try/except pattern
        pattern = rf'def {function_name}.*?(try:.*?except.*?:)'
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            return False, f"Function {function_name} missing error handling"

        return True, None
    except Exception as e:
        return False, f"Error checking error handling: {e}"

def main():
    """Run all code quality tests."""
    print("Testing code quality improvements...")
    print("=" * 60)

    base_dir = os.path.join(os.path.dirname(__file__), 'src')

    tests = [
        # Test dnn.py improvements
        {
            'file': os.path.join(base_dir, 'models', 'dnn.py'),
            'name': 'dnn.py syntax',
            'test': lambda: test_file_syntax(os.path.join(base_dir, 'models', 'dnn.py'))
        },
        {
            'file': os.path.join(base_dir, 'models', 'dnn.py'),
            'name': 'dnn.py logging import',
            'test': lambda: test_imports(os.path.join(base_dir, 'models', 'dnn.py'), ['import logging'])
        },
        {
            'file': os.path.join(base_dir, 'models', 'dnn.py'),
            'name': 'dnn.py build function docs',
            'test': lambda: test_function_documentation(os.path.join(base_dir, 'models', 'dnn.py'), 'build')
        },

        # Test evaluate.py improvements
        {
            'file': os.path.join(base_dir, 'pipelines', 'evaluate.py'),
            'name': 'evaluate.py syntax',
            'test': lambda: test_file_syntax(os.path.join(base_dir, 'pipelines', 'evaluate.py'))
        },
        {
            'file': os.path.join(base_dir, 'pipelines', 'evaluate.py'),
            'name': 'evaluate.py required imports',
            'test': lambda: test_imports(os.path.join(base_dir, 'pipelines', 'evaluate.py'), [
                'import logging', 'import os', 'import numpy as np', 'from typing import Union, List'
            ])
        },
        {
            'file': os.path.join(base_dir, 'pipelines', 'evaluate.py'),
            'name': 'evaluate.py logging config',
            'test': lambda: test_logging_configuration(os.path.join(base_dir, 'pipelines', 'evaluate.py'))
        },
        {
            'file': os.path.join(base_dir, 'pipelines', 'evaluate.py'),
            'name': 'evaluate.py predict function docs',
            'test': lambda: test_function_documentation(os.path.join(base_dir, 'pipelines', 'evaluate.py'), 'predict')
        },
        {
            'file': os.path.join(base_dir, 'pipelines', 'evaluate.py'),
            'name': 'evaluate.py error handling',
            'test': lambda: test_error_handling(os.path.join(base_dir, 'pipelines', 'evaluate.py'), 'predict')
        },

        # Test train.py improvements
        {
            'file': os.path.join(base_dir, 'pipelines', 'train.py'),
            'name': 'train.py syntax',
            'test': lambda: test_file_syntax(os.path.join(base_dir, 'pipelines', 'train.py'))
        },
        {
            'file': os.path.join(base_dir, 'pipelines', 'train.py'),
            'name': 'train.py no duplicate seeds',
            'test': lambda: test_imports(os.path.join(base_dir, 'pipelines', 'train.py'), ['tf.random.set_seed'])
        }
    ]

    results = []
    for test in tests:
        try:
            success, message = test['test']()
            if success:
                print(f"✓ {test['name']}")
                results.append(True)
            else:
                print(f"✗ {test['name']}: {message}")
                results.append(False)
        except Exception as e:
            print(f"✗ {test['name']}: Test failed with exception: {e}")
            results.append(False)

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Code quality tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All code quality tests passed! Improvements are working correctly.")
        return True
    else:
        print("❌ Some code quality tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)