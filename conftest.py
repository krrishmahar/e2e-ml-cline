import glob
import os
import pytest

# Provide all python files in src/ as file paths
@pytest.fixture(params=glob.glob("src/**/*.py", recursive=True))
def file_path(request):
    return request.param

# Provide sample function names to check
@pytest.fixture(params=["train", "evaluate", "build_model"])
def function_name(request):
    return request.param

# Require certain imports to exist
@pytest.fixture
def required_imports():
    return ["logging", "tensorflow", "numpy"]

