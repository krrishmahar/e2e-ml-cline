# Cline Coding Rules


These rules guide Cline when analyzing, suggesting improvements, or refactoring code.


## 1. Code Quality
- Prefer clean, modular functions.
- Follow PEP8 for Python.
- Avoid large monolithic files.
- Always add docstrings to public functions.


## 2. ML Best Practices
- Use torch.utils.data for dataloading.
- Keep model architecture inside `models/`.
- Ensure reproducibility by setting random seeds.
- Logging should use `utils/logger.py`.


## 3. Cline Behavior
- Cline should provide diffs, not rewrite entire files unless necessary.
- Avoid changes to `data/` directory.
- Focus on correctness before optimization.


## 4. Performance
- Prefer vectorized operations (NumPy, Tensor ops).
- Avoid nested loops in Python unless necessary.
