Starting CodeRabbit review in plain text mode...

Connecting to review service
Setting up
Analyzing
Reviewing

============================================================================
File: src/pipelines/evaluate.py
Line: 7
Type: potential_issue

Prompt for AI Agent:
In src/pipelines/evaluate.py around lines 7 and 60, the function return annotation List[float] is wrong because model.predict() yields a 2D array (batch_size, output_dim) and .tolist() produces List[List[float]]; either change the return type to List[List[float]] or flatten the prediction before returning (e.g., convert to 1D with .ravel()/.flatten() or a list comprehension) and update the annotation accordingly; apply the same fix at line 60 where the other return annotation appears.



============================================================================
File: test_code_quality.py
Line: 161 to 165
Type: potential_issue

Prompt for AI Agent:
In test_code_quality.py around lines 161 to 165, the test is named "train.py no duplicate seeds" but currently only asserts that tf.random.set_seed exists; either rename the test to reflect that it only checks presence, or change the implementation to detect duplicate occurrences of the pattern. To fix, replace the current lambda/usage with a function that reads pipelines/train.py, counts occurrences of the string "tf.random.set_seed" and fails if count > 1 (returning a clear message stating how many times the pattern appears), then use that function in the test entry so the test name matches the actual duplicate-detection behavior.



Review completed ✔
