"""
Example demonstrating MLPY's improved validation system.

Shows how the new ValidatedTask provides clear, helpful error messages
instead of cryptic AttributeErrors.
"""

import pandas as pd
import numpy as np
from mlpy import ValidatedTask, validate_task_data

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.normal(0, 1, 100),
    'target': np.random.choice(['A', 'B'], 100)
})

print("=== MLPY Validation System Example ===\n")

# 1. Successful task creation
print("1. Creating a valid task:")
try:
    task = ValidatedTask(
        id='example_task',
        task_type='classif', 
        backend=data,
        target='target'
    )
    print("Task created successfully!")
    print(f"   Task type: {task.task_type}")
    print(f"   Data shape: {task.backend.shape}")
    print(f"   Target: {task.target}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60 + "\n")

# 2. Demonstrate helpful error messages
print("2. Example of helpful error message:")
print("   Trying to create task with non-existent target column...")

try:
    bad_task = ValidatedTask(
        id='bad_task',
        task_type='classif',
        backend=data,
        target='nonexistent_column'  # This column doesn't exist
    )
except Exception as e:
    print("ERROR: Got helpful error message:")
    print(f"   {str(e)[:100]}...")
    print("   (Much better than 'AttributeError: NoneType object...')")

print("\n" + "="*60 + "\n")

# 3. Data validation
print("3. Data validation report:")
validation_result = validate_task_data(data, target='target')

print(f"SUCCESS: Valid: {validation_result['valid']}")
if validation_result['warnings']:
    print(f"WARNING:  Warnings: {validation_result['warnings']}")
if validation_result['suggestions']:
    print(f"SUGGESTION: Suggestions: {validation_result['suggestions']}")

print("\n" + "="*60 + "\n")

# 4. Spatial task example
print("4. Spatial task validation:")
spatial_data = data.copy()
spatial_data['x'] = np.random.uniform(0, 10, 100)
spatial_data['y'] = np.random.uniform(0, 10, 100)

try:
    spatial_task = ValidatedTask(
        id='spatial_task',
        task_type='classif_spatial',
        backend=spatial_data,
        target='target',
        coordinate_names=['x', 'y']
    )
    print("SUCCESS: Spatial task created successfully!")
    print(f"   Coordinates: {spatial_task.coordinate_names}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60 + "\n")

# 5. Missing coordinates error
print("5. Missing coordinates validation:")
try:
    bad_spatial = ValidatedTask(
        id='bad_spatial',
        task_type='classif_spatial',
        backend=data,  # No coordinate columns
        target='target',
        coordinate_names=['missing_x', 'missing_y']
    )
except Exception as e:
    print("ERROR: Got helpful spatial error:")
    print(f"   {str(e)[:150]}...")

print("\n" + "="*60 + "\n")
print("SUMMARY: Summary: MLPY's validation system provides:")
print("   • Clear, actionable error messages")
print("   • Automatic data validation")
print("   • Context-aware suggestions")
print("   • Prevention of common mistakes")
print("   • Better developer experience")