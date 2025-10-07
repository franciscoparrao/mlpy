"""Test simple de imports de MLPY"""

print("Testing imports...")

# Test 1: Imports básicos
try:
    from mlpy.tasks import TaskClassif, TaskRegr
    print("1. Tasks: OK")
except Exception as e:
    print(f"1. Tasks: FALLO - {e}")

# Test 2: Learners
try:
    from mlpy.learners.sklearn import learner_sklearn
    print("2. Sklearn wrapper: OK")
except Exception as e:
    print(f"2. Sklearn wrapper: FALLO - {e}")

# Test 3: Measures
try:
    from mlpy.measures.classification import MeasureClassifAcc
    from mlpy.measures.regression import MeasureRegrRMSE
    print("3. Measures: OK")
except Exception as e:
    print(f"3. Measures: FALLO - {e}")

# Test 4: Core functions
try:
    from mlpy import resample, benchmark
    print("4. Core functions: OK")
except Exception as e:
    print(f"4. Core functions: FALLO - {e}")

# Test 5: Pipelines
try:
    from mlpy.pipelines import PipeOpScale, linear_pipeline
    print("5. Pipelines: OK")
except Exception as e:
    print(f"5. Pipelines: FALLO - {e}")

print("\nDone!")