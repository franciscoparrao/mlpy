"""Test MLPY in Anaconda environment"""

import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import mlpy
    print(f"\n[OK] MLPY imported successfully!")
    print(f"Version: {mlpy.__version__}")
    print(f"Location: {mlpy.__file__}")
    
    # Test some imports
    from mlpy.tasks import TaskClassif
    from mlpy.learners import LearnerClassifSklearn
    from mlpy.measures import MeasureClassifAccuracy
    from mlpy.resamplings import ResamplingCV
    
    print("[OK] All core modules loaded!")
    
    # Quick functional test
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'y': np.random.choice(['A', 'B'], 100)
    })
    
    task = TaskClassif(data=data, target='y')
    from sklearn.ensemble import RandomForestClassifier
    learner = LearnerClassifSklearn(learner_id='rf', estimator=RandomForestClassifier())
    measure = MeasureClassifAccuracy()
    
    print(f"\n[OK] Created task with {len(task.data())} samples")
    print(f"[OK] Learner: {learner.learner_id}")
    print(f"[OK] Measure: {measure.id}")
    
    print("\n" + "="*50)
    print("MLPY is working perfectly in Anaconda!")
    print("="*50)
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()