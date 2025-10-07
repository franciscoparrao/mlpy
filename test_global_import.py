"""Test script to verify MLPY can be imported globally"""

import sys
import os

# Change to parent directory to test from outside the project
os.chdir('..')

# Try importing MLPY
try:
    import mlpy
    print(f"[OK] MLPY imported successfully!")
    print(f"  Version: {mlpy.__version__}")
    print(f"  Location: {mlpy.__file__}")
    
    # Test importing some modules
    from mlpy.tasks import TaskClassif, TaskRegr
    from mlpy.learners import LearnerClassifSklearn
    from mlpy.measures import MeasureClassifAccuracy
    
    print("\n[OK] All core modules imported successfully!")
    
    # Quick test
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'y': np.random.choice(['A', 'B'], 100)
    })
    
    # Create task
    task = TaskClassif(data=data, target='y')
    print(f"\n[OK] Created task with {len(task.data())} observations")
    
    print("\n" + "="*50)
    print("MLPY is working globally! You can now:")
    print("1. Import it from anywhere: import mlpy")
    print("2. Use the CLI: mlpy --help")
    print("   (Add C:\\Users\\gran_\\AppData\\Roaming\\Python\\Python313\\Scripts to PATH)")
    print("="*50)
    
except ImportError as e:
    print(f"[ERROR] Failed to import MLPY: {e}")
    sys.exit(1)