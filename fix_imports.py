"""Script para arreglar todos los imports rotos en MLPY."""

import os
import re
from pathlib import Path

# Mapeo de imports incorrectos a correctos
IMPORT_FIXES = {
    # Tasks fixes
    'from ..tasks.classification import': 'from ..tasks.supervised import',
    'from mlpy.tasks.classification import': 'from mlpy.tasks.supervised import',
    'from .tasks.classification import': 'from .tasks.supervised import',
    'from ..tasks.regression import': 'from ..tasks.supervised import',
    'from mlpy.tasks.regression import': 'from mlpy.tasks.supervised import',
    'from .tasks.regression import': 'from .tasks.supervised import',
    
    # Tuning fixes (remove non-existent imports)
    'from .grid_search import TunerGridSearch': '# Removed: TunerGridSearch (use mlpy.automl.TunerGrid)',
    'from .random_search import TunerRandomSearch': '# Removed: TunerRandomSearch (use mlpy.automl.TunerRandom)',
    
    # Learners fixes
    'from mlpy.learners.sklearn import learner_sklearn': 'from mlpy.learners.sklearn import auto_sklearn',
}

def fix_file(filepath):
    """Fix imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return False
    
    original_content = content
    modified = False
    
    for old_import, new_import in IMPORT_FIXES.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            modified = True
            print(f"  Fixed: {old_import}")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def scan_and_fix(root_dir):
    """Scan all Python files and fix imports."""
    root_path = Path(root_dir)
    fixed_files = []
    
    for py_file in root_path.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        if fix_file(py_file):
            fixed_files.append(py_file)
            print(f"Fixed: {py_file.relative_to(root_path)}")
    
    return fixed_files

def validate_imports():
    """Validate that common imports work."""
    test_imports = [
        "from mlpy.tasks import TaskClassif, TaskRegr",
        "from mlpy.learners import LearnerClassifSklearn",
        "from mlpy.measures import MeasureClassifAccuracy",
        "from mlpy.resamplings import ResamplingCV",
        "from mlpy.pipelines import Graph, GraphLearner",
        "from mlpy.automl import TunerGrid, TunerRandom",
        "from mlpy.filters import Filter",
    ]
    
    print("\nValidating imports...")
    failed = []
    
    for import_str in test_imports:
        try:
            exec(import_str)
            print(f"  OK: {import_str}")
        except ImportError as e:
            print(f"  FAIL: {import_str}: {e}")
            failed.append(import_str)
    
    return len(failed) == 0

if __name__ == "__main__":
    print("MLPY Import Fixer")
    print("=" * 50)
    
    # Fix imports
    mlpy_dir = Path(__file__).parent / "mlpy"
    print(f"\nScanning directory: {mlpy_dir}")
    
    fixed = scan_and_fix(mlpy_dir)
    
    print(f"\nFixed {len(fixed)} files")
    
    # Validate
    if validate_imports():
        print("\nAll imports validated successfully!")
    else:
        print("\nSome imports still have issues")
    
    print("\nNext steps:")
    print("1. Update Anaconda installation: xcopy /E /I /Y mlpy C:\\Users\\gran_\\anaconda3\\Lib\\site-packages\\mlpy")
    print("2. Restart Jupyter kernel")
    print("3. Re-run your notebooks")