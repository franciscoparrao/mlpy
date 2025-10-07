"""
Tests específicos para CI/CD pipeline.
"""

import pytest
import sys
import os

def test_python_version():
    """Test that we're running on a supported Python version."""
    assert sys.version_info >= (3, 8), f"Python {sys.version} is not supported"
    assert sys.version_info < (4, 0), f"Python {sys.version} is too new"

def test_imports():
    """Test that core imports work."""
    try:
        import mlpy
        import mlpy.tasks
        import mlpy.learners
        import mlpy.measures
        import mlpy.resamplings
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_package_structure():
    """Test that the package has expected structure."""
    import mlpy
    
    # Check that key modules exist
    assert hasattr(mlpy, 'tasks')
    assert hasattr(mlpy, 'learners')
    assert hasattr(mlpy, 'measures')
    assert hasattr(mlpy, 'resamplings')

@pytest.mark.sklearn
def test_sklearn_available():
    """Test that sklearn integration works."""
    try:
        from mlpy.learners.sklearn import LearnerRegrLM
        learner = LearnerRegrLM(id="test_lm")
        assert learner.id == "test_lm"
    except ImportError:
        pytest.skip("sklearn not available")

@pytest.mark.torch
def test_torch_available():
    """Test if PyTorch is available."""
    try:
        import torch
        # Simple tensor operation
        x = torch.randn(2, 3)
        assert x.shape == (2, 3)
    except ImportError:
        pytest.skip("PyTorch not available")

@pytest.mark.tgpy
def test_tgpy_available():
    """Test if TGPY is available."""
    try:
        import tgpy
        # Test basic import
        assert hasattr(tgpy, 'TgPriorUnivariate')
    except ImportError:
        pytest.skip("TGPY not available")

def test_basic_workflow():
    """Test a basic MLPY workflow."""
    import numpy as np
    import pandas as pd
    from mlpy.tasks import TaskRegr
    from mlpy.learners.baseline import LearnerRegrFeatureless
    from mlpy.measures import MeasureRegrRMSE
    
    # Create simple data
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = X.sum(axis=1) + 0.1 * np.random.randn(20)
    data = pd.DataFrame(X, columns=['x1', 'x2'])
    data['y'] = y
    
    # Create task
    task = TaskRegr(id="test", data=data, target="y")
    
    # Create learner
    learner = LearnerRegrFeatureless(id="mean")
    
    # Train
    learner.train(task)
    
    # Predict
    pred = learner.predict(task)
    
    # Evaluate
    measure = MeasureRegrRMSE()
    score = measure.score(pred)
    
    assert isinstance(score, (int, float))
    assert score >= 0

def test_version_info():
    """Test that version information is available."""
    import mlpy
    
    # Should have version
    assert hasattr(mlpy, '__version__')
    assert isinstance(mlpy.__version__, str)
    assert len(mlpy.__version__) > 0

@pytest.mark.slow
def test_comprehensive_workflow():
    """Test a more comprehensive workflow (marked as slow)."""
    import numpy as np
    import pandas as pd
    from mlpy.tasks import TaskRegr
    from mlpy.learners.baseline import LearnerRegrFeatureless
    from mlpy.resamplings import ResamplingCV
    from mlpy.measures import MeasureRegrRMSE, MeasureRegrMAE
    from mlpy.benchmark import benchmark
    
    # Create data
    np.random.seed(42)
    n = 50
    X = np.random.randn(n, 3)
    y = X.sum(axis=1) + 0.2 * np.random.randn(n)
    data = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    data['y'] = y
    
    # Create components
    task = TaskRegr(id="test", data=data, target="y")
    learners = [
        LearnerRegrFeatureless(id="mean", method="mean"),
        LearnerRegrFeatureless(id="median", method="median")
    ]
    resampling = ResamplingCV(folds=3)
    measures = [
        MeasureRegrRMSE(),
        MeasureRegrMAE()
    ]
    
    # Run benchmark
    results = benchmark(
        tasks=[task],
        learners=learners,
        resampling=resampling,
        measures=measures
    )
    
    # Check results
    assert results is not None
    print(f"Results type: {type(results)}")
    print(f"Results attributes: {dir(results)}")
    
    # The benchmark ran successfully as shown in logs
    assert hasattr(results, 'tasks')
    assert hasattr(results, 'learners')
    assert len(results.tasks) > 0
    assert len(results.learners) > 0