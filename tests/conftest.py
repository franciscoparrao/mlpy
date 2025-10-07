"""
Pytest configuration and fixtures for MLPY tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Generator, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================================
# FIXTURES FOR TEST DATA
# ============================================================================

@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(['class_A', 'class_B', 'class_C'], n_samples)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    return df


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    return df


@pytest.fixture
def sample_clustering_data():
    """Generate sample clustering data."""
    from sklearn.datasets import make_blobs
    
    X, y_true = make_blobs(n_samples=150, centers=3, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
    
    return df, y_true


@pytest.fixture
def sample_text_data():
    """Generate sample text data for NLP tests."""
    texts = [
        "This is a positive review. Great product!",
        "Terrible experience. Would not recommend.",
        "Average quality, nothing special.",
        "Excellent service and fast delivery!",
        "Poor quality and bad customer support.",
        "Amazing! Exceeded all expectations.",
        "Not worth the price. Very disappointed.",
        "Good value for money. Satisfied.",
        "Outstanding quality and performance!",
        "Worst purchase ever. Complete waste."
    ]
    
    sentiments = [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'positive', 'negative', 'neutral', 'positive', 'negative'
    ]
    
    df = pd.DataFrame({
        'text': texts,
        'sentiment': sentiments
    })
    
    return df


@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data."""
    np.random.seed(42)
    n_samples = 100
    time_steps = 20
    n_features = 3
    
    # Generate sequential data
    data = []
    for i in range(n_samples):
        sequence = np.zeros((time_steps, n_features))
        for t in range(time_steps):
            sequence[t] = np.sin(t / 10 + i/10) + np.random.randn(n_features) * 0.1
        data.append(sequence)
    
    X = np.array(data)
    y = np.random.choice([0, 1], n_samples)
    
    return X, y


@pytest.fixture
def sample_missing_data():
    """Generate sample data with missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        'feature_1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature_2': [np.nan, 2.0, 3.0, 4.0, 5.0],
        'feature_3': [1.0, 2.0, 3.0, np.nan, np.nan],
        'target': ['A', 'B', 'A', 'B', 'A']
    })
    return df


# ============================================================================
# FIXTURES FOR TASKS
# ============================================================================

@pytest.fixture
def classification_task(sample_classification_data):
    """Create a classification task."""
    from mlpy.tasks import TaskClassif
    return TaskClassif(data=sample_classification_data, target='target')


@pytest.fixture
def regression_task(sample_regression_data):
    """Create a regression task."""
    from mlpy.tasks import TaskRegr
    return TaskRegr(data=sample_regression_data, target='target')


# ============================================================================
# FIXTURES FOR MODELS
# ============================================================================

@pytest.fixture
def mock_learner():
    """Create a mock learner for testing."""
    class MockLearner:
        def __init__(self):
            self.is_trained = False
            self.task_type = 'classif'
            
        def train(self, task):
            self.is_trained = True
            return self
            
        def predict(self, task):
            n_samples = len(task.data)
            return ['A'] * n_samples
    
    return MockLearner()


# ============================================================================
# FIXTURES FOR FILES AND DIRECTORIES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_model_file(temp_dir):
    """Create a temporary file path for model saving."""
    return temp_dir / "test_model.pkl"


# ============================================================================
# FIXTURES FOR CONFIGURATION
# ============================================================================

@pytest.fixture
def test_config():
    """Test configuration dictionary."""
    return {
        'random_seed': 42,
        'test_size': 0.2,
        'n_folds': 5,
        'max_iterations': 100,
        'tolerance': 1e-4
    }


# ============================================================================
# PYTEST HOOKS AND CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test"
    )
    config.addinivalue_line(
        "markers", "requires_sklearn: mark test as requiring scikit-learn"
    )
    config.addinivalue_line(
        "markers", "requires_torch: mark test as requiring PyTorch"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test location."""
    for item in items:
        # Add markers based on file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Skip tests requiring optional dependencies if not installed
        if "torch" in item.fixturenames or "pytorch" in str(item.fspath).lower():
            try:
                import torch
            except ImportError:
                skip_torch = pytest.mark.skip(reason="PyTorch not installed")
                item.add_marker(skip_torch)
        
        if "sklearn" in item.fixturenames or "sklearn" in str(item.fspath).lower():
            try:
                import sklearn
            except ImportError:
                skip_sklearn = pytest.mark.skip(reason="scikit-learn not installed")
                item.add_marker(skip_sklearn)


# ============================================================================
# HELPER FUNCTIONS FOR TESTS
# ============================================================================

def assert_array_equal(actual, expected, tolerance=1e-7):
    """Assert that two arrays are equal within tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def assert_dataframe_equal(actual, expected):
    """Assert that two dataframes are equal."""
    pd.testing.assert_frame_equal(actual, expected)


# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
            
        def __enter__(self):
            self.start = time.time()
            return self
            
        def __exit__(self, *args):
            self.end = time.time()
            self.times.append(self.end - self.start)
            
        @property
        def last_time(self):
            return self.times[-1] if self.times else None
            
        @property
        def average_time(self):
            return np.mean(self.times) if self.times else None
    
    return Timer()


# ============================================================================
# MOCK OBJECTS FOR TESTING
# ============================================================================

@pytest.fixture
def mock_model_registry():
    """Mock model registry for testing."""
    class MockRegistry:
        def __init__(self):
            self.models = {}
            
        def register(self, name, model):
            self.models[name] = model
            
        def get(self, name):
            return self.models.get(name)
            
        def list_all(self):
            return list(self.models.values())
    
    return MockRegistry()


@pytest.fixture
def mock_validation_result():
    """Mock validation result."""
    return {
        'valid': True,
        'errors': [],
        'warnings': [],
        'suggestions': ['Consider feature scaling', 'Check for outliers']
    }