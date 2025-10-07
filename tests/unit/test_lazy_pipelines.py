"""Tests for lazy pipeline operations."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from mlpy.pipelines import PipeOp
from mlpy.tasks import TaskRegr, TaskClassif
from mlpy.backends import DataBackendPandas

# Try to import lazy operations
try:
    from mlpy.pipelines import (
        LazyPipeOp, LazyPipeOpScale, LazyPipeOpFilter,
        LazyPipeOpSample, LazyPipeOpCache
    )
    LAZY_OPS_AVAILABLE = True
except ImportError:
    LAZY_OPS_AVAILABLE = False

# Try to import big data backends
try:
    import dask
    import dask.dataframe as dd
    from mlpy.backends import DataBackendDask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import vaex
    from mlpy.backends import DataBackendVaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False


@pytest.fixture
def sample_data():
    """Create sample DataFrame."""
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples) * 2 + 1,
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.randn(n_samples)
    })
    
    return df


@pytest.fixture
def pandas_task(sample_data):
    """Create task with pandas backend."""
    return TaskRegr(data=sample_data, target='target', id='test_task')


@pytest.mark.skipif(not LAZY_OPS_AVAILABLE, reason="Lazy operations not available")
class TestLazyPipeOpBase:
    """Test base lazy pipeline operation functionality."""
    
    def test_backend_type_detection(self, pandas_task):
        """Test detection of backend type."""
        op = LazyPipeOpScale()
        
        assert op._get_backend_type(pandas_task) == 'pandas'
        
        # Test with mock backends
        if DASK_AVAILABLE:
            mock_task = Mock()
            mock_task.backend = DataBackendDask(Mock())
            assert op._get_backend_type(mock_task) == 'dask'
            
        if VAEX_AVAILABLE:
            mock_task = Mock()
            mock_task.backend = DataBackendVaex(Mock())
            assert op._get_backend_type(mock_task) == 'vaex'
            
    def test_supports_lazy_property(self):
        """Test that lazy operations report support correctly."""
        op = LazyPipeOpScale()
        assert op.supports_lazy is True


@pytest.mark.skipif(not LAZY_OPS_AVAILABLE, reason="Lazy operations not available")
class TestLazyPipeOpScale:
    """Test lazy scaling operation."""
    
    def test_scale_pandas(self, pandas_task):
        """Test scaling with pandas backend."""
        op = LazyPipeOpScale(method='standard')
        
        # Train
        result = op.train({'input': pandas_task})
        scaled_task = result['output']
        
        assert isinstance(scaled_task, TaskRegr)
        assert scaled_task.nrow == pandas_task.nrow
        assert scaled_task.ncol == pandas_task.ncol
        
        # Check that numeric features are scaled
        scaled_data = scaled_task.data()
        for col in ['feature_1', 'feature_2']:
            # Standard scaling should result in mean ~0, std ~1
            assert abs(scaled_data[col].mean()) < 0.1
            assert abs(scaled_data[col].std() - 1.0) < 0.1
            
        # Predict
        result = op.predict({'input': pandas_task})
        assert isinstance(result['output'], TaskRegr)
        
    def test_scale_methods(self, pandas_task):
        """Test different scaling methods."""
        # MinMax scaling
        op_minmax = LazyPipeOpScale(method='minmax')
        result = op_minmax.train({'input': pandas_task})
        scaled_data = result['output'].data()
        
        for col in ['feature_1', 'feature_2']:
            assert scaled_data[col].min() >= -0.01  # Allow small numerical error
            assert scaled_data[col].max() <= 1.01
            
        # Robust scaling
        op_robust = LazyPipeOpScale(method='robust')
        result = op_robust.train({'input': pandas_task})
        assert isinstance(result['output'], TaskRegr)
        
    def test_scale_specific_columns(self, pandas_task):
        """Test scaling specific columns only."""
        op = LazyPipeOpScale(columns=['feature_1'])
        
        result = op.train({'input': pandas_task})
        scaled_data = result['output'].data()
        
        # feature_1 should be scaled
        assert abs(scaled_data['feature_1'].mean()) < 0.1
        
        # feature_2 should remain unchanged
        original_data = pandas_task.data()
        assert np.allclose(scaled_data['feature_2'], original_data['feature_2'])
        
    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
    def test_scale_dask(self, sample_data):
        """Test scaling with Dask backend."""
        # Create Dask DataFrame
        df_dask = dd.from_pandas(sample_data, npartitions=4)
        backend = DataBackendDask(df_dask)
        task = TaskRegr(backend=backend, target='target')
        
        op = LazyPipeOpScale(method='standard')
        
        # Train
        result = op.train({'input': task})
        scaled_task = result['output']
        
        assert isinstance(scaled_task.backend, DataBackendDask)
        
        # Compute to check results
        scaled_data = scaled_task.data()
        for col in ['feature_1', 'feature_2']:
            assert abs(scaled_data[col].mean()) < 0.1
            assert abs(scaled_data[col].std() - 1.0) < 0.1


@pytest.mark.skipif(not LAZY_OPS_AVAILABLE, reason="Lazy operations not available")
class TestLazyPipeOpFilter:
    """Test lazy filtering operation."""
    
    def test_filter_string_condition(self, pandas_task):
        """Test filtering with string condition."""
        op = LazyPipeOpFilter(condition="feature_1 > 0")
        
        result = op.train({'input': pandas_task})
        filtered_task = result['output']
        
        # Check that all remaining rows meet condition
        filtered_data = filtered_task.data()
        assert (filtered_data['feature_1'] > 0).all()
        assert filtered_task.nrow < pandas_task.nrow
        
    def test_filter_callable_condition(self, pandas_task):
        """Test filtering with callable condition."""
        def condition(df):
            return (df['feature_1'] > 0) & (df['feature_2'] < 2)
            
        op = LazyPipeOpFilter(condition=condition)
        
        result = op.train({'input': pandas_task})
        filtered_task = result['output']
        
        filtered_data = filtered_task.data()
        assert (filtered_data['feature_1'] > 0).all()
        assert (filtered_data['feature_2'] < 2).all()
        
    def test_filter_no_condition_error(self, pandas_task):
        """Test that filter without condition raises error."""
        op = LazyPipeOpFilter()
        
        with pytest.raises(ValueError, match="Filter condition must be specified"):
            op.train({'input': pandas_task})


@pytest.mark.skipif(not LAZY_OPS_AVAILABLE, reason="Lazy operations not available")
class TestLazyPipeOpSample:
    """Test lazy sampling operation."""
    
    def test_sample_n(self, pandas_task):
        """Test sampling fixed number of rows."""
        n_samples = 100
        op = LazyPipeOpSample(n=n_samples, random_state=42)
        
        result = op.train({'input': pandas_task})
        sampled_task = result['output']
        
        assert sampled_task.nrow == n_samples
        assert sampled_task.ncol == pandas_task.ncol
        
    def test_sample_frac(self, pandas_task):
        """Test sampling fraction of rows."""
        frac = 0.1
        op = LazyPipeOpSample(frac=frac, random_state=42)
        
        result = op.train({'input': pandas_task})
        sampled_task = result['output']
        
        expected_rows = int(pandas_task.nrow * frac)
        assert abs(sampled_task.nrow - expected_rows) <= 1
        
    def test_sample_with_replacement(self, pandas_task):
        """Test sampling with replacement."""
        op = LazyPipeOpSample(n=50, replace=True, random_state=42)
        
        result = op.train({'input': pandas_task})
        sampled_task = result['output']
        
        assert sampled_task.nrow == 50
        
    def test_sample_predict_passthrough(self, pandas_task):
        """Test that predict passes through unchanged."""
        op = LazyPipeOpSample(n=100, random_state=42)
        op.train({'input': pandas_task})
        
        result = op.predict({'input': pandas_task})
        assert result['output'] is pandas_task
        
    def test_sample_invalid_params(self):
        """Test invalid parameter combinations."""
        # Neither n nor frac specified
        with pytest.raises(ValueError, match="Either 'n' or 'frac' must be specified"):
            LazyPipeOpSample()
            
        # Both n and frac specified
        with pytest.raises(ValueError, match="Cannot specify both 'n' and 'frac'"):
            LazyPipeOpSample(n=100, frac=0.1)


@pytest.mark.skipif(not LAZY_OPS_AVAILABLE, reason="Lazy operations not available")
class TestLazyPipeOpCache:
    """Test lazy caching operation."""
    
    def test_cache_pandas(self, pandas_task):
        """Test caching with pandas (no-op)."""
        op = LazyPipeOpCache()
        
        result = op.train({'input': pandas_task})
        assert result['output'] is pandas_task
        
    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
    def test_cache_dask(self, sample_data):
        """Test caching with Dask backend."""
        # Create Dask DataFrame
        df_dask = dd.from_pandas(sample_data, npartitions=4)
        backend = DataBackendDask(df_dask)
        task = TaskRegr(backend=backend, target='target')
        
        op = LazyPipeOpCache()
        
        # Mock persist method
        with patch.object(backend, 'persist') as mock_persist:
            result = op.train({'input': task})
            mock_persist.assert_called_once()
            assert result['output'] is task


@pytest.mark.skipif(not LAZY_OPS_AVAILABLE, reason="Lazy operations not available")
class TestLazyPipelineIntegration:
    """Test integration of lazy operations in pipelines."""
    
    def test_chained_operations(self, pandas_task):
        """Test chaining multiple lazy operations."""
        # Create pipeline: filter -> scale -> sample
        op_filter = LazyPipeOpFilter(condition="feature_1 > -1")
        op_scale = LazyPipeOpScale(method='standard')
        op_sample = LazyPipeOpSample(frac=0.5, random_state=42)
        
        # Apply operations in sequence
        result = op_filter.train({'input': pandas_task})
        result = op_scale.train({'input': result['output']})
        result = op_sample.train({'input': result['output']})
        
        final_task = result['output']
        
        # Check results
        assert final_task.nrow < pandas_task.nrow
        
        # Check scaling was applied
        final_data = final_task.data()
        assert abs(final_data['feature_1'].mean()) < 0.1
        assert abs(final_data['feature_1'].std() - 1.0) < 0.1
        
    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
    def test_lazy_pipeline_with_dask(self, sample_data):
        """Test lazy pipeline with Dask backend."""
        # Create large Dask DataFrame
        df_dask = dd.from_pandas(sample_data, npartitions=4)
        backend = DataBackendDask(df_dask)
        task = TaskRegr(backend=backend, target='target')
        
        # Build pipeline
        op_filter = LazyPipeOpFilter(condition="feature_1 > 0")
        op_scale = LazyPipeOpScale(method='minmax')
        op_cache = LazyPipeOpCache()
        
        # Apply operations
        result = op_filter.train({'input': task})
        result = op_scale.train({'input': result['output']})
        result = op_cache.train({'input': result['output']})
        
        final_task = result['output']
        
        # Verify backend is still Dask
        assert isinstance(final_task.backend, DataBackendDask)
        
        # Compute results to verify transformations
        final_data = final_task.data()
        assert (final_data['feature_1'] > 0).all()  # Filter applied
        assert final_data['feature_1'].min() >= -0.01  # Scaling applied
        assert final_data['feature_1'].max() <= 1.01