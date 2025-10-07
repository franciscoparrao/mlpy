"""Tests for big data backends (Dask and Vaex)."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from mlpy.backends import DataBackend, DataBackendPandas
from mlpy.tasks import TaskRegr, TaskClassif

# Try to import optional dependencies
try:
    import dask
    import dask.dataframe as dd
    from mlpy.backends import DataBackendDask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    DataBackendDask = None

try:
    import vaex
    from mlpy.backends import DataBackendVaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False
    DataBackendVaex = None

try:
    from mlpy.tasks.big_data import (
        task_from_dask, task_from_vaex,
        task_from_csv_lazy, task_from_parquet_lazy
    )
    BIG_DATA_HELPERS_AVAILABLE = True
except ImportError:
    BIG_DATA_HELPERS_AVAILABLE = False


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample DataFrame."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    df['class_target'] = (y > y.mean()).astype(int)
    
    return df


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
class TestDaskBackend:
    """Test Dask backend functionality."""
    
    def test_create_backend(self, sample_data):
        """Test creating Dask backend."""
        # Convert to Dask
        df_dask = dd.from_pandas(sample_data, npartitions=4)
        
        # Create backend
        backend = DataBackendDask(df_dask)
        
        assert isinstance(backend, DataBackendDask)
        assert backend.nrow == len(sample_data)
        assert backend.ncol == len(sample_data.columns)
        assert backend.colnames == list(sample_data.columns)
        assert backend.supports_lazy
        assert backend.supports_out_of_core
        
    def test_data_access(self, sample_data):
        """Test data access methods."""
        df_dask = dd.from_pandas(sample_data, npartitions=4)
        backend = DataBackendDask(df_dask)
        
        # Column selection
        cols = ['feature_0', 'feature_1']
        data_subset = backend.data(cols=cols)
        assert isinstance(data_subset, dd.DataFrame)
        assert list(data_subset.columns) == cols
        
        # Row selection with slice
        data_slice = backend.data(rows=slice(0, 10))
        assert len(data_slice.compute()) == 10
        
        # As numpy
        data_np = backend.data(cols=cols, rows=slice(0, 10), as_numpy=True)
        assert isinstance(data_np, np.ndarray)
        assert data_np.shape == (10, 2)
        
    def test_distinct_values(self, sample_data):
        """Test getting distinct values."""
        df_dask = dd.from_pandas(sample_data, npartitions=4)
        backend = DataBackendDask(df_dask)
        
        # Single column
        distinct = backend.distinct('class_target')
        assert isinstance(distinct, pd.DataFrame)
        assert len(distinct) == 2  # Binary class
        
        # Multiple columns (should handle properly)
        distinct_multi = backend.distinct(['class_target', 'feature_0'])
        assert 'class_target' in distinct_multi.columns
        assert 'feature_0' in distinct_multi.columns
        
    def test_missing_values(self, sample_data):
        """Test missing value counting."""
        # Add some missing values
        sample_data_missing = sample_data.copy()
        sample_data_missing.loc[0:5, 'feature_0'] = np.nan
        sample_data_missing.loc[10:15, 'feature_1'] = np.nan
        
        df_dask = dd.from_pandas(sample_data_missing, npartitions=4)
        backend = DataBackendDask(df_dask)
        
        missings = backend.missings(['feature_0', 'feature_1'])
        assert missings['feature_0'] == 6
        assert missings['feature_1'] == 6
        
    def test_cbind_rbind(self, sample_data):
        """Test column and row binding."""
        df_dask = dd.from_pandas(sample_data, npartitions=4)
        backend1 = DataBackendDask(df_dask[['feature_0', 'feature_1']])
        backend2 = DataBackendDask(df_dask[['feature_2', 'feature_3']])
        
        # Column bind
        backend_cbind = backend1.cbind(backend2)
        assert backend_cbind.ncol == 4
        assert 'feature_0' in backend_cbind.colnames
        assert 'feature_3' in backend_cbind.colnames
        
        # Row bind
        backend_rbind = backend1.rbind(backend1)
        assert backend_rbind.nrow == backend1.nrow * 2
        
    def test_from_pandas(self, sample_data):
        """Test creating from pandas."""
        backend = DataBackendDask.from_pandas(sample_data, npartitions=4)
        assert isinstance(backend, DataBackendDask)
        assert backend.nrow == len(sample_data)
        

@pytest.mark.skipif(not VAEX_AVAILABLE, reason="Vaex not available")
class TestVaexBackend:
    """Test Vaex backend functionality."""
    
    def test_create_backend(self, sample_data):
        """Test creating Vaex backend."""
        # Convert to Vaex
        df_vaex = vaex.from_pandas(sample_data)
        
        # Create backend
        backend = DataBackendVaex(df_vaex)
        
        assert isinstance(backend, DataBackendVaex)
        assert backend.nrow == len(sample_data)
        assert backend.ncol == len(sample_data.columns)
        assert backend.colnames == list(sample_data.columns)
        assert backend.supports_lazy
        assert backend.supports_out_of_core
        assert backend.supports_memory_mapping
        
    def test_data_access(self, sample_data):
        """Test data access methods."""
        df_vaex = vaex.from_pandas(sample_data)
        backend = DataBackendVaex(df_vaex)
        
        # Column selection
        cols = ['feature_0', 'feature_1']
        data_subset = backend.data(cols=cols)
        assert isinstance(data_subset, vaex.dataframe.DataFrame)
        assert data_subset.get_column_names() == cols
        
        # Row selection with slice
        data_slice = backend.data(rows=slice(0, 10))
        assert len(data_slice) == 10
        
        # As numpy
        data_np = backend.data(cols=cols, rows=slice(0, 10), as_numpy=True)
        assert isinstance(data_np, np.ndarray)
        assert data_np.shape == (10, 2)
        
    def test_distinct_values(self, sample_data):
        """Test getting distinct values."""
        df_vaex = vaex.from_pandas(sample_data)
        backend = DataBackendVaex(df_vaex)
        
        # Single column
        distinct = backend.distinct('class_target')
        assert isinstance(distinct, pd.DataFrame)
        assert len(distinct) == 2  # Binary class
        
    def test_missing_values(self, sample_data):
        """Test missing value counting."""
        # Add some missing values
        sample_data_missing = sample_data.copy()
        sample_data_missing.loc[0:5, 'feature_0'] = np.nan
        sample_data_missing.loc[10:15, 'feature_1'] = np.nan
        
        df_vaex = vaex.from_pandas(sample_data_missing)
        backend = DataBackendVaex(df_vaex)
        
        missings = backend.missings(['feature_0', 'feature_1'])
        assert missings['feature_0'] == 6
        assert missings['feature_1'] == 6
        
    def test_cbind_rbind(self, sample_data):
        """Test column and row binding."""
        df_vaex = vaex.from_pandas(sample_data)
        backend1 = DataBackendVaex(df_vaex[['feature_0', 'feature_1']])
        backend2 = DataBackendVaex(df_vaex[['feature_2', 'feature_3']])
        
        # Column bind
        backend_cbind = backend1.cbind(backend2)
        assert backend_cbind.ncol == 4
        assert 'feature_0' in backend_cbind.colnames
        assert 'feature_3' in backend_cbind.colnames
        
        # Row bind
        backend_rbind = backend1.rbind(backend1)
        assert backend_rbind.nrow == backend1.nrow * 2
        
    def test_from_pandas(self, sample_data):
        """Test creating from pandas."""
        backend = DataBackendVaex.from_pandas(sample_data)
        assert isinstance(backend, DataBackendVaex)
        assert backend.nrow == len(sample_data)


@pytest.mark.skipif(not BIG_DATA_HELPERS_AVAILABLE, reason="Big data helpers not available")
class TestBigDataHelpers:
    """Test helper functions for creating tasks with big data backends."""
    
    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
    def test_task_from_dask(self, sample_data):
        """Test creating task from Dask DataFrame."""
        df_dask = dd.from_pandas(sample_data, npartitions=4)
        
        # Regression task
        task_regr = task_from_dask(df_dask, target='target', task_type='regression')
        assert isinstance(task_regr, TaskRegr)
        assert isinstance(task_regr.backend, DataBackendDask)
        assert task_regr.target_names == ['target']
        
        # Classification task
        task_classif = task_from_dask(df_dask, target='class_target', task_type='classification')
        assert isinstance(task_classif, TaskClassif)
        assert isinstance(task_classif.backend, DataBackendDask)
        
        # Auto-detect type
        task_auto = task_from_dask(df_dask, target='class_target')
        assert isinstance(task_auto, TaskClassif)  # Should detect as classification
        
    @pytest.mark.skipif(not VAEX_AVAILABLE, reason="Vaex not available")
    def test_task_from_vaex(self, sample_data):
        """Test creating task from Vaex DataFrame."""
        df_vaex = vaex.from_pandas(sample_data)
        
        # Regression task
        task_regr = task_from_vaex(df_vaex, target='target', task_type='regression')
        assert isinstance(task_regr, TaskRegr)
        assert isinstance(task_regr.backend, DataBackendVaex)
        assert task_regr.target_names == ['target']
        
        # Classification task
        task_classif = task_from_vaex(df_vaex, target='class_target', task_type='classification')
        assert isinstance(task_classif, TaskClassif)
        assert isinstance(task_classif.backend, DataBackendVaex)
        
    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
    def test_task_from_csv_lazy_dask(self, tmp_path, sample_data):
        """Test lazy loading CSV with Dask."""
        # Save sample data
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Load with Dask
        task = task_from_csv_lazy(str(csv_path), target='target', backend='dask')
        assert isinstance(task, TaskRegr)
        assert isinstance(task.backend, DataBackendDask)
        
    @pytest.mark.skipif(not VAEX_AVAILABLE, reason="Vaex not available")
    def test_task_from_csv_lazy_vaex(self, tmp_path, sample_data):
        """Test lazy loading CSV with Vaex."""
        # Save sample data
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Load with Vaex
        task = task_from_csv_lazy(str(csv_path), target='target', backend='vaex')
        assert isinstance(task, TaskRegr)
        assert isinstance(task.backend, DataBackendVaex)


class TestBigDataBackendsIntegration:
    """Test that backends work correctly when libraries are not available."""
    
    def test_imports_without_dask(self):
        """Test that imports work when Dask is not available."""
        with patch.dict('sys.modules', {'dask': None, 'dask.dataframe': None}):
            # Should not raise ImportError
            from mlpy import backends
            assert hasattr(backends, 'DataBackendPandas')
            
    def test_imports_without_vaex(self):
        """Test that imports work when Vaex is not available."""
        with patch.dict('sys.modules', {'vaex': None}):
            # Should not raise ImportError
            from mlpy import backends
            assert hasattr(backends, 'DataBackendPandas')
            
    def test_task_creation_without_backends(self):
        """Test proper errors when backends not available."""
        if not BIG_DATA_HELPERS_AVAILABLE:
            pytest.skip("Big data helpers not available")
            
        # Mock missing Dask
        with patch('mlpy.tasks.big_data.DASK_AVAILABLE', False):
            with pytest.raises(ImportError, match="Dask"):
                task_from_csv_lazy("dummy.csv", "target", backend="dask")
                
        # Mock missing Vaex  
        with patch('mlpy.tasks.big_data.VAEX_AVAILABLE', False):
            with pytest.raises(ImportError, match="Vaex"):
                task_from_csv_lazy("dummy.csv", "target", backend="vaex")