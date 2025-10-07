"""Tests for optional backend implementations (Dask, Vaex)."""

import pytest
import numpy as np
import pandas as pd


# Test Dask backend if available
try:
    import dask
    import dask.dataframe as dd
    from mlpy.backends import DataBackendDask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


# Test Vaex backend if available  
try:
    import vaex
    from mlpy.backends import DataBackendVaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not installed")
class TestDataBackendDask:
    """Test Dask backend functionality."""
    
    @pytest.fixture
    def sample_dask_df(self):
        """Create a sample Dask DataFrame."""
        df = pd.DataFrame({
            "id": range(1000),
            "value": np.random.randn(1000),
            "category": np.random.choice(['A', 'B', 'C'], 1000),
            "target": np.random.choice([0, 1], 1000)
        })
        return dd.from_pandas(df, npartitions=4)
    
    def test_create_dask_backend(self, sample_dask_df):
        """Test creating a Dask backend."""
        backend = DataBackendDask(sample_dask_df)
        
        assert backend.nrow == 1000
        assert backend.ncol == 4
        assert set(backend.colnames) == {"id", "value", "category", "target"}
        
    def test_dask_data_retrieval(self, sample_dask_df):
        """Test data retrieval from Dask backend."""
        backend = DataBackendDask(sample_dask_df)
        
        # Get as DataFrame (computes)
        df = backend.data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        
        # Get subset
        subset = backend.data(rows=list(range(10)), cols=["value", "target"])
        assert len(subset) == 10
        assert list(subset.columns) == ["value", "target"]
        
    def test_dask_head(self, sample_dask_df):
        """Test head method on Dask backend."""
        backend = DataBackendDask(sample_dask_df)
        
        head = backend.head(5)
        assert len(head) == 5
        assert isinstance(head, pd.DataFrame)
        
    def test_dask_distinct(self, sample_dask_df):
        """Test distinct values from Dask backend."""
        backend = DataBackendDask(sample_dask_df)
        
        distinct = backend.distinct(["category"])
        assert set(distinct["category"]) == {'A', 'B', 'C'}
        
    def test_dask_missings(self):
        """Test counting missing values in Dask backend."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [np.nan, 2, 3, 4],
            'c': [1, 2, 3, 4]
        })
        dask_df = dd.from_pandas(df, npartitions=2)
        backend = DataBackendDask(dask_df)
        
        assert backend.missings() == 2
        assert backend.missings(cols=['a']) == 1
        assert backend.missings(cols=['c']) == 0
        
    def test_dask_col_info(self, sample_dask_df):
        """Test column information from Dask backend."""
        backend = DataBackendDask(sample_dask_df)
        info = backend.col_info()
        
        assert 'id' in info
        assert 'value' in info
        assert info['category']['type'] == 'character'
        assert info['value']['type'] == 'numeric'
        
    def test_dask_with_partitions(self):
        """Test Dask backend with different partition configurations."""
        df = pd.DataFrame({
            'x': range(100),
            'y': np.random.randn(100)
        })
        
        # Test with different numbers of partitions
        for nparts in [1, 4, 10]:
            dask_df = dd.from_pandas(df, npartitions=nparts)
            backend = DataBackendDask(dask_df)
            
            assert backend.nrow == 100
            assert backend.ncol == 2
            
            # Operations should work regardless of partitions
            data = backend.data()
            assert len(data) == 100


@pytest.mark.skipif(not VAEX_AVAILABLE, reason="Vaex not installed")
class TestDataBackendVaex:
    """Test Vaex backend functionality."""
    
    @pytest.fixture
    def sample_vaex_df(self):
        """Create a sample Vaex DataFrame."""
        df = pd.DataFrame({
            "id": range(1000),
            "x": np.random.randn(1000),
            "y": np.random.randn(1000),
            "label": np.random.choice(['A', 'B'], 1000)
        })
        return vaex.from_pandas(df)
    
    def test_create_vaex_backend(self, sample_vaex_df):
        """Test creating a Vaex backend."""
        backend = DataBackendVaex(sample_vaex_df)
        
        assert backend.nrow == 1000
        assert backend.ncol == 4
        assert set(backend.colnames) == {"id", "x", "y", "label"}
        
    def test_vaex_data_retrieval(self, sample_vaex_df):
        """Test data retrieval from Vaex backend."""
        backend = DataBackendVaex(sample_vaex_df)
        
        # Get all data as DataFrame
        df = backend.data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        
        # Get subset
        subset = backend.data(rows=list(range(10)), cols=["x", "y"])
        assert len(subset) == 10
        assert list(subset.columns) == ["x", "y"]
        
    def test_vaex_head(self, sample_vaex_df):
        """Test head method on Vaex backend."""
        backend = DataBackendVaex(sample_vaex_df)
        
        head = backend.head(10)
        assert len(head) == 10
        assert isinstance(head, pd.DataFrame)
        
    def test_vaex_distinct(self, sample_vaex_df):
        """Test distinct values from Vaex backend."""
        backend = DataBackendVaex(sample_vaex_df)
        
        distinct = backend.distinct(["label"])
        assert set(distinct["label"]) == {'A', 'B'}
        
    def test_vaex_missings(self):
        """Test counting missing values in Vaex backend."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [np.nan, 2, 3, np.nan, 5],
            'c': [1, 2, 3, 4, 5]
        })
        vaex_df = vaex.from_pandas(df)
        backend = DataBackendVaex(vaex_df)
        
        assert backend.missings() == 3
        assert backend.missings(cols=['a']) == 1
        assert backend.missings(cols=['b']) == 2
        assert backend.missings(cols=['c']) == 0
        
    def test_vaex_col_info(self, sample_vaex_df):
        """Test column information from Vaex backend."""
        backend = DataBackendVaex(sample_vaex_df)
        info = backend.col_info()
        
        assert 'id' in info
        assert 'x' in info
        assert info['x']['type'] == 'numeric'
        assert info['label']['type'] == 'character'
        
    def test_vaex_with_expressions(self):
        """Test Vaex backend with virtual columns/expressions."""
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })
        vaex_df = vaex.from_pandas(df)
        
        # Add virtual column
        vaex_df['z'] = vaex_df.x + vaex_df.y
        
        backend = DataBackendVaex(vaex_df)
        assert backend.ncol == 3
        assert 'z' in backend.colnames
        
        # Virtual column should be accessible
        data = backend.data()
        assert 'z' in data.columns
        np.testing.assert_allclose(
            data['z'].values,
            data['x'].values + data['y'].values
        )
        
    def test_vaex_memory_efficiency(self):
        """Test Vaex backend memory efficiency with large data."""
        # Create a moderately large dataset
        n = 10000
        df = pd.DataFrame({
            'x': np.random.randn(n),
            'y': np.random.randn(n),
            'z': np.random.randn(n)
        })
        vaex_df = vaex.from_pandas(df)
        
        backend = DataBackendVaex(vaex_df)
        
        # Operations should be lazy and efficient
        assert backend.nrow == n
        assert backend.ncol == 3
        
        # Head should be fast even on large data
        head = backend.head(5)
        assert len(head) == 5


class TestBackendComparison:
    """Test comparing different backend implementations."""
    
    def test_pandas_numpy_equivalence(self):
        """Test that Pandas and NumPy backends give same results."""
        # Create same data in different formats
        data_dict = {
            'a': np.array([1, 2, 3, 4, 5]),
            'b': np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            'c': np.array([10, 20, 30, 40, 50])
        }
        
        # Pandas backend
        df = pd.DataFrame(data_dict)
        pandas_backend = DataBackendPandas(df)
        
        # NumPy backend
        numpy_backend = DataBackendNumPy(data_dict)
        
        # Compare basic properties
        assert pandas_backend.nrow == numpy_backend.nrow
        assert pandas_backend.ncol == numpy_backend.ncol
        assert set(pandas_backend.colnames) == set(numpy_backend.colnames)
        
        # Compare data
        pandas_data = pandas_backend.data()
        numpy_data = numpy_backend.data()
        
        for col in data_dict.keys():
            np.testing.assert_array_equal(
                pandas_data[col].values,
                numpy_data[col].values
            )
            
    @pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not installed")
    def test_pandas_dask_equivalence(self):
        """Test that Pandas and Dask backends give same results."""
        df = pd.DataFrame({
            'x': range(100),
            'y': np.random.randn(100),
            'z': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        pandas_backend = DataBackendPandas(df)
        
        dask_df = dd.from_pandas(df, npartitions=4)
        dask_backend = DataBackendDask(dask_df)
        
        # Compare properties
        assert pandas_backend.nrow == dask_backend.nrow
        assert pandas_backend.ncol == dask_backend.ncol
        assert pandas_backend.colnames == dask_backend.colnames
        
        # Compare data
        pandas_data = pandas_backend.data()
        dask_data = dask_backend.data()
        
        pd.testing.assert_frame_equal(pandas_data, dask_data)
        
    @pytest.mark.skipif(not VAEX_AVAILABLE, reason="Vaex not installed")
    def test_pandas_vaex_equivalence(self):
        """Test that Pandas and Vaex backends give same results."""
        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.uniform(0, 10, 100),
            'c': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        
        pandas_backend = DataBackendPandas(df)
        
        vaex_df = vaex.from_pandas(df)
        vaex_backend = DataBackendVaex(vaex_df)
        
        # Compare properties
        assert pandas_backend.nrow == vaex_backend.nrow
        assert pandas_backend.ncol == vaex_backend.ncol
        assert set(pandas_backend.colnames) == set(vaex_backend.colnames)
        
        # Compare head
        pandas_head = pandas_backend.head(10)
        vaex_head = vaex_backend.head(10)
        
        pd.testing.assert_frame_equal(pandas_head, vaex_head)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])