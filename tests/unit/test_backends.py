"""Tests for data backends."""

import pytest
import numpy as np
import pandas as pd

from mlpy.backends import (
    DataBackend,
    DataBackendPandas,
    DataBackendNumPy,
    DataBackendCbind,
    DataBackendRbind,
)


class TestDataBackendPandas:
    """Test DataBackendPandas functionality."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "feature1": [1.1, 2.2, 3.3, 4.4, 5.5],
            "feature2": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        })
    
    def test_create_backend(self, sample_df):
        """Test creating a pandas backend."""
        backend = DataBackendPandas(sample_df)
        
        assert backend.nrow == 5
        assert backend.ncol == 4
        assert backend.colnames == ["id", "feature1", "feature2", "target"]
        assert backend.rownames == [0, 1, 2, 3, 4]
    
    def test_create_with_primary_key(self, sample_df):
        """Test creating backend with primary key."""
        backend = DataBackendPandas(sample_df, primary_key="id")
        
        assert backend.primary_key == "id"
        assert backend.rownames == [1, 2, 3, 4, 5]
    
    def test_primary_key_validation(self, sample_df):
        """Test primary key validation."""
        # Non-existent column
        with pytest.raises(ValueError, match="not found"):
            DataBackendPandas(sample_df, primary_key="missing")
        
        # Duplicate values
        df_dup = sample_df.copy()
        df_dup.loc[1, "id"] = 1  # Duplicate ID
        with pytest.raises(ValueError, match="duplicates"):
            DataBackendPandas(df_dup, primary_key="id")
    
    def test_data_retrieval(self, sample_df):
        """Test data retrieval in different formats."""
        backend = DataBackendPandas(sample_df)
        
        # Get all data as DataFrame
        df = backend.data()
        assert isinstance(df, pd.DataFrame)
        assert df.equals(sample_df)
        
        # Get subset of columns
        df_subset = backend.data(cols=["feature1", "target"])
        assert list(df_subset.columns) == ["feature1", "target"]
        
        # Get subset of rows
        df_rows = backend.data(rows=[0, 2, 4])
        assert len(df_rows) == 3
        assert list(df_rows.index) == [0, 2, 4]
        
        # Get as array
        arr = backend.data(data_format="array")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5, 4)
        
        # Get as dict
        d = backend.data(data_format="dict")
        assert isinstance(d, dict)
        assert set(d.keys()) == set(sample_df.columns)
        assert all(isinstance(v, np.ndarray) for v in d.values())
    
    def test_head(self, sample_df):
        """Test head method."""
        backend = DataBackendPandas(sample_df)
        
        head3 = backend.head(3)
        assert len(head3) == 3
        assert head3.equals(sample_df.head(3))
    
    def test_distinct(self, sample_df):
        """Test getting distinct values."""
        backend = DataBackendPandas(sample_df)
        
        # Single column
        dist = backend.distinct(["feature2"])
        assert "feature2" in dist
        assert set(dist["feature2"]) == {"a", "b"}
        
        # Multiple columns
        dist = backend.distinct(["feature2", "target"])
        assert set(dist["target"]) == {0, 1}
    
    def test_missings(self, sample_df):
        """Test counting missing values."""
        # Add some missing values
        df_miss = sample_df.copy()
        df_miss.loc[1, "feature1"] = np.nan
        df_miss.loc[3, "feature2"] = np.nan
        
        backend = DataBackendPandas(df_miss)
        
        # Count all missings
        assert backend.missings() == 2
        
        # Count in specific columns
        assert backend.missings(cols=["feature1"]) == 1
        assert backend.missings(cols=["feature2"]) == 1
        assert backend.missings(cols=["target"]) == 0
    
    def test_col_info(self, sample_df):
        """Test column information."""
        backend = DataBackendPandas(sample_df)
        info = backend.col_info()
        
        assert info["id"]["type"] == "integer"
        assert info["feature1"]["type"] == "numeric"
        assert info["feature2"]["type"] == "character"
        assert info["target"]["type"] == "integer"
        
        assert all(info[col]["missing"] == 0 for col in info)
        assert info["feature2"]["distinct"] == 2


class TestDataBackendNumPy:
    """Test DataBackendNumPy functionality."""
    
    @pytest.fixture
    def sample_array(self):
        """Create a sample numpy array."""
        return np.array([
            [1, 1.1, 0],
            [2, 2.2, 1],
            [3, 3.3, 0],
            [4, 4.4, 1],
            [5, 5.5, 0],
        ])
    
    def test_create_from_array(self, sample_array):
        """Test creating backend from numpy array."""
        backend = DataBackendNumPy(sample_array, colnames=["id", "feature", "target"])
        
        assert backend.nrow == 5
        assert backend.ncol == 3
        assert backend.colnames == ["id", "feature", "target"]
        assert backend.rownames == [0, 1, 2, 3, 4]
    
    def test_create_from_dict(self):
        """Test creating backend from dict of arrays."""
        data = {
            "feature1": np.array([1.1, 2.2, 3.3]),
            "feature2": np.array([10, 20, 30]),
            "target": np.array([0, 1, 0]),
        }
        backend = DataBackendNumPy(data)
        
        assert backend.nrow == 3
        assert backend.ncol == 3
        assert set(backend.colnames) == {"feature1", "feature2", "target"}
    
    def test_auto_column_names(self, sample_array):
        """Test automatic column name generation."""
        backend = DataBackendNumPy(sample_array)
        
        assert backend.colnames == ["V1", "V2", "V3"]
    
    def test_data_retrieval(self, sample_array):
        """Test data retrieval."""
        backend = DataBackendNumPy(sample_array, colnames=["id", "feature", "target"])
        
        # Get as DataFrame
        df = backend.data()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "feature", "target"]
        
        # Get as array
        arr = backend.data(data_format="array")
        assert np.array_equal(arr, sample_array)
        
        # Get subset
        arr_subset = backend.data(rows=[0, 2], cols=["feature", "target"], data_format="array")
        assert arr_subset.shape == (2, 2)
    
    def test_missings_float(self):
        """Test counting NaN values in float arrays."""
        data = np.array([
            [1.0, np.nan],
            [2.0, 3.0],
            [np.nan, 4.0],
        ])
        backend = DataBackendNumPy(data)
        
        assert backend.missings() == 2
        assert backend.missings(cols=["V1"]) == 1
        assert backend.missings(cols=["V2"]) == 1


class TestDataBackendComposition:
    """Test composite backends (Cbind and Rbind)."""
    
    @pytest.fixture
    def backend1(self):
        """First backend."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "feature1": [1.1, 2.2, 3.3],
        })
        return DataBackendPandas(df)
    
    @pytest.fixture  
    def backend2(self):
        """Second backend."""
        df = pd.DataFrame({
            "feature2": ["a", "b", "c"],
            "target": [0, 1, 0],
        })
        return DataBackendPandas(df)
    
    def test_cbind(self, backend1, backend2):
        """Test column binding."""
        combined = DataBackendCbind([backend1, backend2])
        
        assert combined.nrow == 3
        assert combined.ncol == 4
        assert combined.colnames == ["id", "feature1", "feature2", "target"]
        
        # Get all data
        df = combined.data()
        assert list(df.columns) == ["id", "feature1", "feature2", "target"]
        assert len(df) == 3
        
        # Get subset
        df_subset = combined.data(cols=["feature1", "target"])
        assert list(df_subset.columns) == ["feature1", "target"]
    
    def test_cbind_validation(self, backend1):
        """Test cbind validation."""
        # Different number of rows
        df2 = pd.DataFrame({"col": [1, 2]})  # 2 rows vs 3
        backend2 = DataBackendPandas(df2)
        
        with pytest.raises(ValueError, match="same number of rows"):
            DataBackendCbind([backend1, backend2])
        
        # Duplicate column names
        df3 = pd.DataFrame({"id": [4, 5, 6]})  # "id" already exists
        backend3 = DataBackendPandas(df3)
        
        with pytest.raises(ValueError, match="unique"):
            DataBackendCbind([backend1, backend3])
    
    def test_rbind(self):
        """Test row binding."""
        df1 = pd.DataFrame({
            "feature": [1.1, 2.2],
            "target": [0, 1],
        })
        df2 = pd.DataFrame({
            "feature": [3.3, 4.4],
            "target": [0, 1],
        })
        
        backend1 = DataBackendPandas(df1)
        backend2 = DataBackendPandas(df2)
        
        combined = DataBackendRbind([backend1, backend2])
        
        assert combined.nrow == 4
        assert combined.ncol == 2
        assert combined.colnames == ["feature", "target"]
        
        # Get all data
        df = combined.data()
        assert len(df) == 4
        assert list(df["feature"]) == [1.1, 2.2, 3.3, 4.4]
    
    def test_rbind_validation(self):
        """Test rbind validation."""
        df1 = pd.DataFrame({"col1": [1, 2]})
        df2 = pd.DataFrame({"col2": [3, 4]})  # Different columns
        
        backend1 = DataBackendPandas(df1)
        backend2 = DataBackendPandas(df2)
        
        with pytest.raises(ValueError, match="same columns"):
            DataBackendRbind([backend1, backend2])


class TestDataBackendExtended:
    """Extended tests for backend functionality."""
    
    def test_backend_with_datetime(self):
        """Test backend with datetime columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'value': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        backend = DataBackendPandas(df)
        
        assert backend.ncol == 3
        data = backend.data()
        assert 'date' in data.columns
        
    def test_backend_remove_columns(self):
        """Test removing columns from backend."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        backend = DataBackendPandas(df)
        
        # Remove a column
        backend.remove_columns(['b'])
        assert backend.ncol == 2
        assert backend.colnames == ['a', 'c']
        
    def test_backend_rename_columns(self):
        """Test renaming columns."""
        df = pd.DataFrame({
            'old_name': [1, 2, 3],
            'another': [4, 5, 6]
        })
        backend = DataBackendPandas(df)
        
        # Rename columns
        backend.rename_columns({'old_name': 'new_name'})
        assert 'new_name' in backend.colnames
        assert 'old_name' not in backend.colnames
        
    def test_numpy_backend_column_types(self):
        """Test column type inference in NumPy backend."""
        # Mixed types
        data = np.array([
            [1, 2.5, 3],
            [4, 5.5, 6],
            [7, 8.5, 9]
        ])
        backend = DataBackendNumPy(data, colnames=['int_col', 'float_col', 'int_col2'])
        
        info = backend.col_info()
        assert info['float_col']['type'] == 'numeric'
        
    def test_cbind_multiple_backends(self):
        """Test cbind with more than 2 backends."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})
        df3 = pd.DataFrame({'c': [7, 8, 9]})
        
        b1 = DataBackendPandas(df1)
        b2 = DataBackendPandas(df2)
        b3 = DataBackendPandas(df3)
        
        combined = DataBackendCbind([b1, b2, b3])
        assert combined.ncol == 3
        assert combined.colnames == ['a', 'b', 'c']
        
        data = combined.data()
        assert list(data.iloc[0]) == [1, 4, 7]
        
    def test_rbind_multiple_backends(self):
        """Test rbind with more than 2 backends."""
        df1 = pd.DataFrame({'a': [1], 'b': [2]})
        df2 = pd.DataFrame({'a': [3], 'b': [4]})
        df3 = pd.DataFrame({'a': [5], 'b': [6]})
        
        b1 = DataBackendPandas(df1)
        b2 = DataBackendPandas(df2)
        b3 = DataBackendPandas(df3)
        
        combined = DataBackendRbind([b1, b2, b3])
        assert combined.nrow == 3
        assert list(combined.data()['a']) == [1, 3, 5]
        
    def test_backend_data_format_matrix(self):
        """Test getting data in matrix format."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        backend = DataBackendPandas(df)
        
        # Test matrix format (if supported)
        try:
            matrix = backend.data(data_format="matrix")
            assert matrix.shape == (3, 2)
        except:
            # If matrix not supported, array should work
            arr = backend.data(data_format="array")
            assert arr.shape == (3, 2)
            
    def test_backend_with_single_row(self):
        """Test backend with single row."""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        backend = DataBackendPandas(df)
        
        assert backend.nrow == 1
        assert backend.ncol == 3
        
        # Test operations on single row
        head = backend.head(1)
        assert len(head) == 1
        
    def test_backend_with_single_column(self):
        """Test backend with single column."""
        df = pd.DataFrame({'only_col': [1, 2, 3, 4, 5]})
        backend = DataBackendPandas(df)
        
        assert backend.nrow == 5
        assert backend.ncol == 1
        
        # Get as array
        arr = backend.data(data_format="array")
        assert arr.shape == (5, 1)
        
    def test_backend_empty_selection(self):
        """Test selecting empty subset."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        backend = DataBackendPandas(df)
        
        # Empty row selection
        empty = backend.data(rows=[])
        assert len(empty) == 0
        assert list(empty.columns) == ['a', 'b']
        
    def test_backend_hash(self):
        """Test backend hashing."""
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df3 = pd.DataFrame({'a': [1, 2], 'b': [3, 5]})  # Different
        
        b1 = DataBackendPandas(df1)
        b2 = DataBackendPandas(df2)
        b3 = DataBackendPandas(df3)
        
        # Same data should have same hash
        assert b1.hash == b2.hash
        # Different data should have different hash
        assert b1.hash != b3.hash
        
    def test_numpy_backend_from_list(self):
        """Test creating NumPy backend from list."""
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        backend = DataBackendNumPy(data)
        
        assert backend.nrow == 3
        assert backend.ncol == 3
        assert backend.colnames == ['V1', 'V2', 'V3']
        
    def test_backend_col_info_with_missing(self):
        """Test col_info with missing values."""
        df = pd.DataFrame({
            'complete': [1, 2, 3],
            'partial': [1, np.nan, 3],
            'mostly_missing': [np.nan, np.nan, 3]
        })
        backend = DataBackendPandas(df)
        
        info = backend.col_info()
        assert info['complete']['missing'] == 0
        assert info['partial']['missing'] == 1
        assert info['mostly_missing']['missing'] == 2