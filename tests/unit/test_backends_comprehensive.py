"""
Tests comprehensivos para todos los backends de MLPY.

Este archivo contiene tests detallados para aumentar la cobertura de:
- DataBackendPandas
- DataBackendNumPy  
- DataBackendDask
- DataBackendVaex
- DataBackendCbind
- DataBackendRbind
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from typing import Dict, List, Any
import warnings

from mlpy.backends import (
    DataBackend,
    DataBackendPandas,
    DataBackendNumPy,
    DataBackendCbind,
    DataBackendRbind,
)


class TestDataBackendBase:
    """Tests para la clase base DataBackend."""
    
    def test_abstract_methods(self):
        """Test que DataBackend es abstracta y no se puede instanciar."""
        with pytest.raises(TypeError):
            DataBackend()
    
    def test_backend_properties(self):
        """Test propiedades básicas del backend."""
        # Usar pandas backend como implementación concreta
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        backend = DataBackendPandas(df)
        
        # Propiedades básicas
        assert hasattr(backend, 'nrow')
        assert hasattr(backend, 'ncol')
        assert hasattr(backend, 'colnames')
        assert hasattr(backend, 'rownames')
        assert hasattr(backend, 'formats')
        
        # Hash único
        assert backend.hash is not None
        assert isinstance(backend.hash, str)
        
        # El hash debe ser consistente
        hash1 = backend.hash
        hash2 = backend.hash
        assert hash1 == hash2


class TestDataBackendPandas:
    """Tests completos para DataBackendPandas."""
    
    @pytest.fixture
    def simple_df(self):
        """DataFrame simple para tests."""
        return pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True],
            'cat_col': pd.Categorical(['cat1', 'cat2', 'cat1', 'cat2', 'cat1'])
        })
    
    @pytest.fixture
    def df_with_missing(self):
        """DataFrame con valores faltantes."""
        return pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': ['a', None, 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, np.nan, 5.5]
        })
    
    def test_initialization(self, simple_df):
        """Test inicialización del backend."""
        backend = DataBackendPandas(simple_df)
        
        assert backend.nrow == 5
        assert backend.ncol == 5
        assert backend.colnames == list(simple_df.columns)
        assert backend.rownames == list(simple_df.index)
        
    def test_initialization_with_primary_key(self, simple_df):
        """Test inicialización con primary key."""
        simple_df['id'] = [10, 20, 30, 40, 50]
        backend = DataBackendPandas(simple_df, primary_key='id')
        
        assert backend.primary_key == 'id'
        assert backend.rownames == [10, 20, 30, 40, 50]
        
    def test_formats_property(self, simple_df):
        """Test propiedad formats que indica capacidades del backend."""
        backend = DataBackendPandas(simple_df)
        formats = backend.formats
        
        assert 'dataframe' in formats
        assert 'array' in formats
        assert 'dict' in formats
        
    def test_data_retrieval_all_formats(self, simple_df):
        """Test obtención de datos en todos los formatos."""
        backend = DataBackendPandas(simple_df)
        
        # DataFrame (default)
        df = backend.data()
        assert isinstance(df, pd.DataFrame)
        assert df.equals(simple_df)
        
        # Array
        arr = backend.data(data_format='array')
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5, 5)
        
        # Dict
        d = backend.data(data_format='dict')
        assert isinstance(d, dict)
        assert set(d.keys()) == set(simple_df.columns)
        assert all(len(v) == 5 for v in d.values())
        
    def test_data_subset_selection(self, simple_df):
        """Test selección de subconjuntos de datos."""
        backend = DataBackendPandas(simple_df)
        
        # Seleccionar columnas
        subset_cols = backend.data(cols=['int_col', 'float_col'])
        assert subset_cols.shape == (5, 2)
        assert list(subset_cols.columns) == ['int_col', 'float_col']
        
        # Seleccionar filas
        subset_rows = backend.data(rows=[0, 2, 4])
        assert subset_rows.shape == (3, 5)
        assert list(subset_rows.index) == [0, 2, 4]
        
        # Seleccionar ambos
        subset_both = backend.data(rows=[1, 3], cols=['str_col', 'bool_col'])
        assert subset_both.shape == (2, 2)
        assert list(subset_both.index) == [1, 3]
        assert list(subset_both.columns) == ['str_col', 'bool_col']
        
    def test_head_method(self, simple_df):
        """Test método head."""
        backend = DataBackendPandas(simple_df)
        
        # Default n=6
        head_default = backend.head()
        assert len(head_default) == 5  # Solo hay 5 filas
        
        # Custom n
        head_3 = backend.head(3)
        assert len(head_3) == 3
        assert head_3.equals(simple_df.head(3))
        
        # n mayor que filas disponibles
        head_10 = backend.head(10)
        assert len(head_10) == 5
        
    def test_distinct_values(self, simple_df):
        """Test obtención de valores únicos."""
        backend = DataBackendPandas(simple_df)
        
        # Una columna
        distinct_single = backend.distinct(['str_col'])
        assert 'str_col' in distinct_single
        assert set(distinct_single['str_col']) == {'a', 'b', 'c', 'd', 'e'}
        
        # Múltiples columnas
        distinct_multi = backend.distinct(['bool_col', 'cat_col'])
        assert 'bool_col' in distinct_multi
        assert 'cat_col' in distinct_multi
        assert set(distinct_multi['bool_col']) == {True, False}
        assert set(distinct_multi['cat_col']) == {'cat1', 'cat2'}
        
        # Sin columnas especificadas (todas)
        distinct_all = backend.distinct()
        assert len(distinct_all) == 5
        
    def test_missings_method(self, df_with_missing):
        """Test detección de valores faltantes."""
        backend = DataBackendPandas(df_with_missing)
        
        # Contar missings por columna
        missing_counts = backend.missings()
        assert missing_counts['col1'] == 1
        assert missing_counts['col2'] == 1
        assert missing_counts['col3'] == 1
        
        # Para columnas específicas
        missing_subset = backend.missings(['col1'])
        assert 'col1' in missing_subset
        assert missing_subset['col1'] == 1
        
    def test_with_categorical_data(self):
        """Test manejo de datos categóricos."""
        df = pd.DataFrame({
            'cat': pd.Categorical(['a', 'b', 'a', 'c', 'b']),
            'ordered_cat': pd.Categorical(['low', 'med', 'high', 'low', 'med'], 
                                         categories=['low', 'med', 'high'], 
                                         ordered=True),
            'num': [1, 2, 3, 4, 5]
        })
        
        backend = DataBackendPandas(df)
        
        # Verificar que se mantienen los tipos
        retrieved = backend.data()
        assert isinstance(retrieved['cat'].dtype, pd.CategoricalDtype)
        assert isinstance(retrieved['ordered_cat'].dtype, pd.CategoricalDtype)
        assert retrieved['ordered_cat'].cat.ordered
        
    def test_with_datetime_data(self):
        """Test manejo de datos datetime."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'value': [10, 20, 30, 40, 50]
        })
        
        backend = DataBackendPandas(df)
        retrieved = backend.data()
        
        assert pd.api.types.is_datetime64_any_dtype(retrieved['date'])
        assert retrieved['date'].iloc[0] == pd.Timestamp('2023-01-01')
        
    def test_invalid_format(self, simple_df):
        """Test formato inválido genera error."""
        backend = DataBackendPandas(simple_df)
        
        with pytest.raises(ValueError, match="Unsupported format"):
            backend.data(data_format='invalid_format')
            
    def test_invalid_columns(self, simple_df):
        """Test columnas inválidas."""
        backend = DataBackendPandas(simple_df)
        
        # Columna no existente debería generar KeyError
        with pytest.raises(KeyError):
            backend.data(cols=['nonexistent_column'])
            
    def test_invalid_rows(self, simple_df):
        """Test filas inválidas."""
        backend = DataBackendPandas(simple_df)
        
        # Índices fuera de rango deberían ser ignorados por pandas
        result = backend.data(rows=[0, 10, 20])  
        assert len(result) == 1  # Solo el índice 0 existe


class TestDataBackendNumPy:
    """Tests completos para DataBackendNumPy."""
    
    @pytest.fixture
    def simple_array(self):
        """Array simple para tests."""
        return np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
    
    @pytest.fixture
    def mixed_array(self):
        """Array con tipos mixtos."""
        return np.array([
            [1.5, 2.5, 3.5],
            [4.5, 5.5, 6.5],
            [7.5, 8.5, 9.5]
        ], dtype=np.float32)
    
    def test_initialization(self, simple_array):
        """Test inicialización con array."""
        backend = DataBackendNumPy(simple_array)
        
        assert backend.nrow == 4
        assert backend.ncol == 3
        assert backend.colnames == ['V1', 'V2', 'V3']
        assert backend.rownames == [0, 1, 2, 3]
        
    def test_initialization_with_colnames(self, simple_array):
        """Test inicialización con nombres de columnas personalizados."""
        colnames = ['col_a', 'col_b', 'col_c']
        backend = DataBackendNumPy(simple_array, colnames=colnames)
        
        assert backend.colnames == colnames
        
    def test_initialization_wrong_colnames(self, simple_array):
        """Test error con número incorrecto de nombres de columnas."""
        with pytest.raises(ValueError, match="Number of column names"):
            DataBackendNumPy(simple_array, colnames=['a', 'b'])  # Solo 2 nombres para 3 columnas
            
    def test_formats_property(self, simple_array):
        """Test formatos soportados."""
        backend = DataBackendNumPy(simple_array)
        formats = backend.formats
        
        assert 'array' in formats
        assert 'dataframe' in formats
        assert 'dict' in formats
        
    def test_data_retrieval_all_formats(self, simple_array):
        """Test obtención de datos en todos los formatos."""
        backend = DataBackendNumPy(simple_array)
        
        # Array (default)
        arr = backend.data()
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, simple_array)
        
        # DataFrame
        df = backend.data(data_format='dataframe')
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 3)
        assert list(df.columns) == ['V1', 'V2', 'V3']
        
        # Dict
        d = backend.data(data_format='dict')
        assert isinstance(d, dict)
        assert set(d.keys()) == {'V1', 'V2', 'V3'}
        assert all(len(v) == 4 for v in d.values())
        
    def test_data_subset_selection(self, simple_array):
        """Test selección de subconjuntos."""
        backend = DataBackendNumPy(simple_array)
        
        # Seleccionar columnas por nombre
        subset_cols = backend.data(cols=['V1', 'V3'])
        assert subset_cols.shape == (4, 2)
        np.testing.assert_array_equal(subset_cols[:, 0], simple_array[:, 0])
        np.testing.assert_array_equal(subset_cols[:, 1], simple_array[:, 2])
        
        # Seleccionar filas
        subset_rows = backend.data(rows=[0, 2])
        assert subset_rows.shape == (2, 3)
        np.testing.assert_array_equal(subset_rows[0], simple_array[0])
        np.testing.assert_array_equal(subset_rows[1], simple_array[2])
        
        # Seleccionar ambos
        subset_both = backend.data(rows=[1, 3], cols=['V2'])
        assert subset_both.shape == (2, 1)
        assert subset_both[0, 0] == 5
        assert subset_both[1, 0] == 11
        
    def test_head_method(self, simple_array):
        """Test método head."""
        backend = DataBackendNumPy(simple_array)
        
        # Default n=6
        head_default = backend.head()
        assert isinstance(head_default, pd.DataFrame)
        assert len(head_default) == 4  # Solo hay 4 filas
        
        # Custom n
        head_2 = backend.head(2)
        assert len(head_2) == 2
        np.testing.assert_array_equal(head_2.values, simple_array[:2])
        
    def test_distinct_values(self):
        """Test valores únicos."""
        arr = np.array([
            [1, 1, 3],
            [1, 2, 3],
            [2, 2, 3],
            [2, 1, 4]
        ])
        backend = DataBackendNumPy(arr)
        
        distinct = backend.distinct(['V1', 'V3'])
        assert set(distinct['V1']) == {1, 2}
        assert set(distinct['V3']) == {3, 4}
        
    def test_missings_with_nan(self):
        """Test detección de valores faltantes (NaN)."""
        arr = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, np.nan],
            [np.nan, 11.0, 12.0]
        ])
        backend = DataBackendNumPy(arr)
        
        missings = backend.missings()
        assert missings['V1'] == 1
        assert missings['V2'] == 1
        assert missings['V3'] == 1
        
    def test_with_different_dtypes(self):
        """Test con diferentes tipos de datos."""
        # Enteros
        int_arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        backend_int = DataBackendNumPy(int_arr)
        assert backend_int.data().dtype == np.int32
        
        # Flotantes
        float_arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        backend_float = DataBackendNumPy(float_arr)
        assert backend_float.data().dtype == np.float64
        
        # Booleanos
        bool_arr = np.array([[True, False], [False, True]], dtype=bool)
        backend_bool = DataBackendNumPy(bool_arr)
        assert backend_bool.data().dtype == bool
        
    def test_1d_array_handling(self):
        """Test manejo de arrays 1D."""
        arr_1d = np.array([1, 2, 3, 4, 5])
        
        # Debería convertirse a 2D (columna)
        backend = DataBackendNumPy(arr_1d)
        assert backend.nrow == 5
        assert backend.ncol == 1
        assert backend.colnames == ['V1']
        
    def test_empty_array(self):
        """Test con array vacío."""
        empty_arr = np.array([]).reshape(0, 3)
        backend = DataBackendNumPy(empty_arr, colnames=['a', 'b', 'c'])
        
        assert backend.nrow == 0
        assert backend.ncol == 3
        assert len(backend.data()) == 0


class TestDataBackendCbind:
    """Tests para DataBackendCbind (combinación por columnas)."""
    
    @pytest.fixture
    def backend1(self):
        """Primer backend para combinar."""
        df1 = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        return DataBackendPandas(df1)
    
    @pytest.fixture
    def backend2(self):
        """Segundo backend para combinar."""
        df2 = pd.DataFrame({
            'c': [7, 8, 9],
            'd': [10, 11, 12]
        })
        return DataBackendPandas(df2)
    
    def test_cbind_basic(self, backend1, backend2):
        """Test combinación básica por columnas."""
        combined = DataBackendCbind(backend1, backend2)
        
        assert combined.nrow == 3
        assert combined.ncol == 4
        assert combined.colnames == ['a', 'b', 'c', 'd']
        
        # Verificar datos
        data = combined.data()
        assert data.shape == (3, 4)
        assert list(data.iloc[0]) == [1, 4, 7, 10]
        
    def test_cbind_mismatched_rows(self, backend1):
        """Test error cuando los backends tienen diferente número de filas."""
        df2 = pd.DataFrame({
            'c': [7, 8, 9, 10],  # 4 filas en lugar de 3
            'd': [11, 12, 13, 14]
        })
        backend2 = DataBackendPandas(df2)
        
        with pytest.raises(ValueError, match="same number of rows"):
            DataBackendCbind(backend1, backend2)
            
    def test_cbind_duplicate_columns(self, backend1):
        """Test manejo de columnas duplicadas."""
        df2 = pd.DataFrame({
            'a': [7, 8, 9],  # Columna 'a' duplicada
            'c': [10, 11, 12]
        })
        backend2 = DataBackendPandas(df2)
        
        # Debería manejar columnas duplicadas de alguna manera
        combined = DataBackendCbind(backend1, backend2)
        assert combined.ncol == 4  # Todas las columnas
        
    def test_cbind_data_retrieval(self, backend1, backend2):
        """Test obtención de datos del backend combinado."""
        combined = DataBackendCbind(backend1, backend2)
        
        # Obtener subconjunto de columnas
        subset = combined.data(cols=['a', 'c'])
        assert subset.shape == (3, 2)
        assert list(subset.columns) == ['a', 'c']
        
        # Obtener subconjunto de filas
        subset_rows = combined.data(rows=[0, 2])
        assert subset_rows.shape == (2, 4)
        
    def test_cbind_formats(self, backend1, backend2):
        """Test formatos soportados por cbind."""
        combined = DataBackendCbind(backend1, backend2)
        
        # DataFrame
        df = combined.data(data_format='dataframe')
        assert isinstance(df, pd.DataFrame)
        
        # Array
        arr = combined.data(data_format='array')
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4)
        
        # Dict
        d = combined.data(data_format='dict')
        assert isinstance(d, dict)
        assert len(d) == 4
        
    def test_cbind_with_numpy_backend(self):
        """Test cbind con NumPy backends."""
        arr1 = np.array([[1, 2], [3, 4], [5, 6]])
        arr2 = np.array([[7, 8], [9, 10], [11, 12]])
        
        backend1 = DataBackendNumPy(arr1, colnames=['a', 'b'])
        backend2 = DataBackendNumPy(arr2, colnames=['c', 'd'])
        
        combined = DataBackendCbind(backend1, backend2)
        
        assert combined.ncol == 4
        assert combined.colnames == ['a', 'b', 'c', 'd']
        
        data = combined.data(data_format='array')
        assert data.shape == (3, 4)
        np.testing.assert_array_equal(data[:, 0], [1, 3, 5])
        np.testing.assert_array_equal(data[:, 3], [8, 10, 12])


class TestDataBackendRbind:
    """Tests para DataBackendRbind (combinación por filas)."""
    
    @pytest.fixture
    def backend1(self):
        """Primer backend para combinar."""
        df1 = pd.DataFrame({
            'a': [1, 2],
            'b': [3, 4],
            'c': [5, 6]
        })
        return DataBackendPandas(df1)
    
    @pytest.fixture
    def backend2(self):
        """Segundo backend para combinar."""
        df2 = pd.DataFrame({
            'a': [7, 8],
            'b': [9, 10],
            'c': [11, 12]
        })
        return DataBackendPandas(df2)
    
    def test_rbind_basic(self, backend1, backend2):
        """Test combinación básica por filas."""
        combined = DataBackendRbind(backend1, backend2)
        
        assert combined.nrow == 4  # 2 + 2 filas
        assert combined.ncol == 3
        assert combined.colnames == ['a', 'b', 'c']
        
        # Verificar datos
        data = combined.data()
        assert data.shape == (4, 3)
        assert list(data.iloc[0]) == [1, 3, 5]
        assert list(data.iloc[3]) == [8, 10, 12]
        
    def test_rbind_mismatched_columns(self, backend1):
        """Test error cuando los backends tienen diferentes columnas."""
        df2 = pd.DataFrame({
            'a': [7, 8],
            'b': [9, 10],
            'd': [11, 12]  # Columna 'd' en lugar de 'c'
        })
        backend2 = DataBackendPandas(df2)
        
        with pytest.raises(ValueError, match="same columns"):
            DataBackendRbind(backend1, backend2)
            
    def test_rbind_data_retrieval(self, backend1, backend2):
        """Test obtención de datos del backend combinado."""
        combined = DataBackendRbind(backend1, backend2)
        
        # Obtener subconjunto de columnas
        subset = combined.data(cols=['a', 'c'])
        assert subset.shape == (4, 2)
        assert list(subset.columns) == ['a', 'c']
        
        # Obtener subconjunto de filas
        subset_rows = combined.data(rows=[0, 3])
        assert subset_rows.shape == (2, 3)
        assert list(subset_rows.iloc[0]) == [1, 3, 5]
        assert list(subset_rows.iloc[1]) == [8, 10, 12]
        
    def test_rbind_formats(self, backend1, backend2):
        """Test formatos soportados."""
        combined = DataBackendRbind(backend1, backend2)
        
        # DataFrame
        df = combined.data(data_format='dataframe')
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 3)
        
        # Array
        arr = combined.data(data_format='array')
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 3)
        
        # Dict
        d = combined.data(data_format='dict')
        assert isinstance(d, dict)
        assert all(len(v) == 4 for v in d.values())
        
    def test_rbind_with_numpy_backend(self):
        """Test rbind con NumPy backends."""
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[7, 8, 9], [10, 11, 12]])
        
        backend1 = DataBackendNumPy(arr1, colnames=['a', 'b', 'c'])
        backend2 = DataBackendNumPy(arr2, colnames=['a', 'b', 'c'])
        
        combined = DataBackendRbind(backend1, backend2)
        
        assert combined.nrow == 4
        assert combined.ncol == 3
        
        data = combined.data(data_format='array')
        assert data.shape == (4, 3)
        np.testing.assert_array_equal(data[0], [1, 2, 3])
        np.testing.assert_array_equal(data[3], [10, 11, 12])
        
    def test_rbind_multiple_backends(self):
        """Test rbind con múltiples backends."""
        df1 = pd.DataFrame({'a': [1], 'b': [2]})
        df2 = pd.DataFrame({'a': [3], 'b': [4]})
        df3 = pd.DataFrame({'a': [5], 'b': [6]})
        
        b1 = DataBackendPandas(df1)
        b2 = DataBackendPandas(df2)
        b3 = DataBackendPandas(df3)
        
        # Combinar b1 y b2
        combined12 = DataBackendRbind(b1, b2)
        # Luego combinar con b3
        combined_all = DataBackendRbind(combined12, b3)
        
        assert combined_all.nrow == 3
        data = combined_all.data()
        assert list(data['a']) == [1, 3, 5]


# Intentar importar backends opcionales
try:
    from mlpy.backends import DataBackendDask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    from mlpy.backends import DataBackendVaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not installed")
class TestDataBackendDask:
    """Tests para DataBackendDask."""
    
    @pytest.fixture
    def dask_df(self):
        """Crear un DataFrame de Dask para tests."""
        import dask.dataframe as dd
        
        df = pd.DataFrame({
            'x': range(100),
            'y': range(100, 200),
            'z': ['a', 'b'] * 50
        })
        return dd.from_pandas(df, npartitions=4)
    
    def test_initialization(self, dask_df):
        """Test inicialización con Dask DataFrame."""
        backend = DataBackendDask(dask_df)
        
        assert backend.nrow == 100
        assert backend.ncol == 3
        assert backend.colnames == ['x', 'y', 'z']
        
    def test_data_retrieval(self, dask_df):
        """Test obtención de datos."""
        backend = DataBackendDask(dask_df)
        
        # Como DataFrame de pandas (compute)
        df = backend.data()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 3)
        
        # Como array
        arr = backend.data(data_format='array')
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (100, 3)
        
    def test_head_method(self, dask_df):
        """Test método head."""
        backend = DataBackendDask(dask_df)
        
        head = backend.head(5)
        assert isinstance(head, pd.DataFrame)
        assert len(head) == 5
        
    def test_lazy_evaluation(self, dask_df):
        """Test que Dask mantiene evaluación lazy."""
        backend = DataBackendDask(dask_df)
        
        # Esto no debería computar hasta que se llame .data()
        assert hasattr(backend._data, 'compute')
        
    def test_partitions(self, dask_df):
        """Test manejo de particiones."""
        backend = DataBackendDask(dask_df)
        
        # Verificar que se mantienen las particiones
        assert backend._data.npartitions == 4


@pytest.mark.skipif(not VAEX_AVAILABLE, reason="Vaex not installed")
class TestDataBackendVaex:
    """Tests para DataBackendVaex."""
    
    @pytest.fixture
    def vaex_df(self):
        """Crear un DataFrame de Vaex para tests."""
        import vaex
        
        df = pd.DataFrame({
            'x': range(100),
            'y': np.random.randn(100),
            'z': ['cat1', 'cat2', 'cat3'] * 33 + ['cat1']
        })
        return vaex.from_pandas(df)
    
    def test_initialization(self, vaex_df):
        """Test inicialización con Vaex DataFrame."""
        backend = DataBackendVaex(vaex_df)
        
        assert backend.nrow == 100
        assert backend.ncol == 3
        assert backend.colnames == ['x', 'y', 'z']
        
    def test_data_retrieval(self, vaex_df):
        """Test obtención de datos."""
        backend = DataBackendVaex(vaex_df)
        
        # Como DataFrame de pandas
        df = backend.data()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 3)
        
        # Como array
        arr = backend.data(data_format='array')
        assert isinstance(arr, np.ndarray)
        
    def test_memory_efficiency(self, vaex_df):
        """Test que Vaex es eficiente en memoria."""
        backend = DataBackendVaex(vaex_df)
        
        # Vaex no carga todo en memoria hasta que se necesita
        assert hasattr(backend._data, 'to_pandas_df')
        
    def test_expressions(self, vaex_df):
        """Test que se pueden usar expresiones de Vaex."""
        backend = DataBackendVaex(vaex_df)
        
        # Obtener solo ciertas columnas
        subset = backend.data(cols=['x', 'y'])
        assert subset.shape[1] == 2


class TestBackendIntegration:
    """Tests de integración entre diferentes backends."""
    
    def test_pandas_to_numpy_conversion(self):
        """Test conversión entre backends."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pandas_backend = DataBackendPandas(df)
        
        # Obtener como array
        arr = pandas_backend.data(data_format='array')
        
        # Crear NumPy backend con el array
        numpy_backend = DataBackendNumPy(arr, colnames=['a', 'b'])
        
        # Deberían tener los mismos datos
        assert numpy_backend.nrow == pandas_backend.nrow
        assert numpy_backend.ncol == pandas_backend.ncol
        np.testing.assert_array_equal(
            numpy_backend.data(),
            pandas_backend.data(data_format='array')
        )
        
    def test_combined_backends_chain(self):
        """Test cadena de backends combinados."""
        # Crear backends individuales
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        df3 = pd.DataFrame({'c': [9, 10], 'd': [11, 12]})
        
        b1 = DataBackendPandas(df1)
        b2 = DataBackendPandas(df2)
        b3 = DataBackendPandas(df3)
        
        # Combinar por filas primero
        rbind = DataBackendRbind(b1, b2)
        assert rbind.nrow == 4
        
        # Luego por columnas
        final = DataBackendCbind(rbind, b3)
        assert final.shape == (4, 4)
        assert final.colnames == ['a', 'b', 'c', 'd']
        
        # Verificar datos finales
        data = final.data()
        assert data.iloc[0, 0] == 1  # Primera fila, primera columna
        assert data.iloc[3, 3] == 12  # Última fila, última columna
        
    def test_backend_hash_consistency(self):
        """Test que el hash es consistente."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        backend1 = DataBackendPandas(df)
        backend2 = DataBackendPandas(df.copy())
        
        # Los hashes deberían ser iguales para datos iguales
        assert backend1.hash == backend2.hash
        
        # Cambiar datos debería cambiar el hash
        df_modified = df.copy()
        df_modified.loc[0, 'a'] = 999
        backend3 = DataBackendPandas(df_modified)
        assert backend1.hash != backend3.hash


class TestBackendPerformance:
    """Tests de rendimiento y casos límite."""
    
    def test_large_dataset_handling(self):
        """Test manejo de datasets grandes."""
        # Crear dataset grande
        n_rows = 10000
        n_cols = 50
        
        data = np.random.randn(n_rows, n_cols)
        colnames = [f'col_{i}' for i in range(n_cols)]
        
        # Test con NumPy backend
        backend_np = DataBackendNumPy(data, colnames=colnames)
        assert backend_np.nrow == n_rows
        assert backend_np.ncol == n_cols
        
        # Test con Pandas backend
        df = pd.DataFrame(data, columns=colnames)
        backend_pd = DataBackendPandas(df)
        assert backend_pd.nrow == n_rows
        
        # Test subset selection en dataset grande
        subset = backend_pd.data(rows=list(range(0, 100, 10)), cols=colnames[:5])
        assert subset.shape == (10, 5)
        
    def test_memory_efficiency(self):
        """Test eficiencia de memoria."""
        # Crear dataset con tipos específicos para ahorrar memoria
        df = pd.DataFrame({
            'int8_col': np.array(range(100), dtype=np.int8),
            'int16_col': np.array(range(100), dtype=np.int16),
            'float32_col': np.array(np.random.randn(100), dtype=np.float32),
            'bool_col': np.random.choice([True, False], 100)
        })
        
        backend = DataBackendPandas(df)
        
        # Verificar que se mantienen los tipos
        retrieved = backend.data()
        assert retrieved['int8_col'].dtype == np.int8
        assert retrieved['float32_col'].dtype == np.float32
        assert retrieved['bool_col'].dtype == bool
        
    def test_edge_cases(self):
        """Test casos límite."""
        # DataFrame con una sola fila
        df_single_row = pd.DataFrame({'a': [1], 'b': [2]})
        backend_single = DataBackendPandas(df_single_row)
        assert backend_single.nrow == 1
        
        # DataFrame con una sola columna
        df_single_col = pd.DataFrame({'only_col': [1, 2, 3, 4, 5]})
        backend_single_col = DataBackendPandas(df_single_col)
        assert backend_single_col.ncol == 1
        
        # DataFrame vacío (0 filas pero con columnas)
        df_empty = pd.DataFrame(columns=['a', 'b', 'c'])
        backend_empty = DataBackendPandas(df_empty)
        assert backend_empty.nrow == 0
        assert backend_empty.ncol == 3
        
        # Array con valores extremos
        extreme_arr = np.array([[np.inf, -np.inf], [np.finfo(float).max, np.finfo(float).min]])
        backend_extreme = DataBackendNumPy(extreme_arr)
        assert backend_extreme.nrow == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])