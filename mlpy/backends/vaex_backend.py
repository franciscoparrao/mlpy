"""Vaex backend for handling large datasets with MLPY.

This backend allows MLPY to work with Vaex DataFrames, enabling
memory-mapped and out-of-core computation for massive datasets.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any, Dict, Tuple
import warnings

try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False
    vaex = None

from .base import DataBackend


class DataBackendVaex(DataBackend):
    """Data backend for Vaex DataFrames.
    
    This backend enables MLPY to work with massive datasets using Vaex,
    providing memory-mapped operations and lazy evaluation.
    
    Parameters
    ----------
    data : vaex.DataFrame
        The Vaex DataFrame to wrap.
    primary_key : str, optional
        Column name to use as primary key.
    
    Attributes
    ----------
    supports_lazy : bool
        True, as Vaex supports lazy evaluation.
    supports_out_of_core : bool
        True, as Vaex supports out-of-core computation.
    supports_memory_mapping : bool
        True, as Vaex can memory-map files.
    """
    
    def __init__(self, data: "vaex.DataFrame", primary_key: Optional[str] = None):
        if not VAEX_AVAILABLE:
            raise ImportError(
                "Vaex is not installed. Install it with: pip install vaex"
            )
            
        if not isinstance(data, vaex.dataframe.DataFrame):
            raise TypeError(f"Expected vaex.DataFrame, got {type(data)}")
            
        self._data = data
        self._primary_key = primary_key
        
        # Cache metadata
        self._nrow = len(data)
        self._ncol = len(data.get_column_names())
        self._colnames = data.get_column_names()
        self._coltypes = {col: str(data[col].dtype) for col in self._colnames}
        
    @property
    def nrow(self) -> int:
        """Number of rows."""
        return self._nrow
        
    @property
    def ncol(self) -> int:
        """Number of columns."""
        return self._ncol
        
    @property
    def colnames(self) -> List[str]:
        """Column names."""
        return self._colnames.copy()
        
    @property
    def rownames(self) -> Optional[List[Any]]:
        """Row names (not typically used in Vaex)."""
        # Vaex doesn't have row names like pandas
        return None
        
    @property
    def primary_keys(self) -> List[str]:
        """Primary key columns."""
        if self._primary_key:
            return [self._primary_key]
        return []
        
    @property
    def supports_lazy(self) -> bool:
        """Whether this backend supports lazy evaluation."""
        return True
        
    @property
    def supports_out_of_core(self) -> bool:
        """Whether this backend supports out-of-core computation."""
        return True
        
    @property
    def supports_memory_mapping(self) -> bool:
        """Whether this backend supports memory-mapped files."""
        return True
        
    def data(
        self,
        rows: Optional[Union[List[int], slice]] = None,
        cols: Optional[Union[List[str], List[int]]] = None,
        as_numpy: bool = False
    ) -> Union[pd.DataFrame, np.ndarray, "vaex.DataFrame"]:
        """Get data subset.
        
        Parameters
        ----------
        rows : list of int or slice, optional
            Row indices to select.
        cols : list of str or int, optional
            Column names or indices to select.
        as_numpy : bool, default=False
            If True, return numpy array. This will trigger computation.
            
        Returns
        -------
        data : DataFrame or ndarray
            If as_numpy=False and no row selection, returns Vaex DataFrame.
            Otherwise computes and returns pandas DataFrame or numpy array.
        """
        data = self._data
        
        # Column selection
        if cols is not None:
            if isinstance(cols[0], int):
                cols = [self._colnames[i] for i in cols]
            data = data[cols]
            
        # Row selection
        if rows is not None:
            if isinstance(rows, slice):
                start = rows.start or 0
                stop = rows.stop or len(data)
                step = rows.step or 1
                data = data[start:stop:step]
            else:
                # Vaex supports fancy indexing
                data = data.take(rows)
                
        if as_numpy:
            # This triggers computation
            # For multiple columns, we need to combine them
            if isinstance(data, vaex.dataframe.DataFrame) and len(data.get_column_names()) > 1:
                arrays = [data[col].to_numpy() for col in data.get_column_names()]
                return np.column_stack(arrays)
            elif isinstance(data, vaex.dataframe.DataFrame):
                # Single column
                col = data.get_column_names()[0]
                return data[col].to_numpy()
            else:
                # Already computed
                return np.array(data)
                
        # Return Vaex DataFrame for lazy evaluation
        return data
        
    def distinct(
        self,
        cols: Union[str, List[str]],
        as_numpy: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Get distinct values for columns.
        
        This operation may require computation for large datasets.
        """
        if isinstance(cols, str):
            cols = [cols]
            
        # Vaex can compute unique values efficiently
        result_dict = {}
        for col in cols:
            unique_values = self._data[col].unique()
            result_dict[col] = unique_values
            
        if as_numpy:
            # Stack arrays
            arrays = [result_dict[col] for col in cols]
            if len(arrays) == 1:
                return arrays[0]
            else:
                # Need to handle different lengths
                max_len = max(len(arr) for arr in arrays)
                padded = []
                for arr in arrays:
                    if len(arr) < max_len:
                        # Pad with NaN
                        padded_arr = np.full(max_len, np.nan, dtype=object)
                        padded_arr[:len(arr)] = arr
                        padded.append(padded_arr)
                    else:
                        padded.append(arr)
                return np.column_stack(padded)
        else:
            # Return as pandas DataFrame
            return pd.DataFrame(result_dict)
        
    def missings(
        self,
        cols: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, int]:
        """Count missing values per column."""
        if cols is None:
            cols = self._colnames
        elif isinstance(cols, str):
            cols = [cols]
            
        missing_counts = {}
        for col in cols:
            # Vaex has efficient missing value counting
            missing_counts[col] = self._data[col].countmissing()
            
        return missing_counts
        
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows as pandas DataFrame."""
        return self._data.head(n).to_pandas_df()
        
    def tail(self, n: int = 5) -> pd.DataFrame:
        """Get last n rows as pandas DataFrame."""
        return self._data.tail(n).to_pandas_df()
        
    def set_col_names(self, old: List[str], new: List[str]) -> None:
        """Rename columns."""
        for old_name, new_name in zip(old, new):
            self._data.rename(old_name, new_name)
        self._colnames = self._data.get_column_names()
        self._coltypes = {col: str(self._data[col].dtype) for col in self._colnames}
        
    def set_row_names(self, row_names: List[Any]) -> None:
        """Set row names (not supported in Vaex)."""
        warnings.warn("Vaex does not support row names. This operation is ignored.")
        
    def cbind(self, other: "DataBackend") -> "DataBackendVaex":
        """Column-bind with another backend."""
        if isinstance(other, DataBackendVaex):
            # Both are Vaex DataFrames
            # Vaex doesn't have direct cbind, so we add columns
            result = self._data.copy()
            for col in other._data.get_column_names():
                result[col] = other._data[col]
        else:
            # Convert other to Vaex
            other_df = other.data()
            if not isinstance(other_df, pd.DataFrame):
                other_df = pd.DataFrame(other_df)
            other_vaex = vaex.from_pandas(other_df)
            
            result = self._data.copy()
            for col in other_vaex.get_column_names():
                result[col] = other_vaex[col]
                
        return DataBackendVaex(result)
        
    def rbind(self, other: "DataBackend") -> "DataBackendVaex":
        """Row-bind with another backend."""
        if isinstance(other, DataBackendVaex):
            # Both are Vaex DataFrames
            result = self._data.concat(other._data)
        else:
            # Convert other to Vaex
            other_df = other.data()
            if not isinstance(other_df, pd.DataFrame):
                other_df = pd.DataFrame(other_df)
            other_vaex = vaex.from_pandas(other_df)
            result = self._data.concat(other_vaex)
            
        return DataBackendVaex(result)
        
    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.
        
        Warning: This loads the entire dataset into memory.
        """
        warnings.warn(
            "Converting entire Vaex DataFrame to pandas. "
            "This may consume significant memory."
        )
        return self._data.to_pandas_df()
        
    def export_hdf5(self, path: str, **kwargs):
        """Export to HDF5 format for efficient storage."""
        self._data.export_hdf5(path, **kwargs)
        
    def export_arrow(self, path: str, **kwargs):
        """Export to Arrow format."""
        self._data.export_arrow(path, **kwargs)
        
    def export_parquet(self, path: str, **kwargs):
        """Export to Parquet format."""
        self._data.export_parquet(path, **kwargs)
        
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        copy_index: bool = True
    ) -> "DataBackendVaex":
        """Create from pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input pandas DataFrame.
        copy_index : bool, default=True
            Whether to copy the index as a column.
            
        Returns
        -------
        DataBackendVaex
            Vaex backend wrapping the data.
        """
        vaex_df = vaex.from_pandas(df, copy_index=copy_index)
        return cls(vaex_df)
        
    @classmethod
    def open(
        cls,
        path: str,
        convert: bool = True,
        **kwargs
    ) -> "DataBackendVaex":
        """Open various file formats.
        
        Parameters
        ----------
        path : str
            Path to file. Supports CSV, HDF5, Arrow, Parquet, etc.
        convert : bool, default=True
            Convert to HDF5 for better performance.
        **kwargs
            Additional arguments passed to vaex.open.
            
        Returns
        -------
        DataBackendVaex
            Vaex backend wrapping the data.
        """
        vaex_df = vaex.open(path, convert=convert, **kwargs)
        return cls(vaex_df)
        
    @classmethod
    def open_many(
        cls,
        paths: List[str],
        **kwargs
    ) -> "DataBackendVaex":
        """Open multiple files and concatenate.
        
        Parameters
        ----------
        paths : list of str
            Paths to files.
        **kwargs
            Additional arguments passed to vaex.open_many.
            
        Returns
        -------
        DataBackendVaex
            Vaex backend wrapping the concatenated data.
        """
        vaex_df = vaex.open_many(paths, **kwargs)
        return cls(vaex_df)
        
    def __repr__(self) -> str:
        """String representation."""
        return f"<DataBackendVaex: {self.nrow:,} x {self.ncol}>"