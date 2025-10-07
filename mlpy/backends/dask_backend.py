"""Dask backend for handling large datasets with MLPY.

This backend allows MLPY to work with Dask DataFrames, enabling
out-of-core computation for datasets that don't fit in memory.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any, Dict, Tuple
import warnings

try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

from .base import DataBackend


class DataBackendDask(DataBackend):
    """Data backend for Dask DataFrames.
    
    This backend enables MLPY to work with large datasets using Dask,
    providing lazy evaluation and distributed computing capabilities.
    
    Parameters
    ----------
    data : dask.dataframe.DataFrame
        The Dask DataFrame to wrap.
    primary_key : str, optional
        Column name to use as primary key.
    
    Attributes
    ----------
    supports_lazy : bool
        True, as Dask supports lazy evaluation.
    supports_out_of_core : bool
        True, as Dask supports out-of-core computation.
    """
    
    def __init__(self, data: "dd.DataFrame", primary_key: Optional[str] = None):
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is not installed. Install it with: pip install dask[dataframe]"
            )
            
        if not isinstance(data, dd.DataFrame):
            raise TypeError(f"Expected dask.dataframe.DataFrame, got {type(data)}")
            
        self._data = data
        self._primary_key = primary_key
        
        # Cache some metadata to avoid repeated computation
        self._ncol = len(data.columns)
        self._colnames = list(data.columns)
        self._coltypes = dict(zip(data.columns, data.dtypes))
        
        # These will be computed lazily
        self._nrow = None
        self._row_ids = None
        
    @property
    def nrow(self) -> int:
        """Number of rows (computed lazily)."""
        if self._nrow is None:
            self._nrow = len(self._data)
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
        """Row names (index values)."""
        if self._row_ids is None:
            # For large datasets, we might not want to compute all row IDs
            warnings.warn(
                "Computing row IDs for large Dask DataFrame. "
                "This may be slow and memory-intensive."
            )
            self._row_ids = self._data.index.compute().tolist()
        return self._row_ids
        
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
        
    def data(
        self,
        rows: Optional[Union[List[int], slice]] = None,
        cols: Optional[Union[List[str], List[int]]] = None,
        as_numpy: bool = False
    ) -> Union[pd.DataFrame, np.ndarray, "dd.DataFrame"]:
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
            If as_numpy=False and no row selection, returns Dask DataFrame.
            Otherwise computes and returns pandas DataFrame or numpy array.
        """
        # Column selection
        if cols is not None:
            if isinstance(cols[0], int):
                cols = [self._colnames[i] for i in cols]
            data = self._data[cols]
        else:
            data = self._data
            
        # Row selection
        if rows is not None:
            if isinstance(rows, slice):
                # Dask supports iloc with slices
                data = data.iloc[rows]
            else:
                # For specific indices, we need to compute
                warnings.warn(
                    "Row selection by indices requires computation. "
                    "Consider using slices for better performance."
                )
                data = data.compute().iloc[rows]
                # Convert back to Dask if it's still large
                if len(data) > 10000:
                    data = dd.from_pandas(data, npartitions=max(1, len(data) // 10000))
                    
        if as_numpy:
            # This triggers computation
            if isinstance(data, dd.DataFrame):
                return data.compute().values
            return data.values
            
        # Return Dask DataFrame for lazy evaluation
        if isinstance(data, dd.DataFrame):
            return data
        # Or pandas DataFrame if already computed
        return data
        
    def distinct(
        self,
        cols: Union[str, List[str]],
        as_numpy: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Get distinct values for columns.
        
        This operation requires computation.
        """
        if isinstance(cols, str):
            cols = [cols]
            
        # This triggers computation
        distinct_values = self._data[cols].drop_duplicates().compute()
        
        if as_numpy:
            return distinct_values.values
        return distinct_values
        
    def missings(
        self,
        cols: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, int]:
        """Count missing values per column.
        
        This operation requires computation.
        """
        if cols is None:
            cols = self._colnames
        elif isinstance(cols, str):
            cols = [cols]
            
        # Compute missing counts
        missing_counts = {}
        for col in cols:
            missing_counts[col] = self._data[col].isna().sum().compute()
            
        return missing_counts
        
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows as pandas DataFrame."""
        return self._data.head(n)
        
    def tail(self, n: int = 5) -> pd.DataFrame:
        """Get last n rows as pandas DataFrame."""
        return self._data.tail(n)
        
    def set_col_names(self, old: List[str], new: List[str]) -> None:
        """Rename columns."""
        rename_dict = dict(zip(old, new))
        self._data = self._data.rename(columns=rename_dict)
        self._colnames = list(self._data.columns)
        self._coltypes = dict(zip(self._data.columns, self._data.dtypes))
        
    def set_row_names(self, row_names: List[Any]) -> None:
        """Set row names (index)."""
        # This requires computation and may not be efficient for large datasets
        warnings.warn(
            "Setting row names on Dask DataFrame requires computation. "
            "This may be slow for large datasets."
        )
        self._data = self._data.assign(__index__=row_names).set_index('__index__')
        self._row_ids = row_names
        
    def cbind(self, other: "DataBackend") -> "DataBackendDask":
        """Column-bind with another backend."""
        if isinstance(other, DataBackendDask):
            # Both are Dask DataFrames
            combined = dd.concat([self._data, other._data], axis=1)
        else:
            # Convert other to Dask
            other_df = other.data()
            if not isinstance(other_df, pd.DataFrame):
                other_df = pd.DataFrame(other_df)
            other_dask = dd.from_pandas(other_df, npartitions=self._data.npartitions)
            combined = dd.concat([self._data, other_dask], axis=1)
            
        return DataBackendDask(combined)
        
    def rbind(self, other: "DataBackend") -> "DataBackendDask":
        """Row-bind with another backend."""
        if isinstance(other, DataBackendDask):
            # Both are Dask DataFrames
            combined = dd.concat([self._data, other._data], axis=0)
        else:
            # Convert other to Dask
            other_df = other.data()
            if not isinstance(other_df, pd.DataFrame):
                other_df = pd.DataFrame(other_df)
            other_dask = dd.from_pandas(other_df, npartitions=1)
            combined = dd.concat([self._data, other_dask], axis=0)
            
        return DataBackendDask(combined)
        
    def compute(self) -> pd.DataFrame:
        """Compute and return as pandas DataFrame.
        
        Warning: This loads the entire dataset into memory.
        """
        warnings.warn(
            "Computing entire Dask DataFrame. "
            "This may consume significant memory."
        )
        return self._data.compute()
        
    def persist(self) -> "DataBackendDask":
        """Persist the DataFrame in distributed memory.
        
        This is useful for iterative algorithms that reuse the same data.
        """
        self._data = self._data.persist()
        return self
        
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        npartitions: Optional[int] = None,
        chunksize: Optional[int] = None
    ) -> "DataBackendDask":
        """Create from pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input pandas DataFrame.
        npartitions : int, optional
            Number of partitions. If not specified, uses chunksize.
        chunksize : int, optional
            Rows per partition. Default is 100,000.
            
        Returns
        -------
        DataBackendDask
            Dask backend wrapping the data.
        """
        if npartitions is None:
            if chunksize is None:
                chunksize = 100_000
            npartitions = max(1, len(df) // chunksize)
            
        dask_df = dd.from_pandas(df, npartitions=npartitions)
        return cls(dask_df)
        
    @classmethod
    def read_csv(
        cls,
        filepath: str,
        **kwargs
    ) -> "DataBackendDask":
        """Read CSV file(s) as Dask DataFrame.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file(s). Can include wildcards.
        **kwargs
            Additional arguments passed to dask.dataframe.read_csv.
            
        Returns
        -------
        DataBackendDask
            Dask backend wrapping the data.
        """
        dask_df = dd.read_csv(filepath, **kwargs)
        return cls(dask_df)
        
    @classmethod
    def read_parquet(
        cls,
        filepath: str,
        **kwargs
    ) -> "DataBackendDask":
        """Read Parquet file(s) as Dask DataFrame.
        
        Parameters
        ----------
        filepath : str
            Path to Parquet file(s).
        **kwargs
            Additional arguments passed to dask.dataframe.read_parquet.
            
        Returns
        -------
        DataBackendDask
            Dask backend wrapping the data.
        """
        dask_df = dd.read_parquet(filepath, **kwargs)
        return cls(dask_df)
        
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<DataBackendDask: {self.nrow} x {self.ncol}, "
            f"{self._data.npartitions} partitions>"
        )