"""
NumPy array backend for MLPY.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from .base import DataBackend


class DataBackendNumPy(DataBackend):
    """
    DataBackend using NumPy arrays for storage.
    
    This backend is optimized for numeric data and provides
    efficient operations on homogeneous arrays.
    
    Parameters
    ----------
    data : np.ndarray or dict
        Either a 2D numpy array or a dict mapping column names to 1D arrays
    colnames : List[str], optional
        Column names. Required if data is a numpy array.
    primary_key : str, optional
        Name of the column to use as primary key
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        colnames: Optional[List[str]] = None,
        primary_key: Optional[str] = None,
        **kwargs
    ):
        self._colnames = colnames
        super().__init__(data=data, primary_key=primary_key, **kwargs)
    
    def _initialize_backend(self, data: Union[np.ndarray, Dict[str, np.ndarray]]) -> None:
        """Initialize the numpy backend."""
        if isinstance(data, dict):
            # Dictionary of arrays
            if not data:
                raise ValueError("Data dictionary cannot be empty")
            
            # Validate all arrays have same length
            lengths = [len(arr) for arr in data.values()]
            if len(set(lengths)) > 1:
                raise ValueError(f"All arrays must have same length, got {lengths}")
            
            # Store as structured array for efficiency
            self._colnames = list(data.keys())
            self._data = np.column_stack([data[col] for col in self._colnames])
            self._col_map = {col: i for i, col in enumerate(self._colnames)}
            
        elif isinstance(data, np.ndarray):
            # Single array
            if data.ndim != 2:
                raise ValueError(f"Expected 2D array, got {data.ndim}D")
            
            if self._colnames is None:
                # Generate column names
                self._colnames = [f"V{i+1}" for i in range(data.shape[1])]
            elif len(self._colnames) != data.shape[1]:
                raise ValueError(
                    f"Number of column names ({len(self._colnames)}) "
                    f"doesn't match array shape ({data.shape[1]})"
                )
            
            self._data = data.copy()
            self._col_map = {col: i for i, col in enumerate(self._colnames)}
            
        else:
            raise TypeError(f"Expected numpy array or dict, got {type(data)}")
        
        # Create row index
        self._row_index = np.arange(len(self._data))
        
        # Validate primary key
        if self._primary_key is not None:
            if self._primary_key not in self._colnames:
                raise ValueError(f"Primary key '{self._primary_key}' not found in columns")
            
            # Check uniqueness
            pk_col = self._col_map[self._primary_key]
            if len(np.unique(self._data[:, pk_col])) != len(self._data):
                raise ValueError(f"Primary key column '{self._primary_key}' contains duplicates")
    
    @property
    def nrow(self) -> int:
        """Number of rows."""
        return len(self._data)
    
    @property
    def ncol(self) -> int:
        """Number of columns."""
        return self._data.shape[1]
    
    @property
    def colnames(self) -> List[str]:
        """Column names."""
        return self._colnames.copy()
    
    @property
    def rownames(self) -> List[Union[str, int]]:
        """Row names/indices."""
        if self._primary_key:
            pk_col = self._col_map[self._primary_key]
            return self._data[:, pk_col].tolist()
        else:
            return self._row_index.tolist()
    
    def data(
        self, 
        rows: Optional[Sequence[Union[int, str]]] = None,
        cols: Optional[Sequence[str]] = None,
        data_format: str = "dataframe"
    ) -> Any:
        """Retrieve data in specified format."""
        # Select columns
        if cols is None:
            col_indices = list(range(self.ncol))
            selected_cols = self._colnames
        else:
            # Validate columns exist
            missing_cols = set(cols) - set(self._colnames)
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            
            col_indices = [self._col_map[col] for col in cols]
            selected_cols = list(cols)
        
        # Select rows
        if rows is None:
            row_mask = np.ones(self.nrow, dtype=bool)
        else:
            if self._primary_key:
                # Use primary key for selection
                pk_col = self._col_map[self._primary_key]
                pk_values = self._data[:, pk_col]
                row_mask = np.isin(pk_values, rows)
            else:
                # Use indices
                if all(isinstance(r, int) for r in rows):
                    row_mask = np.zeros(self.nrow, dtype=bool)
                    valid_rows = [r for r in rows if 0 <= r < self.nrow]
                    row_mask[valid_rows] = True
                else:
                    raise ValueError("String row names not supported without primary key")
        
        # Extract data
        subset_data = self._data[row_mask][:, col_indices]
        
        # Convert to requested format
        if data_format == "dataframe":
            return pd.DataFrame(subset_data, columns=selected_cols)
        elif data_format == "array":
            return subset_data
        elif data_format == "dict":
            return {col: subset_data[:, i] for i, col in enumerate(selected_cols)}
        else:
            raise ValueError(f"Unknown data format: {data_format}")
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows as DataFrame."""
        n_rows = min(n, self.nrow)
        return pd.DataFrame(
            self._data[:n_rows],
            columns=self._colnames
        )
    
    def distinct(
        self, 
        cols: Sequence[str],
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """Get distinct values for columns."""
        result = {}
        
        # Get row mask
        if rows is None:
            row_mask = np.ones(self.nrow, dtype=bool)
        else:
            if self._primary_key:
                pk_col = self._col_map[self._primary_key]
                pk_values = self._data[:, pk_col]
                row_mask = np.isin(pk_values, rows)
            else:
                if all(isinstance(r, int) for r in rows):
                    row_mask = np.zeros(self.nrow, dtype=bool)
                    valid_rows = [r for r in rows if 0 <= r < self.nrow]
                    row_mask[valid_rows] = True
                else:
                    raise ValueError("String row names not supported without primary key")
        
        # Get unique values for each column
        for col in cols:
            if col not in self._col_map:
                raise ValueError(f"Column '{col}' not found")
            
            col_idx = self._col_map[col]
            col_data = self._data[row_mask, col_idx]
            
            # Remove NaN values
            if np.issubdtype(col_data.dtype, np.floating):
                col_data = col_data[~np.isnan(col_data)]
            
            unique_vals = np.unique(col_data)
            result[col] = unique_vals
        
        return result
    
    def missings(
        self, 
        cols: Optional[Sequence[str]] = None,
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> int:
        """Count missing values (NaN for numeric columns)."""
        # Select columns
        if cols is None:
            col_indices = list(range(self.ncol))
        else:
            col_indices = [self._col_map[col] for col in cols]
        
        # Select rows
        if rows is None:
            row_mask = np.ones(self.nrow, dtype=bool)
        else:
            if self._primary_key:
                pk_col = self._col_map[self._primary_key]
                pk_values = self._data[:, pk_col]
                row_mask = np.isin(pk_values, rows)
            else:
                if all(isinstance(r, int) for r in rows):
                    row_mask = np.zeros(self.nrow, dtype=bool)
                    valid_rows = [r for r in rows if 0 <= r < self.nrow]
                    row_mask[valid_rows] = True
                else:
                    raise ValueError("String row names not supported without primary key")
        
        # Count NaN values
        subset = self._data[row_mask][:, col_indices]
        if np.issubdtype(subset.dtype, np.floating):
            return int(np.isnan(subset).sum())
        else:
            # Non-float types don't have NaN
            return 0
    
    def _infer_col_type(self, col: str) -> str:
        """Infer column type from numpy dtype."""
        col_idx = self._col_map[col]
        col_data = self._data[:, col_idx]
        dtype = col_data.dtype
        
        if np.issubdtype(dtype, np.bool_):
            return "logical"
        elif np.issubdtype(dtype, np.integer):
            return "integer"
        elif np.issubdtype(dtype, np.floating):
            return "numeric"
        elif np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_):
            return "character"
        else:
            return "unknown"
    
    def _get_params_for_hash(self) -> Dict[str, Any]:
        """Include data content in hash calculation."""
        params = super()._get_params_for_hash()
        
        # Add hash of data content
        # Use view as bytes for consistent hashing
        data_bytes = self._data.tobytes()
        data_hash = hash(data_bytes)
        params["data_hash"] = str(data_hash)
        
        return params
    
    @property
    def _properties(self) -> set[str]:
        """Properties of numpy backend."""
        props = super()._properties
        props.update({"numpy", "homogeneous", "numeric_optimized"})
        return props


__all__ = ["DataBackendNumPy"]