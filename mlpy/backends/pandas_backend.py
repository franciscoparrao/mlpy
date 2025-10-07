"""
Pandas DataFrame backend for MLPY.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from .base import DataBackend


class DataBackendPandas(DataBackend):
    """
    DataBackend using pandas DataFrame for storage.
    
    This is the primary backend for most use cases, providing
    efficient storage and operations on tabular data.
    
    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame to wrap
    primary_key : str, optional
        Name of the column to use as primary key
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        primary_key: Optional[str] = None,
        **kwargs
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data)}")
        
        super().__init__(data=data, primary_key=primary_key, **kwargs)
    
    def _initialize_backend(self, data: pd.DataFrame) -> None:
        """Store the DataFrame."""
        self._data = data.copy()
        
        # Validate primary key if provided
        if self._primary_key is not None:
            if self._primary_key not in self._data.columns:
                raise ValueError(f"Primary key column '{self._primary_key}' not found in data")
            
            # Check if primary key values are unique
            if self._data[self._primary_key].duplicated().any():
                raise ValueError(f"Primary key column '{self._primary_key}' contains duplicates")
    
    @property
    def nrow(self) -> int:
        """Number of rows."""
        return len(self._data)
    
    @property
    def ncol(self) -> int:
        """Number of columns."""
        return len(self._data.columns)
    
    @property
    def colnames(self) -> List[str]:
        """Column names."""
        return self._data.columns.tolist()
    
    @property
    def rownames(self) -> List[Union[str, int]]:
        """Row names/indices."""
        if self._primary_key:
            return self._data[self._primary_key].tolist()
        else:
            return self._data.index.tolist()
    
    def data(
        self, 
        rows: Optional[Sequence[Union[int, str]]] = None,
        cols: Optional[Sequence[str]] = None,
        data_format: str = "dataframe"
    ) -> Any:
        """
        Retrieve data in specified format.
        
        Parameters
        ----------
        rows : Sequence[int or str], optional
            Row indices/names to retrieve
        cols : Sequence[str], optional
            Column names to retrieve
        data_format : str
            Format: "dataframe", "array", or "dict"
            
        Returns
        -------
        Any
            Data in requested format
        """
        # Select columns
        if cols is None:
            df_subset = self._data
        else:
            # Validate column names
            missing_cols = set(cols) - set(self._data.columns)
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            df_subset = self._data[cols]
        
        # Select rows
        if rows is not None:
            if self._primary_key:
                # Use primary key for row selection
                df_subset = df_subset[df_subset.index.isin(self._data[self._data[self._primary_key].isin(rows)].index)]
            else:
                # Use index
                if all(isinstance(r, int) for r in rows):
                    df_subset = df_subset.iloc[list(rows)]
                else:
                    df_subset = df_subset.loc[list(rows)]
        
        # Convert to requested format
        if data_format == "dataframe":
            return df_subset
        elif data_format == "array":
            return df_subset.to_numpy()
        elif data_format == "dict":
            return {col: df_subset[col].to_numpy() for col in df_subset.columns}
        else:
            raise ValueError(f"Unknown data format: {data_format}")
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows."""
        return self._data.head(n)
    
    def distinct(
        self, 
        cols: Sequence[str],
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """Get distinct values for columns."""
        # Get subset of data
        if rows is None:
            df_subset = self._data
        else:
            if self._primary_key:
                df_subset = self._data[self._data[self._primary_key].isin(rows)]
            else:
                if all(isinstance(r, int) for r in rows):
                    df_subset = self._data.iloc[list(rows)]
                else:
                    df_subset = self._data.loc[list(rows)]
        
        # Get unique values for each column
        result = {}
        for col in cols:
            if col not in self._data.columns:
                raise ValueError(f"Column '{col}' not found")
            
            unique_vals = df_subset[col].dropna().unique()
            # Sort if possible
            try:
                unique_vals = np.sort(unique_vals)
            except TypeError:
                # Can't sort mixed types
                pass
            result[col] = unique_vals
        
        return result
    
    def missings(
        self, 
        cols: Optional[Sequence[str]] = None,
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> int:
        """Count missing values."""
        # Get subset of data
        if cols is None:
            df_subset = self._data
        else:
            df_subset = self._data[cols]
        
        if rows is not None:
            if self._primary_key:
                df_subset = df_subset[self._data[self._primary_key].isin(rows)]
            else:
                if all(isinstance(r, int) for r in rows):
                    df_subset = df_subset.iloc[list(rows)]
                else:
                    df_subset = df_subset.loc[list(rows)]
        
        return int(df_subset.isna().sum().sum())
    
    def _infer_col_type(self, col: str) -> str:
        """Infer column type."""
        dtype = self._data[col].dtype
        
        if pd.api.types.is_bool_dtype(dtype):
            return "logical"
        elif pd.api.types.is_integer_dtype(dtype):
            return "integer"
        elif pd.api.types.is_float_dtype(dtype):
            return "numeric"
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            # Check if it's actually a string column
            sample = self._data[col].dropna().head(10)
            if all(isinstance(x, str) for x in sample):
                return "character"
            else:
                return "mixed"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        elif pd.api.types.is_categorical_dtype(dtype):
            return "factor"
        else:
            return "unknown"
    
    def _get_params_for_hash(self) -> Dict[str, Any]:
        """Include data content in hash calculation."""
        params = super()._get_params_for_hash()
        
        # Add a hash of the actual data content
        # Use pandas util for efficient hashing
        data_hash = pd.util.hash_pandas_object(self._data).sum()
        params["data_hash"] = str(data_hash)
        
        return params
    
    @property
    def _properties(self) -> set[str]:
        """Properties of pandas backend."""
        props = super()._properties
        props.update({"pandas", "efficient_slicing", "mixed_types"})
        return props


__all__ = ["DataBackendPandas"]