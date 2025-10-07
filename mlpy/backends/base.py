"""
Base class for data backends.

DataBackends provide an abstraction layer for storing and accessing data
in different formats (pandas DataFrame, numpy array, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from mlpy.core.base import MLPYObject


class DataBackend(MLPYObject, ABC):
    """
    Abstract base class for data storage backends.
    
    A DataBackend provides a unified interface for accessing tabular data
    regardless of the underlying storage format.
    
    Parameters
    ----------
    data : Any
        The data to store in the backend
    primary_key : str, optional
        Name of the column to use as primary key. If None, row indices are used.
    """
    
    def __init__(
        self, 
        data: Any,
        primary_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._primary_key = primary_key
        self._initialize_backend(data)
        
    @property
    def _properties(self) -> set[str]:
        """Properties of this backend."""
        return {"tabular", "row_access", "col_access"}
    
    @abstractmethod
    def _initialize_backend(self, data: Any) -> None:
        """
        Initialize the backend with data.
        
        Parameters
        ----------
        data : Any
            The data to store
        """
        pass
    
    @property
    @abstractmethod
    def nrow(self) -> int:
        """Number of rows in the data."""
        pass
    
    @property
    @abstractmethod
    def ncol(self) -> int:
        """Number of columns in the data."""
        pass
    
    @property
    @abstractmethod
    def colnames(self) -> List[str]:
        """List of column names."""
        pass
    
    @property
    @abstractmethod
    def rownames(self) -> List[Union[str, int]]:
        """List of row names/indices."""
        pass
    
    @abstractmethod
    def data(
        self, 
        rows: Optional[Sequence[Union[int, str]]] = None,
        cols: Optional[Sequence[str]] = None,
        data_format: str = "dataframe"
    ) -> Any:
        """
        Retrieve data from the backend.
        
        Parameters
        ----------
        rows : Sequence[int or str], optional
            Row indices/names to retrieve. If None, all rows are returned.
        cols : Sequence[str], optional
            Column names to retrieve. If None, all columns are returned.
        data_format : str, default="dataframe"
            Format for the returned data. Options: "dataframe", "array", "dict"
            
        Returns
        -------
        Any
            The requested data in the specified format
        """
        pass
    
    @abstractmethod
    def head(self, n: int = 5) -> Any:
        """
        Get the first n rows of data.
        
        Parameters
        ----------
        n : int, default=5
            Number of rows to return
            
        Returns
        -------
        Any
            First n rows of data
        """
        pass
    
    @abstractmethod
    def distinct(
        self, 
        cols: Sequence[str],
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get distinct values for specified columns.
        
        Parameters
        ----------
        cols : Sequence[str]
            Column names to get distinct values for
        rows : Sequence[int or str], optional
            Row indices/names to consider. If None, all rows are used.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping column names to arrays of distinct values
        """
        pass
    
    @abstractmethod
    def missings(
        self, 
        cols: Optional[Sequence[str]] = None,
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> int:
        """
        Count missing values.
        
        Parameters
        ----------
        cols : Sequence[str], optional
            Column names to check. If None, all columns are checked.
        rows : Sequence[int or str], optional
            Row indices/names to check. If None, all rows are checked.
            
        Returns
        -------
        int
            Number of missing values
        """
        pass
    
    @property
    def hash(self) -> str:
        """
        Hash of the backend.
        
        The hash should change when the data changes.
        """
        # Include data dimensions in hash calculation
        return super().hash
    
    def _get_params_for_hash(self) -> Dict[str, Any]:
        """Include backend-specific parameters in hash."""
        params = super()._get_params_for_hash()
        params.update({
            "nrow": self.nrow,
            "ncol": self.ncol,
            "colnames": sorted(self.colnames),
            "primary_key": self._primary_key,
        })
        return params
    
    @property
    def primary_key(self) -> Optional[str]:
        """Name of the primary key column."""
        return self._primary_key
    
    def col_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about columns.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping column names to their properties
            (type, missing count, etc.)
        """
        info = {}
        for col in self.colnames:
            col_data = self.data(cols=[col])
            info[col] = {
                "type": self._infer_col_type(col),
                "missing": self.missings(cols=[col]),
                "distinct": len(self.distinct([col])[col]),
            }
        return info
    
    @abstractmethod
    def _infer_col_type(self, col: str) -> str:
        """
        Infer the type of a column.
        
        Parameters
        ----------
        col : str
            Column name
            
        Returns
        -------
        str
            Type name (e.g., "numeric", "integer", "character", "logical")
        """
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__}({self.nrow} x {self.ncol})>"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.__class__.__name__} with {self.nrow} rows and {self.ncol} columns"


class DataBackendCbind(DataBackend):
    """
    Virtual backend that column-binds multiple backends.
    
    Parameters
    ----------
    backends : List[DataBackend]
        Backends to combine column-wise
    """
    
    def __init__(self, backends: List[DataBackend], **kwargs):
        self.backends = backends
        super().__init__(data=None, **kwargs)
        
    def _initialize_backend(self, data: Any) -> None:
        """Initialize by validating backends have same number of rows."""
        if not self.backends:
            raise ValueError("At least one backend is required")
            
        nrows = [b.nrow for b in self.backends]
        if len(set(nrows)) > 1:
            raise ValueError(f"All backends must have the same number of rows, got {nrows}")
            
        # Check for duplicate column names
        all_cols = []
        for b in self.backends:
            all_cols.extend(b.colnames)
        if len(all_cols) != len(set(all_cols)):
            raise ValueError("Column names must be unique across all backends")
    
    @property
    def nrow(self) -> int:
        """Number of rows."""
        return self.backends[0].nrow if self.backends else 0
    
    @property
    def ncol(self) -> int:
        """Total number of columns across all backends."""
        return sum(b.ncol for b in self.backends)
    
    @property
    def colnames(self) -> List[str]:
        """Combined column names from all backends."""
        cols = []
        for b in self.backends:
            cols.extend(b.colnames)
        return cols
    
    @property
    def rownames(self) -> List[Union[str, int]]:
        """Row names from the first backend."""
        return self.backends[0].rownames if self.backends else []
    
    def data(
        self, 
        rows: Optional[Sequence[Union[int, str]]] = None,
        cols: Optional[Sequence[str]] = None,
        data_format: str = "dataframe"
    ) -> Any:
        """Retrieve data by delegating to appropriate backends."""
        if cols is None:
            cols = self.colnames
            
        # Group columns by backend
        backend_cols = {}
        for col in cols:
            for i, b in enumerate(self.backends):
                if col in b.colnames:
                    if i not in backend_cols:
                        backend_cols[i] = []
                    backend_cols[i].append(col)
                    break
        
        # Get data from each backend
        if data_format == "dataframe":
            import pandas as pd
            dfs = []
            for i, col_list in backend_cols.items():
                df = self.backends[i].data(rows=rows, cols=col_list, data_format="dataframe")
                dfs.append(df)
            return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
            
        elif data_format == "array":
            arrays = []
            for i, col_list in backend_cols.items():
                arr = self.backends[i].data(rows=rows, cols=col_list, data_format="array")
                arrays.append(arr)
            return np.hstack(arrays) if arrays else np.array([])
            
        elif data_format == "dict":
            result = {}
            for i, col_list in backend_cols.items():
                d = self.backends[i].data(rows=rows, cols=col_list, data_format="dict")
                result.update(d)
            return result
            
        else:
            raise ValueError(f"Unknown data format: {data_format}")
    
    def head(self, n: int = 5) -> Any:
        """Get first n rows."""
        return self.data(rows=list(range(min(n, self.nrow))))
    
    def distinct(
        self, 
        cols: Sequence[str],
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """Get distinct values by delegating to appropriate backends."""
        result = {}
        for col in cols:
            for b in self.backends:
                if col in b.colnames:
                    result.update(b.distinct([col], rows))
                    break
        return result
    
    def missings(
        self, 
        cols: Optional[Sequence[str]] = None,
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> int:
        """Count missings across all backends."""
        if cols is None:
            return sum(b.missings(rows=rows) for b in self.backends)
        
        total = 0
        for col in cols:
            for b in self.backends:
                if col in b.colnames:
                    total += b.missings(cols=[col], rows=rows)
                    break
        return total
    
    def _infer_col_type(self, col: str) -> str:
        """Infer type by delegating to appropriate backend."""
        for b in self.backends:
            if col in b.colnames:
                return b._infer_col_type(col)
        raise ValueError(f"Column {col} not found")


class DataBackendRbind(DataBackend):
    """
    Virtual backend that row-binds multiple backends.
    
    Parameters
    ----------
    backends : List[DataBackend]
        Backends to combine row-wise
    """
    
    def __init__(self, backends: List[DataBackend], **kwargs):
        self.backends = backends
        super().__init__(data=None, **kwargs)
        
    def _initialize_backend(self, data: Any) -> None:
        """Initialize by validating backends have same columns."""
        if not self.backends:
            raise ValueError("At least one backend is required")
            
        # Check all backends have the same columns
        col_sets = [set(b.colnames) for b in self.backends]
        if len(set(frozenset(s) for s in col_sets)) > 1:
            raise ValueError("All backends must have the same columns")
    
    @property
    def nrow(self) -> int:
        """Total number of rows across all backends."""
        return sum(b.nrow for b in self.backends)
    
    @property
    def ncol(self) -> int:
        """Number of columns."""
        return self.backends[0].ncol if self.backends else 0
    
    @property
    def colnames(self) -> List[str]:
        """Column names from the first backend."""
        return self.backends[0].colnames if self.backends else []
    
    @property
    def rownames(self) -> List[Union[str, int]]:
        """Combined row names from all backends."""
        names = []
        offset = 0
        for b in self.backends:
            # If using integer indices, offset them
            if all(isinstance(n, int) for n in b.rownames):
                names.extend([n + offset for n in b.rownames])
                offset += b.nrow
            else:
                names.extend(b.rownames)
        return names
    
    def _map_rows(self, rows: Sequence[Union[int, str]]) -> Dict[int, List[Union[int, str]]]:
        """Map global row indices to backend-specific indices."""
        backend_rows = {}
        
        # Build cumulative row counts
        cum_rows = [0]
        for b in self.backends:
            cum_rows.append(cum_rows[-1] + b.nrow)
        
        for row in rows:
            if isinstance(row, int):
                # Find which backend this row belongs to
                for i, (start, end) in enumerate(zip(cum_rows[:-1], cum_rows[1:])):
                    if start <= row < end:
                        if i not in backend_rows:
                            backend_rows[i] = []
                        backend_rows[i].append(row - start)
                        break
            else:
                # String row names - search in each backend
                for i, b in enumerate(self.backends):
                    if row in b.rownames:
                        if i not in backend_rows:
                            backend_rows[i] = []
                        backend_rows[i].append(row)
                        break
                        
        return backend_rows
    
    def data(
        self, 
        rows: Optional[Sequence[Union[int, str]]] = None,
        cols: Optional[Sequence[str]] = None,
        data_format: str = "dataframe"
    ) -> Any:
        """Retrieve data by delegating to appropriate backends."""
        if rows is None:
            # Get all data
            if data_format == "dataframe":
                import pandas as pd
                dfs = [b.data(cols=cols, data_format="dataframe") for b in self.backends]
                return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()
                
            elif data_format == "array":
                arrays = [b.data(cols=cols, data_format="array") for b in self.backends]
                return np.vstack(arrays) if arrays else np.array([])
                
            elif data_format == "dict":
                # Combine dictionaries, concatenating arrays
                result = {}
                for b in self.backends:
                    d = b.data(cols=cols, data_format="dict")
                    for k, v in d.items():
                        if k not in result:
                            result[k] = []
                        result[k].extend(v if isinstance(v, list) else v.tolist())
                return {k: np.array(v) for k, v in result.items()}
                
        else:
            # Get specific rows
            backend_rows = self._map_rows(rows)
            
            if data_format == "dataframe":
                import pandas as pd
                dfs = []
                for i, row_list in backend_rows.items():
                    df = self.backends[i].data(rows=row_list, cols=cols, data_format="dataframe")
                    dfs.append(df)
                return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()
                
            elif data_format == "array":
                arrays = []
                for i, row_list in backend_rows.items():
                    arr = self.backends[i].data(rows=row_list, cols=cols, data_format="array")
                    arrays.append(arr)
                return np.vstack(arrays) if arrays else np.array([])
                
            elif data_format == "dict":
                result = {}
                for i, row_list in backend_rows.items():
                    d = self.backends[i].data(rows=row_list, cols=cols, data_format="dict")
                    for k, v in d.items():
                        if k not in result:
                            result[k] = []
                        result[k].extend(v if isinstance(v, list) else v.tolist())
                return {k: np.array(v) for k, v in result.items()}
        
        raise ValueError(f"Unknown data format: {data_format}")
    
    def head(self, n: int = 5) -> Any:
        """Get first n rows."""
        return self.data(rows=list(range(min(n, self.nrow))))
    
    def distinct(
        self, 
        cols: Sequence[str],
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """Get distinct values across all backends."""
        if rows is None:
            # Get distinct from all backends and combine
            all_distincts = {}
            for col in cols:
                values = []
                for b in self.backends:
                    d = b.distinct([col])
                    values.extend(d[col].tolist())
                all_distincts[col] = np.unique(values)
            return all_distincts
        else:
            # Get distinct from specific rows
            backend_rows = self._map_rows(rows)
            all_distincts = {}
            for col in cols:
                values = []
                for i, row_list in backend_rows.items():
                    d = self.backends[i].distinct([col], rows=row_list)
                    values.extend(d[col].tolist())
                all_distincts[col] = np.unique(values)
            return all_distincts
    
    def missings(
        self, 
        cols: Optional[Sequence[str]] = None,
        rows: Optional[Sequence[Union[int, str]]] = None
    ) -> int:
        """Count missings across all backends."""
        if rows is None:
            return sum(b.missings(cols=cols) for b in self.backends)
        else:
            backend_rows = self._map_rows(rows)
            total = 0
            for i, row_list in backend_rows.items():
                total += self.backends[i].missings(cols=cols, rows=row_list)
            return total
    
    def _infer_col_type(self, col: str) -> str:
        """Infer type from first backend."""
        return self.backends[0]._infer_col_type(col) if self.backends else "unknown"


__all__ = ["DataBackend", "DataBackendCbind", "DataBackendRbind"]