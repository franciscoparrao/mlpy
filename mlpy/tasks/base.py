"""
Base Task class for MLPY.

Tasks encapsulate data with metadata and provide a consistent interface
for machine learning problems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import warnings
import numpy as np
import pandas as pd

from mlpy.core.base import MLPYObject
from mlpy.backends.base import DataBackend
from mlpy.backends.pandas_backend import DataBackendPandas
from mlpy.utils.registry import mlpy_tasks


class Task(MLPYObject, ABC):
    """
    Abstract base class for ML tasks.
    
    A Task encapsulates:
    - A DataBackend containing the data
    - Column roles (features, target, etc.)
    - Task metadata (type, properties)
    - Methods for data manipulation
    
    Parameters
    ----------
    id : str, optional
        Unique identifier for the task
    label : str, optional
        Human-readable label
    backend : DataBackend
        The data backend
    """
    
    def __init__(
        self,
        backend: DataBackend,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        super().__init__(id=id, label=label, **kwargs)
        self._backend = backend
        self._col_roles: Dict[str, Set[str]] = {
            "feature": set(),
            "target": set(),
            "name": set(),  # ID/name columns
            "order": set(),  # Ordering columns
            "stratum": set(),  # Stratification columns
            "group": set(),  # Grouping columns
            "weight": set(),  # Instance weights
        }
        self._row_roles: Dict[str, Set[int]] = {
            "use": set(range(backend.nrow)),  # Rows to use
            "test": set(),  # Hold-out test rows
        }
        
    @property
    @abstractmethod
    def task_type(self) -> str:
        """Type of task (e.g., 'classif', 'regr')."""
        pass
    
    @property
    def backend(self) -> DataBackend:
        """The data backend."""
        return self._backend
    
    @property
    def nrow(self) -> int:
        """Number of rows in use."""
        return len(self._row_roles["use"])
    
    @property
    def ncol(self) -> int:
        """Number of active columns (features + target)."""
        return len(self.feature_names) + len(self.target_names)
    
    @property
    def feature_names(self) -> List[str]:
        """Names of feature columns."""
        return sorted(self._col_roles["feature"])
    
    @property
    def target_names(self) -> List[str]:
        """Names of target columns."""
        return sorted(self._col_roles["target"])
    
    @property
    def col_roles(self) -> Dict[str, Set[str]]:
        """Get column roles (read-only)."""
        return self._col_roles.copy()
    
    @property
    def row_roles(self) -> Dict[str, Set[int]]:
        """Get row roles (read-only)."""
        return self._row_roles.copy()
    
    @property
    def feature_types(self) -> Dict[str, str]:
        """Types of feature columns."""
        col_info = self._backend.col_info()
        return {
            col: col_info[col]["type"]
            for col in self.feature_names
        }
    
    def data(
        self,
        rows: Optional[Sequence[int]] = None,
        cols: Optional[Sequence[str]] = None,
        data_format: str = "dataframe"
    ) -> Any:
        """
        Get data from the task.
        
        Parameters
        ----------
        rows : Sequence[int], optional
            Row indices to retrieve. If None, all rows in use are returned.
        cols : Sequence[str], optional
            Column names to retrieve. If None, all features and target are returned.
        data_format : str
            Format: "dataframe", "array", or "dict"
            
        Returns
        -------
        Any
            Data in requested format
        """
        # Default to rows in use
        if rows is None:
            rows = sorted(self._row_roles["use"])
        else:
            # Validate rows are in use
            invalid_rows = set(rows) - self._row_roles["use"]
            if invalid_rows:
                raise ValueError(f"Rows not in use: {invalid_rows}")
        
        # Default to features + target
        if cols is None:
            cols = self.feature_names + self.target_names
        
        return self._backend.data(rows=rows, cols=cols, data_format=data_format)
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows of data in use."""
        rows_in_use = sorted(self._row_roles["use"])[:n]
        cols = self.feature_names + self.target_names
        return self._backend.data(rows=rows_in_use, cols=cols, data_format="dataframe")
    
    def set_col_roles(self, col_roles: Dict[str, Union[str, List[str]]]) -> None:
        """
        Set column roles.
        
        Parameters
        ----------
        col_roles : Dict[str, Union[str, List[str]]]
            Mapping of role names to column names
            
        Examples
        --------
        >>> task.set_col_roles({
        ...     "target": "species",
        ...     "feature": ["sepal_length", "sepal_width"],
        ...     "name": "id"
        ... })
        """
        # Clear existing roles for columns being set
        for role, cols in col_roles.items():
            if role not in self._col_roles:
                raise ValueError(f"Unknown column role: {role}")
            
            if isinstance(cols, str):
                cols = [cols]
            
            # Validate columns exist
            all_cols = set(self._backend.colnames)
            invalid_cols = set(cols) - all_cols
            if invalid_cols:
                raise ValueError(f"Columns not found: {invalid_cols}")
            
            # Remove these columns from all roles
            for col in cols:
                for r in self._col_roles:
                    self._col_roles[r].discard(col)
            
            # Set new role
            self._col_roles[role].update(cols)
        
        # Validate task after role changes
        self._validate_task()
        self._mark_dirty()
    
    def set_row_roles(self, row_roles: Dict[str, Union[int, List[int]]]) -> None:
        """
        Set row roles.
        
        Parameters
        ----------
        row_roles : Dict[str, Union[int, List[int]]]
            Mapping of role names to row indices
        """
        for role, rows in row_roles.items():
            if role not in self._row_roles:
                raise ValueError(f"Unknown row role: {role}")
            
            if isinstance(rows, int):
                rows = [rows]
            
            # Validate row indices
            max_row = self._backend.nrow
            invalid_rows = [r for r in rows if r < 0 or r >= max_row]
            if invalid_rows:
                raise ValueError(f"Invalid row indices: {invalid_rows}")
            
            self._row_roles[role] = set(rows)
        
        self._mark_dirty()
    
    def filter(self, rows: Sequence[int]) -> "Task":
        """
        Create a new task with filtered rows.
        
        Parameters
        ----------
        rows : Sequence[int]
            Row indices to keep
            
        Returns
        -------
        Task
            New task with filtered data
        """
        new_task = self.clone()
        new_task.set_row_roles({"use": rows})
        return new_task
    
    def select(self, cols: Sequence[str]) -> "Task":
        """
        Create a new task with selected columns.
        
        Parameters
        ----------
        cols : Sequence[str]
            Column names to keep as features
            
        Returns
        -------
        Task
            New task with selected features
        """
        new_task = self.clone()
        
        # Keep only selected features
        current_features = set(self.feature_names)
        new_features = set(cols) & current_features
        
        if not new_features:
            warnings.warn("No valid features selected", UserWarning)
        
        # Remove all features first, then add back only selected ones
        all_features = list(current_features)
        features_to_remove = current_features - new_features
        
        # Clear all features from feature role
        for feat in features_to_remove:
            new_task._col_roles["feature"].discard(feat)
        
        # Ensure only selected features remain
        new_task._col_roles["feature"] = new_features
        
        return new_task
    
    def cbind(self, data: Union[pd.DataFrame, DataBackend]) -> "Task":
        """
        Add columns to the task.
        
        Parameters
        ----------
        data : pd.DataFrame or DataBackend
            Data to add (must have same number of rows)
            
        Returns
        -------
        Task
            New task with additional columns
        """
        from mlpy.backends.base import DataBackendCbind
        
        # Convert to backend if needed
        if isinstance(data, pd.DataFrame):
            data = DataBackendPandas(data)
        
        # Create combined backend
        new_backend = DataBackendCbind([self._backend, data])
        
        # Create new task with cloned settings
        new_task = self.clone()
        new_task._backend = new_backend
        
        # Add new columns as features by default
        existing_cols = set(self._backend.colnames)
        new_cols = set(data.colnames) - existing_cols
        
        # Add new columns to feature role
        if new_cols:
            new_task._col_roles["feature"].update(new_cols)
        
        return new_task
    
    def rbind(self, data: Union[pd.DataFrame, DataBackend, "Task"]) -> "Task":
        """
        Add rows to the task.
        
        Parameters
        ----------
        data : pd.DataFrame, DataBackend, or Task
            Data to add (must have same columns)
            
        Returns
        -------
        Task
            New task with additional rows
        """
        from mlpy.backends.base import DataBackendRbind
        
        # Handle different input types
        if isinstance(data, Task):
            if not set(data._backend.colnames) == set(self._backend.colnames):
                raise ValueError("Tasks must have the same columns")
            other_backend = data._backend
        elif isinstance(data, pd.DataFrame):
            other_backend = DataBackendPandas(data)
        elif isinstance(data, DataBackend):
            other_backend = data
        else:
            raise TypeError(f"Cannot rbind object of type {type(data)}")
        
        # Create combined backend
        new_backend = DataBackendRbind([self._backend, other_backend])
        
        # Create new task with cloned settings
        new_task = self.clone()
        new_task._backend = new_backend
        
        # Update row roles (offset indices for second backend)
        offset = self._backend.nrow
        new_task._row_roles["use"] = self._row_roles["use"].copy()
        if isinstance(data, Task):
            # Add rows from other task with offset
            new_task._row_roles["use"].update(
                r + offset for r in data._row_roles["use"]
            )
        else:
            # Add all new rows
            new_task._row_roles["use"].update(
                range(offset, new_backend.nrow)
            )
        
        return new_task
    
    @abstractmethod
    def _validate_task(self) -> None:
        """
        Validate task-specific requirements.
        
        Should be implemented by subclasses to check:
        - Target column requirements
        - Feature requirements
        - Task-specific constraints
        """
        pass
    
    def _get_params_for_hash(self) -> Dict[str, Any]:
        """Include task-specific parameters in hash."""
        params = super()._get_params_for_hash()
        params.update({
            "backend_hash": self._backend.hash,
            "col_roles": {k: sorted(v) for k, v in self._col_roles.items()},
            "row_roles": {k: sorted(v) for k, v in self._row_roles.items()},
            "task_type": self.task_type,
        })
        return params
    
    @property
    def _properties(self) -> Set[str]:
        """Task properties."""
        props = {self.task_type}
        
        # Add data-related properties
        if self._backend.missings() > 0:
            props.add("missings")
        
        if "weight" in self._col_roles and self._col_roles["weight"]:
            props.add("weights")
            
        if "group" in self._col_roles and self._col_roles["group"]:
            props.add("groups")
            
        if "stratum" in self._col_roles and self._col_roles["stratum"]:
            props.add("strata")
        
        return props
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<{self.__class__.__name__}:{self.id} ({self.nrow} x {self.ncol})>"
        )
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"{self.__class__.__name__} '{self.label}' with {self.nrow} rows "
            f"and {self.ncol} columns ({len(self.feature_names)} features)"
        )


__all__ = ["Task"]