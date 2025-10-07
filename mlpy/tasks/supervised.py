"""
Supervised learning tasks for MLPY.
"""

from abc import ABC
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .base import Task
from mlpy.backends.base import DataBackend
from mlpy.backends.pandas_backend import DataBackendPandas
from mlpy.utils.registry import mlpy_tasks


class TaskSupervised(Task, ABC):
    """
    Abstract base class for supervised learning tasks.
    
    Supervised tasks have a target variable to predict.
    
    Parameters
    ----------
    data : pd.DataFrame, DataBackend, or dict
        The data for the task
    target : str or List[str]
        Name(s) of the target column(s)
    id : str, optional
        Task identifier
    label : str, optional
        Task label
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, DataBackend, Dict[str, Any]],
        target: Union[str, List[str]],
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        # Convert data to backend if needed
        if isinstance(data, pd.DataFrame):
            backend = DataBackendPandas(data)
        elif isinstance(data, dict):
            # Assume dict of arrays
            df = pd.DataFrame(data)
            backend = DataBackendPandas(df)
        elif isinstance(data, DataBackend):
            backend = data
        else:
            raise TypeError(
                f"data must be DataFrame, DataBackend, or dict, got {type(data)}"
            )
        
        super().__init__(backend=backend, id=id, label=label, **kwargs)
        
        # Set target
        if isinstance(target, str):
            target = [target]
        
        # All other columns become features by default
        all_cols = set(backend.colnames)
        target_cols = set(target)
        feature_cols = all_cols - target_cols
        
        self.set_col_roles({
            "target": list(target_cols),
            "feature": list(feature_cols),
        })
    
    @property
    def target_names(self) -> List[str]:
        """Names of target columns."""
        return sorted(self._col_roles["target"])
    
    @property
    def formula(self) -> str:
        """String representation of the task formula."""
        targets = " + ".join(self.target_names)
        features = " + ".join(self.feature_names[:3])
        if len(self.feature_names) > 3:
            features += " + ..."
        return f"{targets} ~ {features}"
    
    def truth(self, rows: Optional[List[int]] = None) -> np.ndarray:
        """
        Get true target values.
        
        Parameters
        ----------
        rows : List[int], optional
            Row indices. If None, all rows in use.
            
        Returns
        -------
        np.ndarray
            True target values
        """
        data = self.data(rows=rows, cols=self.target_names, data_format="array")
        if data.shape[1] == 1:
            return data.ravel()
        return data


class TaskClassif(TaskSupervised):
    """
    Classification task.
    
    Parameters
    ----------
    data : pd.DataFrame, DataBackend, or dict
        The data for the task
    target : str
        Name of the target column (must be categorical)
    positive : str, optional
        Name of the positive class for binary classification
    id : str, optional
        Task identifier
    label : str, optional  
        Task label
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, DataBackend, Dict[str, Any]],
        target: str,
        positive: Optional[str] = None,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        # Initialize positive before super().__init__ since validation may be called
        self._positive = positive
        
        super().__init__(
            data=data,
            target=target,
            id=id,
            label=label,
            **kwargs
        )
        
        self._validate_task()
    
    @property
    def task_type(self) -> str:
        """Task type identifier."""
        return "classif"
    
    @property
    def class_names(self) -> List[str]:
        """Unique class labels in the target, sorted."""
        target_col = self.target_names[0]
        rows_in_use = sorted(self._row_roles["use"])
        
        distinct = self._backend.distinct([target_col], rows=rows_in_use)
        classes = distinct[target_col]
        
        # Convert to strings and sort
        return sorted(str(c) for c in classes)
    
    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(self.class_names)
    
    @property
    def positive(self) -> Optional[str]:
        """Positive class for binary classification."""
        if self.n_classes == 2 and self._positive is None:
            # Default to second class (like "1" vs "0")
            return self.class_names[1]
        return self._positive
    
    @property
    def negative(self) -> Optional[str]:
        """Negative class for binary classification."""
        if self.n_classes == 2:
            pos = self.positive
            return [c for c in self.class_names if c != pos][0]
        return None
    
    def _validate_task(self) -> None:
        """Validate classification task requirements."""
        # Must have exactly one target
        if len(self.target_names) != 1:
            raise ValueError(
                f"Classification requires exactly one target column, "
                f"got {len(self.target_names)}"
            )
        
        # Must have at least one feature
        if not self.feature_names:
            raise ValueError("Classification requires at least one feature")
        
        # Check target has at least 2 classes
        if self.n_classes < 2:
            raise ValueError(
                f"Target must have at least 2 classes, got {self.n_classes}"
            )
        
        # Validate positive class if specified
        if self._positive is not None:
            if str(self._positive) not in self.class_names:
                raise ValueError(
                    f"Positive class '{self._positive}' not found in target. "
                    f"Available classes: {self.class_names}"
                )
    
    @property
    def _properties(self) -> set[str]:
        """Task properties."""
        props = super()._properties
        
        if self.n_classes == 2:
            props.add("binary")
        else:
            props.add("multiclass")
        
        return props


class TaskRegr(TaskSupervised):
    """
    Regression task.
    
    Parameters
    ----------
    data : pd.DataFrame, DataBackend, or dict
        The data for the task
    target : str
        Name of the target column (must be numeric)
    id : str, optional
        Task identifier
    label : str, optional
        Task label
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, DataBackend, Dict[str, Any]],
        target: str,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            data=data,
            target=target,
            id=id,
            label=label,
            **kwargs
        )
        
        self._validate_task()
    
    @property
    def task_type(self) -> str:
        """Task type identifier."""
        return "regr"
    
    def _validate_task(self) -> None:
        """Validate regression task requirements."""
        # Must have exactly one target
        if len(self.target_names) != 1:
            raise ValueError(
                f"Regression requires exactly one target column, "
                f"got {len(self.target_names)}"
            )
        
        # Must have at least one feature
        if not self.feature_names:
            raise ValueError("Regression requires at least one feature")
        
        # Check target is numeric
        target_col = self.target_names[0]
        col_info = self._backend.col_info()
        target_type = col_info[target_col]["type"]
        
        if target_type not in ("numeric", "integer"):
            raise ValueError(
                f"Regression target must be numeric, got type '{target_type}'"
            )


# Register task types
mlpy_tasks.register("classif", TaskClassif, aliases=["classification"])
mlpy_tasks.register("regr", TaskRegr, aliases=["regression"])


__all__ = ["TaskSupervised", "TaskClassif", "TaskRegr"]