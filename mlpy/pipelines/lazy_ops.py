"""Lazy evaluation operators for MLPY pipelines.

This module provides pipeline operators that support lazy evaluation
for working with big data backends (Dask, Vaex) without materializing
intermediate results.
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import numpy as np
import pandas as pd
from functools import wraps

from .base import PipeOp, PipeOpInput, PipeOpOutput, mlpy_pipeops
from ..tasks import Task, TaskClassif, TaskRegr
from ..backends import DataBackend

# Check for Dask/Vaex availability
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False


class LazyPipeOp(PipeOp):
    """Base class for pipeline operations with lazy evaluation support.
    
    This class provides infrastructure for operations that can work
    with both in-memory and out-of-core data backends.
    """
    
    @property
    def supports_lazy(self) -> bool:
        """Whether this operation supports lazy evaluation."""
        return True
        
    def _get_backend_type(self, task: Task) -> str:
        """Get the backend type of a task.
        
        Parameters
        ----------
        task : Task
            The task to check.
            
        Returns
        -------
        str
            Backend type: 'pandas', 'dask', 'vaex', or 'unknown'.
        """
        backend_name = type(task.backend).__name__
        
        if 'Pandas' in backend_name:
            return 'pandas'
        elif 'Dask' in backend_name:
            return 'dask'
        elif 'Vaex' in backend_name:
            return 'vaex'
        else:
            return 'unknown'
            
    def _ensure_lazy_compatible(self, task: Task) -> None:
        """Ensure the task backend supports lazy operations.
        
        Parameters
        ----------
        task : Task
            The task to check.
            
        Raises
        ------
        ValueError
            If the backend doesn't support required operations.
        """
        backend_type = self._get_backend_type(task)
        
        if backend_type == 'unknown':
            raise ValueError(
                f"Unknown backend type: {type(task.backend).__name__}. "
                "LazyPipeOp requires Pandas, Dask, or Vaex backend."
            )
            
        # Check specific backend capabilities
        if hasattr(task.backend, 'supports_lazy') and not task.backend.supports_lazy:
            raise ValueError(
                f"Backend {type(task.backend).__name__} does not support lazy evaluation"
            )


class LazyPipeOpScale(LazyPipeOp):
    """Lazy scaling operation for numeric features.
    
    This operation computes statistics lazily and applies transformations
    without materializing the full dataset.
    
    Parameters
    ----------
    id : str, default="lazy_scale"
        Unique identifier.
    method : str, default="standard"
        Scaling method: "standard", "minmax", "robust".
    columns : list, optional
        Specific columns to scale. If None, scales all numeric features.
    """
    
    def __init__(
        self,
        id: str = "lazy_scale",
        method: str = "standard",
        columns: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.method = method
        self.columns = columns
        self._stats = {}
        
    @property
    def input(self) -> Dict[str, PipeOpInput]:
        """Expects a Task."""
        return {
            "input": PipeOpInput(
                name="input",
                train=Task,
                predict=Task
            )
        }
        
    @property
    def output(self) -> Dict[str, PipeOpOutput]:
        """Returns a modified Task."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def _compute_stats_dask(self, df: "dd.DataFrame", cols: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute statistics for Dask DataFrame."""
        stats = {}
        
        for col in cols:
            if self.method == "standard":
                mean = df[col].mean()
                std = df[col].std()
                stats[col] = {
                    'mean': mean.compute(),
                    'std': std.compute()
                }
            elif self.method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                stats[col] = {
                    'min': min_val.compute(),
                    'max': max_val.compute()
                }
            elif self.method == "robust":
                # Dask doesn't have built-in quantile, so we sample
                sample = df[col].sample(frac=0.1).compute()
                q1 = np.percentile(sample, 25)
                q3 = np.percentile(sample, 75)
                median = np.median(sample)
                stats[col] = {
                    'median': median,
                    'q1': q1,
                    'q3': q3,
                    'iqr': q3 - q1
                }
                
        return stats
        
    def _compute_stats_vaex(self, df: "vaex.DataFrame", cols: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute statistics for Vaex DataFrame."""
        stats = {}
        
        for col in cols:
            if self.method == "standard":
                stats[col] = {
                    'mean': df[col].mean().item(),
                    'std': df[col].std().item()
                }
            elif self.method == "minmax":
                stats[col] = {
                    'min': df[col].min().item(),
                    'max': df[col].max().item()
                }
            elif self.method == "robust":
                # Vaex has efficient quantile computation
                quantiles = df.percentile_approx(df[col], [25, 50, 75])
                stats[col] = {
                    'median': quantiles[1],
                    'q1': quantiles[0],
                    'q3': quantiles[2],
                    'iqr': quantiles[2] - quantiles[0]
                }
                
        return stats
        
    def _apply_transform_dask(self, df: "dd.DataFrame", cols: List[str]) -> "dd.DataFrame":
        """Apply scaling transformation to Dask DataFrame."""
        df = df.copy()
        
        for col in cols:
            stats = self._stats[col]
            
            if self.method == "standard":
                df[col] = (df[col] - stats['mean']) / (stats['std'] + 1e-8)
            elif self.method == "minmax":
                range_val = stats['max'] - stats['min'] + 1e-8
                df[col] = (df[col] - stats['min']) / range_val
            elif self.method == "robust":
                df[col] = (df[col] - stats['median']) / (stats['iqr'] + 1e-8)
                
        return df
        
    def _apply_transform_vaex(self, df: "vaex.DataFrame", cols: List[str]) -> "vaex.DataFrame":
        """Apply scaling transformation to Vaex DataFrame."""
        # Vaex uses virtual columns, so transformations are lazy
        df = df.copy()
        
        for col in cols:
            stats = self._stats[col]
            
            if self.method == "standard":
                df[col] = (df[col] - stats['mean']) / (stats['std'] + 1e-8)
            elif self.method == "minmax":
                range_val = stats['max'] - stats['min'] + 1e-8
                df[col] = (df[col] - stats['min']) / range_val
            elif self.method == "robust":
                df[col] = (df[col] - stats['median']) / (stats['iqr'] + 1e-8)
                
        return df
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics and transform task lazily."""
        self.validate_inputs(inputs, "train")
        task = inputs["input"]
        
        # Ensure lazy compatibility
        self._ensure_lazy_compatible(task)
        
        # Get numeric columns
        if self.columns:
            numeric_cols = self.columns
        else:
            # Determine numeric columns based on backend type
            backend_type = self._get_backend_type(task)
            
            if backend_type == 'pandas':
                data = task.data()
                numeric_cols = [col for col in task.feature_names
                               if pd.api.types.is_numeric_dtype(data[col])]
            elif backend_type == 'dask':
                # For Dask, we need to check dtypes without computing
                df = task.backend._data
                numeric_cols = [col for col in task.feature_names
                               if df[col].dtype.kind in 'biufc']
            elif backend_type == 'vaex':
                # For Vaex, check dtypes
                df = task.backend._data
                numeric_cols = [col for col in task.feature_names
                               if df[col].dtype.kind in 'biufc']
                               
        if not numeric_cols:
            # No numeric columns to scale
            self.state.is_trained = True
            return {"output": task}
            
        # Compute statistics based on backend
        backend_type = self._get_backend_type(task)
        
        if backend_type == 'dask':
            df = task.backend._data
            self._stats = self._compute_stats_dask(df, numeric_cols)
        elif backend_type == 'vaex':
            df = task.backend._data
            self._stats = self._compute_stats_vaex(df, numeric_cols)
        else:  # pandas
            data = task.data()
            self._stats = {}
            for col in numeric_cols:
                if self.method == "standard":
                    self._stats[col] = {
                        'mean': data[col].mean(),
                        'std': data[col].std()
                    }
                elif self.method == "minmax":
                    self._stats[col] = {
                        'min': data[col].min(),
                        'max': data[col].max()
                    }
                elif self.method == "robust":
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    self._stats[col] = {
                        'median': data[col].median(),
                        'q1': q1,
                        'q3': q3,
                        'iqr': q3 - q1
                    }
                    
        # Create transformed task
        new_task = self._create_transformed_task(task, numeric_cols)
        
        self.state.is_trained = True
        self.state["stats"] = self._stats
        self.state["columns"] = numeric_cols
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply scaling transformation lazily."""
        if not self.is_trained:
            raise RuntimeError("LazyPipeOpScale must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        task = inputs["input"]
        
        # Get columns to transform
        columns = self.state["columns"]
        if not columns:
            return {"output": task}
            
        # Apply transformation
        new_task = self._create_transformed_task(task, columns)
        
        return {"output": new_task}
        
    def _create_transformed_task(self, task: Task, columns: List[str]) -> Task:
        """Create a new task with transformed data."""
        backend_type = self._get_backend_type(task)
        
        # Get transformed data based on backend
        if backend_type == 'dask':
            df = task.backend._data
            new_df = self._apply_transform_dask(df, columns)
            
            # Create new backend
            from ..backends import DataBackendDask
            new_backend = DataBackendDask(new_df)
            
        elif backend_type == 'vaex':
            df = task.backend._data
            new_df = self._apply_transform_vaex(df, columns)
            
            # Create new backend
            from ..backends import DataBackendVaex
            new_backend = DataBackendVaex(new_df)
            
        else:  # pandas
            data = task.data()
            new_data = data.copy()
            
            for col in columns:
                stats = self._stats[col]
                if self.method == "standard":
                    new_data[col] = (data[col] - stats['mean']) / (stats['std'] + 1e-8)
                elif self.method == "minmax":
                    range_val = stats['max'] - stats['min'] + 1e-8
                    new_data[col] = (data[col] - stats['min']) / range_val
                elif self.method == "robust":
                    new_data[col] = (data[col] - stats['median']) / (stats['iqr'] + 1e-8)
                    
            # Use existing backend creation
            new_backend = type(task.backend)(new_data)
            
        # Create new task with same type
        task_class = type(task)
        new_task = task_class(
            backend=new_backend,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Preserve roles
        new_task.set_col_roles({role: list(cols) for role, cols in task.col_roles.items()})
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        return new_task


class LazyPipeOpFilter(LazyPipeOp):
    """Lazy filtering operation for rows.
    
    This operation applies row filters without materializing data.
    
    Parameters
    ----------
    id : str, default="lazy_filter"
        Unique identifier.
    condition : callable or str
        Filter condition. Can be a callable that takes a DataFrame
        and returns a boolean mask, or a string query expression.
    """
    
    def __init__(
        self,
        id: str = "lazy_filter",
        condition: Union[Callable, str] = None,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.condition = condition
        
    @property
    def input(self) -> Dict[str, PipeOpInput]:
        """Expects a Task."""
        return {
            "input": PipeOpInput(
                name="input",
                train=Task,
                predict=Task
            )
        }
        
    @property
    def output(self) -> Dict[str, PipeOpOutput]:
        """Returns a filtered Task."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def _apply_filter(self, task: Task) -> Task:
        """Apply filter based on backend type."""
        backend_type = self._get_backend_type(task)
        
        if backend_type == 'dask':
            df = task.backend._data
            
            if isinstance(self.condition, str):
                # Use query for string conditions
                filtered_df = df.query(self.condition)
            else:
                # Apply callable condition
                mask = self.condition(df)
                filtered_df = df[mask]
                
            # Create new backend
            from ..backends import DataBackendDask
            new_backend = DataBackendDask(filtered_df)
            
        elif backend_type == 'vaex':
            df = task.backend._data
            
            if isinstance(self.condition, str):
                # Vaex has its own query syntax
                filtered_df = df.filter(self.condition)
            else:
                # Apply callable condition
                mask = self.condition(df)
                filtered_df = df[mask]
                
            # Create new backend
            from ..backends import DataBackendVaex
            new_backend = DataBackendVaex(filtered_df)
            
        else:  # pandas
            data = task.data()
            
            if isinstance(self.condition, str):
                filtered_data = data.query(self.condition)
            else:
                mask = self.condition(data)
                filtered_data = data[mask]
                
            new_backend = type(task.backend)(filtered_data)
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            backend=new_backend,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Update row roles to reflect filtering
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        row_roles = {}  # Reset row roles as indices have changed
        
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles(row_roles)
        
        return new_task
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filter during training."""
        self.validate_inputs(inputs, "train")
        task = inputs["input"]
        
        # Ensure lazy compatibility
        self._ensure_lazy_compatible(task)
        
        if self.condition is None:
            raise ValueError("Filter condition must be specified")
            
        # Apply filter
        new_task = self._apply_filter(task)
        
        self.state.is_trained = True
        self.state["condition"] = self.condition
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply same filter during prediction."""
        if not self.is_trained:
            raise RuntimeError("LazyPipeOpFilter must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        task = inputs["input"]
        
        # Apply filter
        new_task = self._apply_filter(task)
        
        return {"output": new_task}


class LazyPipeOpSample(LazyPipeOp):
    """Lazy sampling operation.
    
    This operation samples data without materializing the full dataset.
    Useful for creating training subsets from large datasets.
    
    Parameters
    ----------
    id : str, default="lazy_sample"
        Unique identifier.
    n : int, optional
        Number of samples to take.
    frac : float, optional
        Fraction of samples to take (0 to 1).
    replace : bool, default=False
        Whether to sample with replacement.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        id: str = "lazy_sample",
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        
        if n is None and frac is None:
            raise ValueError("Either 'n' or 'frac' must be specified")
        if n is not None and frac is not None:
            raise ValueError("Cannot specify both 'n' and 'frac'")
            
        self.n = n
        self.frac = frac
        self.replace = replace
        self.random_state = random_state
        
    @property
    def input(self) -> Dict[str, PipeOpInput]:
        """Expects a Task."""
        return {
            "input": PipeOpInput(
                name="input",
                train=Task,
                predict=Task
            )
        }
        
    @property
    def output(self) -> Dict[str, PipeOpOutput]:
        """Returns a sampled Task."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sample data during training."""
        self.validate_inputs(inputs, "train")
        task = inputs["input"]
        
        # Ensure lazy compatibility
        self._ensure_lazy_compatible(task)
        
        backend_type = self._get_backend_type(task)
        
        if backend_type == 'dask':
            df = task.backend._data
            
            if self.frac is not None:
                sampled_df = df.sample(frac=self.frac, replace=self.replace, 
                                     random_state=self.random_state)
            else:
                # For n samples, we need to know total size
                total_rows = len(df)
                frac = min(1.0, self.n / total_rows)
                sampled_df = df.sample(frac=frac, replace=self.replace,
                                     random_state=self.random_state)
                # Take first n rows to ensure exact count
                sampled_df = sampled_df.head(self.n)
                
            from ..backends import DataBackendDask
            new_backend = DataBackendDask(sampled_df)
            
        elif backend_type == 'vaex':
            df = task.backend._data
            
            if self.frac is not None:
                sampled_df = df.sample(frac=self.frac, random_state=self.random_state)
            else:
                sampled_df = df.sample(n=self.n, random_state=self.random_state)
                
            from ..backends import DataBackendVaex
            new_backend = DataBackendVaex(sampled_df)
            
        else:  # pandas
            data = task.data()
            
            sampled_data = data.sample(
                n=self.n,
                frac=self.frac,
                replace=self.replace,
                random_state=self.random_state
            )
            
            new_backend = type(task.backend)(sampled_data)
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            backend=new_backend,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Preserve column roles, reset row roles
        new_task.set_col_roles({role: list(cols) for role, cols in task.col_roles.items()})
        new_task.set_row_roles({})
        
        self.state.is_trained = True
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """No sampling during prediction - pass through."""
        self.validate_inputs(inputs, "predict")
        
        # During prediction, we typically don't sample
        # Just pass through the data unchanged
        return {"output": inputs["input"]}


class LazyPipeOpCache(LazyPipeOp):
    """Cache/persist operation for lazy DataFrames.
    
    This operation triggers computation and caches results in memory
    (for Dask) or creates efficient storage (for Vaex).
    
    Parameters
    ----------
    id : str, default="lazy_cache"
        Unique identifier.
    """
    
    def __init__(self, id: str = "lazy_cache", **kwargs):
        super().__init__(id=id, **kwargs)
        
    @property
    def input(self) -> Dict[str, PipeOpInput]:
        """Expects a Task."""
        return {
            "input": PipeOpInput(
                name="input",
                train=Task,
                predict=Task
            )
        }
        
    @property
    def output(self) -> Dict[str, PipeOpOutput]:
        """Returns the same Task with cached data."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Cache data during training."""
        self.validate_inputs(inputs, "train")
        task = inputs["input"]
        
        backend_type = self._get_backend_type(task)
        
        if backend_type == 'dask':
            # Persist Dask DataFrame in memory
            task.backend.persist()
        elif backend_type == 'vaex':
            # Vaex is already memory-mapped, nothing to do
            pass
        else:
            # Pandas is already in memory
            pass
            
        self.state.is_trained = True
        
        return {"output": task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pass through during prediction."""
        return {"output": inputs["input"]}


# Register lazy operations
mlpy_pipeops.register("lazy_scale", LazyPipeOpScale)
mlpy_pipeops.register("lazy_filter", LazyPipeOpFilter)
mlpy_pipeops.register("lazy_sample", LazyPipeOpSample)
mlpy_pipeops.register("lazy_cache", LazyPipeOpCache)


__all__ = [
    "LazyPipeOp",
    "LazyPipeOpScale",
    "LazyPipeOpFilter",
    "LazyPipeOpSample",
    "LazyPipeOpCache"
]