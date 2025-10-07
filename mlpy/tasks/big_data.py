"""Task helpers for big data backends.

This module provides convenience functions for creating tasks
with Dask or Vaex backends for handling large datasets.
"""

from typing import Optional, Union, Dict, Any, List
import warnings

from .supervised import TaskClassif, TaskRegr
from ..backends import DataBackend

# Check backend availability
try:
    from ..backends import DataBackendDask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    DataBackendDask = None

try:
    from ..backends import DataBackendVaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False
    DataBackendVaex = None


def task_from_dask(
    df: "dask.dataframe.DataFrame",
    target: str,
    task_type: str = "auto",
    id: Optional[str] = None,
    label: Optional[str] = None,
    positive: Optional[str] = None,
    **kwargs
) -> Union[TaskClassif, TaskRegr]:
    """Create a task from a Dask DataFrame.
    
    Parameters
    ----------
    df : dask.dataframe.DataFrame
        The Dask DataFrame containing features and target.
    target : str
        Name of the target column.
    task_type : str, default="auto"
        Type of task: "classification", "regression", or "auto" to infer.
    id : str, optional
        Task identifier.
    label : str, optional
        Human-readable label.
    positive : str, optional
        Positive class for binary classification.
    **kwargs
        Additional arguments passed to Task constructor.
        
    Returns
    -------
    Task
        TaskClassif or TaskRegr with Dask backend.
        
    Examples
    --------
    >>> import dask.dataframe as dd
    >>> df = dd.read_csv('large_dataset.csv')
    >>> task = task_from_dask(df, target='y')
    """
    if not DASK_AVAILABLE:
        raise ImportError(
            "Dask is not installed. Install it with: pip install dask[dataframe]"
        )
        
    # Create backend
    backend = DataBackendDask(df)
    
    # Infer task type if needed
    if task_type == "auto":
        # Sample a small portion to check target type
        target_sample = df[target].head(1000)
        n_unique = target_sample.nunique().compute()
        
        if n_unique <= 20:  # Likely classification
            task_type = "classification"
        else:
            task_type = "regression"
            
    # Create appropriate task
    if task_type in ["classification", "classif"]:
        return TaskClassif(
            backend=backend,
            target=target,
            id=id or "dask_classif",
            label=label,
            positive=positive,
            **kwargs
        )
    else:
        return TaskRegr(
            backend=backend,
            target=target,
            id=id or "dask_regr",
            label=label,
            **kwargs
        )


def task_from_vaex(
    df: "vaex.DataFrame",
    target: str,
    task_type: str = "auto",
    id: Optional[str] = None,
    label: Optional[str] = None,
    positive: Optional[str] = None,
    **kwargs
) -> Union[TaskClassif, TaskRegr]:
    """Create a task from a Vaex DataFrame.
    
    Parameters
    ----------
    df : vaex.DataFrame
        The Vaex DataFrame containing features and target.
    target : str
        Name of the target column.
    task_type : str, default="auto"
        Type of task: "classification", "regression", or "auto" to infer.
    id : str, optional
        Task identifier.
    label : str, optional
        Human-readable label.
    positive : str, optional
        Positive class for binary classification.
    **kwargs
        Additional arguments passed to Task constructor.
        
    Returns
    -------
    Task
        TaskClassif or TaskRegr with Vaex backend.
        
    Examples
    --------
    >>> import vaex
    >>> df = vaex.open('large_dataset.hdf5')
    >>> task = task_from_vaex(df, target='y')
    """
    if not VAEX_AVAILABLE:
        raise ImportError(
            "Vaex is not installed. Install it with: pip install vaex"
        )
        
    # Create backend
    backend = DataBackendVaex(df)
    
    # Infer task type if needed
    if task_type == "auto":
        # Get unique values efficiently
        n_unique = df[target].nunique()
        
        if n_unique <= 20:  # Likely classification
            task_type = "classification"
        else:
            task_type = "regression"
            
    # Create appropriate task
    if task_type in ["classification", "classif"]:
        return TaskClassif(
            backend=backend,
            target=target,
            id=id or "vaex_classif",
            label=label,
            positive=positive,
            **kwargs
        )
    else:
        return TaskRegr(
            backend=backend,
            target=target,
            id=id or "vaex_regr",
            label=label,
            **kwargs
        )


def task_from_csv_lazy(
    filepath: str,
    target: str,
    backend: str = "dask",
    task_type: str = "auto",
    chunksize: Optional[int] = None,
    **kwargs
) -> Union[TaskClassif, TaskRegr]:
    """Create a task from CSV file(s) using lazy loading.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file(s). Can include wildcards for multiple files.
    target : str
        Name of the target column.
    backend : str, default="dask"
        Backend to use: "dask" or "vaex".
    task_type : str, default="auto"
        Type of task: "classification", "regression", or "auto".
    chunksize : int, optional
        Rows per chunk/partition (Dask only).
    **kwargs
        Additional arguments passed to read functions and Task constructor.
        
    Returns
    -------
    Task
        Task with appropriate big data backend.
        
    Examples
    --------
    >>> # Single large file
    >>> task = task_from_csv_lazy('data.csv', target='y')
    >>> 
    >>> # Multiple files with pattern
    >>> task = task_from_csv_lazy('data_*.csv', target='y', backend='vaex')
    """
    if backend == "dask":
        if not DASK_AVAILABLE:
            raise ImportError("Dask not available. Install with: pip install dask[dataframe]")
            
        import dask.dataframe as dd
        
        # Read CSV with Dask
        read_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['id', 'label', 'positive', 'task_type']}
        if chunksize:
            read_kwargs['blocksize'] = f"{chunksize}KB"
            
        df = dd.read_csv(filepath, **read_kwargs)
        
        # Create task
        task_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['id', 'label', 'positive']}
        return task_from_dask(df, target, task_type, **task_kwargs)
        
    elif backend == "vaex":
        if not VAEX_AVAILABLE:
            raise ImportError("Vaex not available. Install with: pip install vaex")
            
        import vaex
        
        # Read CSV with Vaex (converts to HDF5 for efficiency)
        read_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['id', 'label', 'positive', 'task_type']}
        df = vaex.from_csv(filepath, convert=True, **read_kwargs)
        
        # Create task
        task_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['id', 'label', 'positive']}
        return task_from_vaex(df, target, task_type, **task_kwargs)
        
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'dask' or 'vaex'.")


def task_from_parquet_lazy(
    filepath: str,
    target: str,
    backend: str = "dask",
    task_type: str = "auto",
    **kwargs
) -> Union[TaskClassif, TaskRegr]:
    """Create a task from Parquet file(s) using lazy loading.
    
    Parameters
    ----------
    filepath : str
        Path to Parquet file(s).
    target : str
        Name of the target column.
    backend : str, default="dask"
        Backend to use: "dask" or "vaex".
    task_type : str, default="auto"
        Type of task: "classification", "regression", or "auto".
    **kwargs
        Additional arguments passed to read functions and Task constructor.
        
    Returns
    -------
    Task
        Task with appropriate big data backend.
        
    Examples
    --------
    >>> task = task_from_parquet_lazy('data.parquet', target='y')
    """
    if backend == "dask":
        if not DASK_AVAILABLE:
            raise ImportError("Dask not available. Install with: pip install dask[dataframe]")
            
        import dask.dataframe as dd
        
        # Read Parquet with Dask
        read_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['id', 'label', 'positive', 'task_type']}
        df = dd.read_parquet(filepath, **read_kwargs)
        
        # Create task
        task_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['id', 'label', 'positive']}
        return task_from_dask(df, target, task_type, **task_kwargs)
        
    elif backend == "vaex":
        if not VAEX_AVAILABLE:
            raise ImportError("Vaex not available. Install with: pip install vaex")
            
        import vaex
        
        # Read Parquet with Vaex
        read_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['id', 'label', 'positive', 'task_type']}
        df = vaex.open(filepath, **read_kwargs)
        
        # Create task
        task_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['id', 'label', 'positive']}
        return task_from_vaex(df, target, task_type, **task_kwargs)
        
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'dask' or 'vaex'.")