"""Task implementations for MLPY."""

from .base import Task
from .supervised import TaskSupervised, TaskClassif, TaskRegr
from .cluster import TaskCluster
from .timeseries import TaskTimeSeries, TaskForecasting, TaskTimeSeriesClassification
from .spatial import TaskClassifSpatial, TaskRegrSpatial, create_spatial_task

# Enhanced task creation with validation
from ..validation import ValidatedTask, validate_task_data

# Optional imports for big data support
try:
    from .big_data import (
        task_from_dask,
        task_from_vaex,
        task_from_csv_lazy,
        task_from_parquet_lazy
    )
    BIG_DATA_AVAILABLE = True
except ImportError:
    BIG_DATA_AVAILABLE = False

__all__ = [
    "Task", 
    "TaskSupervised", 
    "TaskClassif", 
    "TaskRegr", 
    "TaskCluster", 
    "TaskTimeSeries", 
    "TaskForecasting", 
    "TaskTimeSeriesClassification",
    "TaskClassifSpatial",
    "TaskRegrSpatial",
    "create_spatial_task",
    "ValidatedTask",
    "validate_task_data"
]

# Add big data functions if available
if BIG_DATA_AVAILABLE:
    __all__.extend([
        "task_from_dask",
        "task_from_vaex", 
        "task_from_csv_lazy",
        "task_from_parquet_lazy"
    ])