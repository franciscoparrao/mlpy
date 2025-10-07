"""
Task validation using Pydantic for robust type checking.
"""

from typing import Optional, List, Union, Any, Literal
from pydantic import BaseModel, field_validator, Field, model_validator
import pandas as pd
import numpy as np
from pathlib import Path


class TaskConfig(BaseModel):
    """Base configuration for all tasks with strict validation."""
    
    id: str = Field(..., min_length=1, description="Unique task identifier")
    task_type: Literal['classif', 'regr', 'classif_spatial', 'regr_spatial']
    backend: Union[pd.DataFrame, np.ndarray, str, Path] = Field(
        ..., 
        description="Data source: DataFrame, array, or path to file"
    )
    target: Union[str, int] = Field(
        ...,
        description="Target column name or index"
    )
    feature_names: Optional[List[str]] = None
    row_roles: Optional[dict] = None
    col_roles: Optional[dict] = None
    
    class Config:
        arbitrary_types_allowed = True
        
    @field_validator('backend')
    def validate_backend(cls, v, info):
        """Validate and potentially load data backend."""
        if isinstance(v, (str, Path)):
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Data file not found: {path}")
            
            # Load based on extension
            if path.suffix == '.csv':
                return pd.read_csv(path)
            elif path.suffix == '.parquet':
                return pd.read_parquet(path)
            elif path.suffix in ['.pkl', '.pickle']:
                return pd.read_pickle(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        if isinstance(v, pd.DataFrame):
            if v.empty:
                raise ValueError("DataFrame is empty")
        elif isinstance(v, np.ndarray):
            if v.size == 0:
                raise ValueError("Array is empty")
            if len(v.shape) != 2:
                raise ValueError(f"Expected 2D array, got shape {v.shape}")
        
        return v
    
    @field_validator('target')
    def validate_target(cls, v, info):
        """Validate target exists in backend."""
        backend = info.data.get('backend')
        if backend is None:
            return v
            
        if isinstance(backend, pd.DataFrame):
            if isinstance(v, str) and v not in backend.columns:
                available = list(backend.columns)
                raise ValueError(
                    f"Target column '{v}' not found in DataFrame. "
                    f"Available columns: {available[:5]}{'...' if len(available) > 5 else ''}"
                )
        elif isinstance(backend, np.ndarray):
            if isinstance(v, int):
                if v >= backend.shape[1]:
                    raise ValueError(
                        f"Target index {v} out of range for array with {backend.shape[1]} columns"
                    )
        
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate overall consistency of configuration."""
        task_type = self.task_type
        backend = self.backend
        
        # Check for NaN in target
        if isinstance(backend, pd.DataFrame):
            target = self.target
            if target and target in backend.columns:
                if backend[target].isna().any():
                    n_missing = backend[target].isna().sum()
                    raise ValueError(
                        f"Target column '{target}' contains {n_missing} missing values. "
                        "Please handle missing values before creating task."
                    )
        
        return self


class TaskClassifConfig(TaskConfig):
    """Configuration for classification tasks."""
    
    task_type: Literal['classif'] = 'classif'
    n_classes: Optional[int] = Field(None, ge=2)
    class_names: Optional[List[str]] = None
    stratify: bool = True
    
    @field_validator('n_classes')
    def validate_n_classes(cls, v, info):
        """Validate number of classes matches data."""
        backend = info.data.get('backend')
        target = info.data.get('target')
        
        if backend is not None and target is not None:
            if isinstance(backend, pd.DataFrame):
                if target in backend.columns:
                    actual_classes = backend[target].nunique()
                    if v and v != actual_classes:
                        raise ValueError(
                            f"Specified n_classes={v} doesn't match "
                            f"actual classes in data: {actual_classes}"
                        )
                    if not v:
                        v = actual_classes
        
        if v and v < 2:
            raise ValueError(f"Classification requires at least 2 classes, got {v}")
        
        return v


class TaskRegrConfig(TaskConfig):
    """Configuration for regression tasks."""
    
    task_type: Literal['regr'] = 'regr'
    normalize_target: bool = False
    target_transform: Optional[str] = Field(
        None,
        description="Target transformation: 'log', 'sqrt', 'box-cox'"
    )
    
    @field_validator('target_transform')
    def validate_transform(cls, v, info):
        """Validate target transformation is supported."""
        valid_transforms = ['log', 'sqrt', 'box-cox', None]
        if v not in valid_transforms:
            raise ValueError(
                f"Invalid transform '{v}'. Must be one of {valid_transforms}"
            )
        return v


class TaskSpatialConfig(TaskConfig):
    """Configuration for spatial tasks."""
    
    coordinate_names: List[str] = Field(
        ...,
        min_items=2,
        max_items=3,
        description="Names of coordinate columns [x, y] or [x, y, z]"
    )
    crs: Optional[str] = Field(None, description="Coordinate Reference System")
    buffer_distance: Optional[float] = Field(None, ge=0)
    
    @field_validator('task_type')
    def must_be_spatial(cls, v, info):
        """Ensure task type is spatial."""
        if 'spatial' not in v:
            raise ValueError(
                f"TaskSpatialConfig requires spatial task type, got '{v}'"
            )
        return v
    
    @field_validator('coordinate_names')
    def validate_coordinates(cls, v, info):
        """Validate coordinate columns exist."""
        backend = info.data.get('backend')
        
        if isinstance(backend, pd.DataFrame):
            missing = [col for col in v if col not in backend.columns]
            if missing:
                raise ValueError(
                    f"Coordinate columns not found: {missing}. "
                    f"Available columns: {list(backend.columns)}"
                )
            
            # Check for NaN in coordinates
            for col in v:
                if backend[col].isna().any():
                    n_missing = backend[col].isna().sum()
                    raise ValueError(
                        f"Coordinate column '{col}' contains {n_missing} missing values"
                    )
        
        return v


class ValidatedTask:
    """
    Wrapper class that validates task creation and provides helpful errors.
    
    Examples
    --------
    >>> # This will provide clear error messages
    >>> task = ValidatedTask(
    ...     id='my_task',
    ...     task_type='classif',
    ...     backend=df,
    ...     target='label'
    ... )
    
    >>> # Instead of cryptic: AttributeError: 'NoneType' object...
    >>> # You get: "Target column 'label' not found. Available: ['col1', 'col2']"
    """
    
    def __init__(self, **kwargs):
        # Determine which config to use
        task_type = kwargs.get('task_type', 'classif')
        
        if 'spatial' in task_type:
            config_class = TaskSpatialConfig
        elif task_type == 'classif':
            config_class = TaskClassifConfig
        elif task_type == 'regr':
            config_class = TaskRegrConfig
        else:
            config_class = TaskConfig
        
        try:
            self.config = config_class(**kwargs)
        except Exception as e:
            # Convert Pydantic errors to helpful messages
            raise TaskValidationError(
                f"Task validation failed: {str(e)}\n"
                f"Please check your configuration:\n"
                f"  - task_type: {task_type}\n"
                f"  - backend type: {type(kwargs.get('backend'))}\n"
                f"  - target: {kwargs.get('target')}"
            )
        
        # Create actual task after validation
        self._create_task()
    
    def _create_task(self):
        """Create the actual MLPY task after validation."""
        from ..tasks import TaskClassif, TaskRegr
        from ..tasks.spatial import TaskClassifSpatial, TaskRegrSpatial
        
        config = self.config.dict()
        task_type = config.pop('task_type')
        
        # Map 'backend' to 'data' for MLPY tasks
        if 'backend' in config:
            config['data'] = config.pop('backend')
        
        # Only pass parameters that MLPY tasks actually accept
        # Core parameters that all tasks accept
        task_params = {
            'data': config['data'],
            'target': config['target'],
            'id': config.get('id')
        }
        
        # Add spatial parameters if needed
        if 'coordinate_names' in config:
            task_params['coordinate_names'] = config['coordinate_names']
        if 'crs' in config:
            task_params['crs'] = config['crs']
        if 'buffer_distance' in config:
            task_params['buffer_distance'] = config['buffer_distance']
        
        if task_type == 'classif':
            self.task = TaskClassif(**task_params)
        elif task_type == 'regr':
            self.task = TaskRegr(**task_params)
        elif task_type == 'classif_spatial':
            self.task = TaskClassifSpatial(**task_params)
        elif task_type == 'regr_spatial':
            self.task = TaskRegrSpatial(**task_params)
    
    def __getattr__(self, name):
        """Delegate to underlying task."""
        return getattr(self.task, name)


class TaskValidationError(Exception):
    """Custom exception for task validation errors with helpful messages."""
    pass


def validate_task_data(data: Any, target: Any = None) -> dict:
    """
    Quick validation of data for task creation.
    
    Returns dict with validation results and suggestions.
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'suggestions': []
    }
    
    # Check data type
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        results['valid'] = False
        results['errors'].append(
            f"Data must be DataFrame or array, got {type(data)}"
        )
        return results
    
    # Check size
    if isinstance(data, pd.DataFrame):
        n_rows, n_cols = data.shape
        if n_rows < 10:
            results['warnings'].append(
                f"Very few samples ({n_rows}). Consider getting more data."
            )
        if n_cols < 2:
            results['errors'].append(
                f"Need at least 2 columns (features + target), got {n_cols}"
            )
            results['valid'] = False
    
    # Check for missing values
    if isinstance(data, pd.DataFrame):
        missing = data.isna().sum()
        if missing.any():
            results['warnings'].append(
                f"Found missing values in {missing[missing > 0].index.tolist()}"
            )
            results['suggestions'].append(
                "Consider using SimpleImputer or dropping missing values"
            )
    
    # Check target
    if target and isinstance(data, pd.DataFrame):
        if target not in data.columns:
            results['valid'] = False
            results['errors'].append(
                f"Target '{target}' not found in columns"
            )
            similar = [col for col in data.columns if target.lower() in col.lower()]
            if similar:
                results['suggestions'].append(
                    f"Did you mean one of these? {similar}"
                )
    
    return results