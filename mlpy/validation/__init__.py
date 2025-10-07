"""
Validation module for MLPY using Pydantic.

This module provides robust type checking and validation for all MLPY components,
eliminating cryptic errors and providing clear, actionable error messages.
"""

from .task_validators import (
    TaskConfig,
    ValidatedTask,
    TaskClassifConfig,
    TaskRegrConfig,
    TaskSpatialConfig,
    validate_task_data
)

from .errors import (
    MLPYValidationError,
    TaskValidationError,
    LearnerValidationError,
    MeasureValidationError,
    PipelineValidationError,
    provide_helpful_error,
    ErrorContext
)

__all__ = [
    # Task validation
    'TaskConfig',
    'ValidatedTask',
    'TaskClassifConfig', 
    'TaskRegrConfig',
    'TaskSpatialConfig',
    'validate_task_data',
    
    # Errors
    'MLPYValidationError',
    'TaskValidationError',
    'LearnerValidationError',
    'MeasureValidationError',
    'PipelineValidationError',
    'provide_helpful_error',
    'ErrorContext'
]