"""
Enhanced error handling with helpful messages and suggestions.
"""

from typing import Any, Optional, Dict, List
import traceback
import difflib


class MLPYValidationError(Exception):
    """Base exception for all MLPY validation errors."""
    
    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
        self.message = message
        self.suggestions = suggestions or []
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with suggestions."""
        msg = f"\n{'='*60}\n"
        msg += f"❌ MLPY VALIDATION ERROR\n"
        msg += f"{'='*60}\n\n"
        msg += f"{self.message}\n"
        
        if self.suggestions:
            msg += f"\n💡 SUGGESTIONS:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"   {i}. {suggestion}\n"
        
        msg += f"\n{'='*60}\n"
        return msg


class TaskValidationError(MLPYValidationError):
    """Task-specific validation errors."""
    pass


class LearnerValidationError(MLPYValidationError):
    """Learner-specific validation errors."""
    pass


class MeasureValidationError(MLPYValidationError):
    """Measure-specific validation errors."""
    pass


class PipelineValidationError(MLPYValidationError):
    """Pipeline-specific validation errors."""
    pass


def provide_helpful_error(error: Exception, context: Dict[str, Any]) -> MLPYValidationError:
    """
    Convert generic errors to helpful MLPY errors with context and suggestions.
    
    Parameters
    ----------
    error : Exception
        The original error
    context : dict
        Context information (what was being attempted)
    
    Returns
    -------
    MLPYValidationError
        Enhanced error with helpful messages
    """
    
    error_str = str(error)
    suggestions = []
    
    # Common error patterns and their solutions
    if "has no attribute" in error_str:
        # Extract what attribute was missing
        parts = error_str.split("'")
        if len(parts) >= 3:
            obj_type = parts[1]
            attribute = parts[3] if len(parts) > 3 else "unknown"
            
            if obj_type == "NoneType":
                suggestions.append(
                    "The object is None. Check if previous steps completed successfully."
                )
                suggestions.append(
                    "Ensure all required parameters are provided."
                )
            else:
                # Try to find similar attributes
                if 'object' in context:
                    obj = context['object']
                    available = dir(obj)
                    similar = difflib.get_close_matches(attribute, available, n=3)
                    if similar:
                        suggestions.append(
                            f"Did you mean one of these? {similar}"
                        )
    
    elif "TypeError" in str(type(error)):
        if "got an unexpected keyword argument" in error_str:
            # Extract the unexpected argument
            parts = error_str.split("'")
            if len(parts) >= 2:
                bad_arg = parts[1]
                suggestions.append(
                    f"Remove the '{bad_arg}' parameter or check the documentation."
                )
        
        elif "missing" in error_str and "required positional argument" in error_str:
            # Extract missing argument
            parts = error_str.split("'")
            if len(parts) >= 2:
                missing_arg = parts[1]
                suggestions.append(
                    f"Add the required '{missing_arg}' parameter."
                )
                suggestions.append(
                    "Check the function signature in the documentation."
                )
    
    elif "ValueError" in str(type(error)):
        if "could not convert" in error_str:
            suggestions.append(
                "Check data types. Ensure numeric columns don't contain strings."
            )
            suggestions.append(
                "Use pd.to_numeric() or handle non-numeric values."
            )
        
        elif "shape" in error_str:
            suggestions.append(
                "Check array dimensions match expected shape."
            )
            suggestions.append(
                "Use .reshape() or verify data preprocessing steps."
            )
    
    elif "KeyError" in str(type(error)):
        key = str(error).strip("'\"")
        if 'available_keys' in context:
            similar = difflib.get_close_matches(key, context['available_keys'], n=3)
            if similar:
                suggestions.append(
                    f"Key '{key}' not found. Did you mean one of these? {similar}"
                )
        else:
            suggestions.append(
                f"Key '{key}' not found. Check available keys with .keys() or dir()"
            )
    
    elif "ImportError" in str(type(error)):
        if "No module named" in error_str:
            module = error_str.split("'")[1]
            suggestions.append(
                f"Install missing module: pip install {module}"
            )
            suggestions.append(
                "Check if you're in the correct virtual environment"
            )
    
    # Add context-specific suggestions
    if 'operation' in context:
        operation = context['operation']
        if operation == 'benchmark':
            suggestions.append(
                "Ensure task, learners, and measures are compatible"
            )
            suggestions.append(
                "Try with a simple example first to isolate the issue"
            )
        elif operation == 'spatial':
            suggestions.append(
                "Verify coordinate columns exist and contain valid values"
            )
            suggestions.append(
                "Check if task type includes 'spatial' suffix"
            )
    
    # Create enhanced error
    enhanced_message = f"{error_str}\n\nContext: {context.get('description', 'Unknown operation')}"
    
    return MLPYValidationError(enhanced_message, suggestions)


class ErrorContext:
    """Context manager for enhanced error reporting."""
    
    def __init__(self, operation: str, **kwargs):
        self.operation = operation
        self.context = kwargs
        self.context['operation'] = operation
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # Enhance the error with context
            enhanced = provide_helpful_error(
                exc_val,
                self.context
            )
            raise enhanced from exc_val
        return False


# Common error templates for quick use
class ErrorTemplates:
    """Pre-defined error messages for common scenarios."""
    
    @staticmethod
    def wrong_task_type(expected: str, got: str) -> TaskValidationError:
        return TaskValidationError(
            f"Expected task type '{expected}', but got '{got}'",
            suggestions=[
                f"Change task_type to '{expected}'",
                f"Or use a different measure/learner compatible with '{got}'"
            ]
        )
    
    @staticmethod
    def missing_coordinates() -> TaskValidationError:
        return TaskValidationError(
            "Spatial task requires coordinate columns",
            suggestions=[
                "Add coordinate_names=['x_col', 'y_col'] when creating task",
                "Or use non-spatial task type (remove '_spatial' suffix)"
            ]
        )
    
    @staticmethod
    def incompatible_measure(measure: str, task_type: str) -> MeasureValidationError:
        return MeasureValidationError(
            f"Measure '{measure}' is not compatible with task type '{task_type}'",
            suggestions=[
                "Use measures designed for this task type",
                "Or change the task type to match the measure",
                f"For '{task_type}', try measures with matching prefix"
            ]
        )
    
    @staticmethod  
    def class_vs_instance(class_name: str) -> MLPYValidationError:
        return MLPYValidationError(
            f"You passed the class {class_name} instead of an instance",
            suggestions=[
                f"Add parentheses: {class_name}() to create an instance",
                f"Example: {class_name}(n_folds=5) with parameters"
            ]
        )