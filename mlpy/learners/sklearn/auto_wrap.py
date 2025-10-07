"""Auto-wrapper functionality for scikit-learn models."""

from typing import Any, Type, Optional, Dict
import inspect
import warnings

from .base import LearnerSKLearn, LearnerClassifSKLearn, LearnerRegrSKLearn


def auto_sklearn(
    estimator: Any,
    id: Optional[str] = None,
    task_type: Optional[str] = None,
    **kwargs
) -> LearnerSKLearn:
    """Automatically wrap a scikit-learn estimator into an MLPY learner.
    
    This function creates an MLPY learner wrapper for any scikit-learn
    compatible estimator. It automatically detects whether the estimator
    is for classification or regression.
    
    Parameters
    ----------
    estimator : estimator instance or class
        A scikit-learn estimator (either instance or class).
    id : str, optional
        Unique identifier for the learner. If None, uses estimator name.
    task_type : str, optional
        Explicitly specify task type ("classif" or "regr").
        If None, attempts to auto-detect.
    **kwargs
        Additional parameters to pass to the wrapper.
        
    Returns
    -------
    LearnerSKLearn
        MLPY learner wrapper for the estimator.
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlpy.learners.sklearn import auto_sklearn
    >>> 
    >>> # Wrap a class
    >>> learner = auto_sklearn(RandomForestClassifier, n_estimators=100)
    >>> 
    >>> # Wrap an instance
    >>> rf = RandomForestClassifier(n_estimators=100)
    >>> learner = auto_sklearn(rf)
    """
    # Check if it's an instance or class
    if inspect.isclass(estimator):
        estimator_class = estimator
        estimator_instance = None
        estimator_params = kwargs
    else:
        # It's an instance
        estimator_class = estimator.__class__
        estimator_instance = estimator
        # Extract parameters from instance
        if hasattr(estimator, 'get_params'):
            estimator_params = estimator.get_params()
            estimator_params.update(kwargs)
        else:
            estimator_params = kwargs
            
    # Auto-detect task type if not specified
    if task_type is None:
        task_type = _detect_task_type(estimator_class, estimator_instance)
        
    # Create ID if not provided
    if id is None:
        id = _create_id(estimator_class)
        
    # Create appropriate wrapper
    if task_type == "classif":
        wrapper_class = _create_classif_wrapper(estimator_class)
    elif task_type == "regr":
        wrapper_class = _create_regr_wrapper(estimator_class)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
        
    # Create and return learner instance
    learner = wrapper_class(id=id, **estimator_params)
    
    # If we had an instance that was already fitted, transfer the model
    if estimator_instance is not None and hasattr(estimator_instance, 'classes_'):
        # The estimator was already fitted
        learner.estimator = estimator_instance
        learner.is_trained = True
        learner.model = estimator_instance
        warnings.warn(
            "The provided estimator instance appears to be already fitted. "
            "The learner will use this fitted model."
        )
        
    return learner


def _detect_task_type(estimator_class: Type, estimator_instance: Any = None) -> str:
    """Detect whether estimator is for classification or regression.
    
    Parameters
    ----------
    estimator_class : type
        Estimator class.
    estimator_instance : estimator, optional
        Estimator instance.
        
    Returns
    -------
    str
        Task type ("classif" or "regr").
    """
    # First check instance if available
    if estimator_instance is not None:
        if hasattr(estimator_instance, '_estimator_type'):
            if estimator_instance._estimator_type == 'classifier':
                return "classif"
            elif estimator_instance._estimator_type == 'regressor':
                return "regr"
                
    # Check class
    if hasattr(estimator_class, '_estimator_type'):
        if estimator_class._estimator_type == 'classifier':
            return "classif"
        elif estimator_class._estimator_type == 'regressor':
            return "regr"
            
    # Check by name
    class_name = estimator_class.__name__.lower()
    if any(x in class_name for x in ['classifier', 'classif', 'logistic']):
        return "classif"
    elif any(x in class_name for x in ['regressor', 'regr', 'regression']):
        return "regr"
        
    # Check for specific methods
    if hasattr(estimator_class, 'predict_proba') or hasattr(estimator_class, 'decision_function'):
        return "classif"
        
    # Default error
    raise ValueError(
        f"Cannot auto-detect task type for {estimator_class.__name__}. "
        "Please specify task_type='classif' or task_type='regr'."
    )


def _create_id(estimator_class: Type) -> str:
    """Create a reasonable ID from estimator class name.
    
    Parameters
    ----------
    estimator_class : type
        Estimator class.
        
    Returns
    -------
    str
        Learner ID.
    """
    name = estimator_class.__name__
    
    # Convert CamelCase to snake_case
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    # Remove common suffixes
    for suffix in ['_classifier', '_regressor', '_classif', '_regr']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            
    return name


def _create_classif_wrapper(estimator_class: Type) -> Type[LearnerClassifSKLearn]:
    """Create a classification wrapper class dynamically.
    
    Parameters
    ----------
    estimator_class : type
        Estimator class to wrap.
        
    Returns
    -------
    type
        Dynamically created wrapper class.
    """
    class DynamicClassifWrapper(LearnerClassifSKLearn):
        """Dynamically created classification wrapper."""
        
        def __init__(self, id: Optional[str] = None, predict_type: str = "response", **kwargs):
            super().__init__(
                estimator_class=estimator_class,
                id=id,
                predict_type=predict_type,
                **kwargs
            )
            
    # Set better class name
    DynamicClassifWrapper.__name__ = f"{estimator_class.__name__}Wrapper"
    DynamicClassifWrapper.__qualname__ = f"{estimator_class.__name__}Wrapper"
    
    return DynamicClassifWrapper


def _create_regr_wrapper(estimator_class: Type) -> Type[LearnerRegrSKLearn]:
    """Create a regression wrapper class dynamically.
    
    Parameters
    ----------
    estimator_class : type
        Estimator class to wrap.
        
    Returns
    -------
    type
        Dynamically created wrapper class.
    """
    class DynamicRegrWrapper(LearnerRegrSKLearn):
        """Dynamically created regression wrapper."""
        
        def __init__(self, id: Optional[str] = None, **kwargs):
            super().__init__(
                estimator_class=estimator_class,
                id=id,
                **kwargs
            )
            
    # Set better class name
    DynamicRegrWrapper.__name__ = f"{estimator_class.__name__}Wrapper"
    DynamicRegrWrapper.__qualname__ = f"{estimator_class.__name__}Wrapper"
    
    return DynamicRegrWrapper


def list_available_sklearn_models() -> Dict[str, list]:
    """List all available scikit-learn models that can be wrapped.
    
    Returns
    -------
    dict
        Dictionary with 'classifiers' and 'regressors' lists.
    """
    models = {
        'classifiers': [],
        'regressors': []
    }
    
    try:
        import sklearn.utils
        
        # Get all estimators
        all_estimators = sklearn.utils.all_estimators()
        
        for name, estimator_class in all_estimators:
            if hasattr(estimator_class, '_estimator_type'):
                if estimator_class._estimator_type == 'classifier':
                    models['classifiers'].append(name)
                elif estimator_class._estimator_type == 'regressor':
                    models['regressors'].append(name)
                    
    except ImportError:
        warnings.warn("scikit-learn not installed, cannot list models")
        
    return models


__all__ = [
    "auto_sklearn",
    "list_available_sklearn_models"
]