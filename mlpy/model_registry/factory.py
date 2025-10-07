"""
Model Factory for MLPY.

This module provides factory functions for creating learner instances
from registry metadata and managing model instantiation.
"""

from typing import Any, Dict, Optional, Type, Union
import importlib
import inspect
from ..learners.base import Learner


class ModelFactory:
    """Factory for creating model instances from registry metadata."""
    
    @staticmethod
    def create_learner(
        class_path: str,
        init_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Learner:
        """
        Create a learner instance from class path.
        
        Parameters
        ----------
        class_path : str
            Fully qualified class path (e.g., 'mlpy.learners.sklearn.RandomForest')
        init_params : dict, optional
            Initialization parameters for the learner
        **kwargs
            Additional parameters to pass to the learner
        
        Returns
        -------
        Learner
            Instantiated learner object
        """
        try:
            # Parse module and class name
            module_path, class_name = class_path.rsplit('.', 1)
            
            # Import module
            module = importlib.import_module(module_path)
            
            # Get class
            learner_class = getattr(module, class_name)
            
            # Validate it's a Learner subclass
            if not inspect.isclass(learner_class) or not issubclass(learner_class, Learner):
                raise TypeError(f"{class_path} is not a valid Learner class")
            
            # Combine parameters
            params = init_params or {}
            params.update(kwargs)
            
            # Create instance
            return learner_class(**params)
            
        except ImportError as e:
            raise ImportError(f"Could not import {class_path}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found in {module_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create learner from {class_path}: {e}")
    
    @staticmethod
    def get_learner_class(class_path: str) -> Type[Learner]:
        """
        Get learner class without instantiating.
        
        Parameters
        ----------
        class_path : str
            Fully qualified class path
        
        Returns
        -------
        type
            The learner class
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    @staticmethod
    def validate_params(
        learner_class: Type[Learner],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate parameters for a learner class.
        
        Parameters
        ----------
        learner_class : type
            The learner class to validate against
        params : dict
            Parameters to validate
        
        Returns
        -------
        dict
            Validated parameters
        """
        # Get init signature
        sig = inspect.signature(learner_class.__init__)
        
        # Get valid parameter names
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        # Check for **kwargs
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        
        # Validate parameters
        validated = {}
        invalid = []
        
        for key, value in params.items():
            if key in valid_params or has_kwargs:
                validated[key] = value
            else:
                invalid.append(key)
        
        if invalid and not has_kwargs:
            import warnings
            warnings.warn(
                f"Invalid parameters for {learner_class.__name__}: {invalid}",
                UserWarning
            )
        
        return validated


class ModelBuilder:
    """Builder pattern for complex model construction."""
    
    def __init__(self):
        self.class_path = None
        self.params = {}
        self.preprocessors = []
        self.postprocessors = []
    
    def with_model(self, class_path: str) -> "ModelBuilder":
        """Set the model class."""
        self.class_path = class_path
        return self
    
    def with_params(self, **params) -> "ModelBuilder":
        """Add parameters."""
        self.params.update(params)
        return self
    
    def with_preprocessor(self, preprocessor) -> "ModelBuilder":
        """Add a preprocessor."""
        self.preprocessors.append(preprocessor)
        return self
    
    def with_postprocessor(self, postprocessor) -> "ModelBuilder":
        """Add a postprocessor."""
        self.postprocessors.append(postprocessor)
        return self
    
    def build(self) -> Learner:
        """Build the model."""
        if not self.class_path:
            raise ValueError("Model class path not specified")
        
        # Create base model
        model = ModelFactory.create_learner(self.class_path, self.params)
        
        # Wrap with processors if needed
        if self.preprocessors or self.postprocessors:
            from ..learners.pipeline import PipelineLearner
            model = PipelineLearner(
                learner=model,
                preprocessors=self.preprocessors,
                postprocessors=self.postprocessors
            )
        
        return model


def create_ensemble(
    base_learners: list,
    ensemble_type: str = "voting",
    **ensemble_params
) -> Learner:
    """
    Convenience function to create ensemble models.
    
    Parameters
    ----------
    base_learners : list
        List of base learner configurations or instances
    ensemble_type : str
        Type of ensemble ('voting', 'stacking', 'boosting')
    **ensemble_params
        Parameters for the ensemble
    
    Returns
    -------
    Learner
        Ensemble learner instance
    """
    # Convert configurations to instances if needed
    learners = []
    for learner in base_learners:
        if isinstance(learner, dict):
            learners.append(ModelFactory.create_learner(**learner))
        elif isinstance(learner, str):
            learners.append(ModelFactory.create_learner(learner))
        else:
            learners.append(learner)
    
    # Create ensemble
    if ensemble_type == "voting":
        from ..learners.ensemble import LearnerVoting
        return LearnerVoting(base_learners=learners, **ensemble_params)
    elif ensemble_type == "stacking":
        from ..learners.ensemble import LearnerStacking
        return LearnerStacking(base_learners=learners, **ensemble_params)
    elif ensemble_type == "boosting":
        from ..learners.ensemble_advanced import LearnerBoosting
        return LearnerBoosting(base_learners=learners, **ensemble_params)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def create_pipeline(
    steps: list,
    **pipeline_params
) -> Learner:
    """
    Create a pipeline of learners and transformers.
    
    Parameters
    ----------
    steps : list
        List of (name, component) tuples
    **pipeline_params
        Additional pipeline parameters
    
    Returns
    -------
    Learner
        Pipeline learner instance
    """
    from ..learners.pipeline import PipelineLearner
    
    # Process steps
    processed_steps = []
    for name, component in steps:
        if isinstance(component, str):
            component = ModelFactory.create_learner(component)
        elif isinstance(component, dict):
            component = ModelFactory.create_learner(**component)
        processed_steps.append((name, component))
    
    return PipelineLearner(steps=processed_steps, **pipeline_params)


def create_model(
    model_name: str,
    **params
) -> Learner:
    """
    Create a model instance by name from the registry.
    
    Parameters
    ----------
    model_name : str
        Name of the model in the registry
    **params
        Parameters for model initialization
    
    Returns
    -------
    Learner
        Instantiated model
    """
    from .registry import ModelRegistry
    
    registry = ModelRegistry()
    registry.initialize()
    
    metadata = registry.get(model_name)
    if metadata is None:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    return ModelFactory.create_learner(metadata.class_path, params)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model from the registry.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    
    Returns
    -------
    dict
        Model information including metadata
    """
    from .registry import ModelRegistry
    
    registry = ModelRegistry()
    registry.initialize()
    
    metadata = registry.get(model_name)
    if metadata is None:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    return {
        'name': metadata.name,
        'display_name': metadata.display_name,
        'description': metadata.description,
        'category': metadata.category.value,
        'class_path': metadata.class_path,
        'task_types': [t.value for t in metadata.task_types],
        'complexity': metadata.complexity.value,
        'dependencies': metadata.dependencies,
        'hyperparameters': metadata.hyperparameters,
        'capabilities': metadata.capabilities,
        'limitations': metadata.limitations
    }


# Export main components
__all__ = [
    'ModelFactory',
    'ModelBuilder',
    'create_ensemble',
    'create_pipeline',
    'create_model',
    'get_model_info'
]