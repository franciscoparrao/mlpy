"""Base classes for model interpretability in MLPY."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from ..base import MLPYObject
from ..learners import Learner
from ..tasks import Task
from ..predictions import Prediction


@dataclass
class FeatureImportance:
    """Container for feature importance scores.
    
    Parameters
    ----------
    features : list of str
        Feature names.
    importances : array-like
        Importance scores for each feature.
    method : str
        Method used to calculate importances.
    """
    features: List[str]
    importances: np.ndarray
    method: str
    
    def __post_init__(self):
        """Validate inputs."""
        if len(self.features) != len(self.importances):
            raise ValueError("Features and importances must have same length")
        self.importances = np.asarray(self.importances)
        
    def as_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame sorted by importance.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with features and importances.
        """
        df = pd.DataFrame({
            'feature': self.features,
            'importance': self.importances
        })
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def top_features(self, n: int = 10) -> List[str]:
        """Get top n most important features.
        
        Parameters
        ----------
        n : int
            Number of top features to return.
            
        Returns
        -------
        list of str
            Top feature names.
        """
        df = self.as_dataframe()
        return df.head(n)['feature'].tolist()


@dataclass
class InterpretationResult(MLPYObject):
    """Base class for interpretation results.
    
    Parameters
    ----------
    learner : Learner
        The learner that was interpreted.
    task : Task
        The task used for interpretation.
    method : str
        Interpretation method used.
    global_importance : FeatureImportance, optional
        Global feature importance.
    local_explanations : dict, optional
        Local explanations for specific instances.
    metadata : dict, optional
        Additional metadata.
    """
    learner: Learner
    task: Task
    method: str
    global_importance: Optional[FeatureImportance] = None
    local_explanations: Optional[Dict[int, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize result."""
        super().__init__(id=f"{self.method}_result")
        
    def get_local_explanation(self, index: int) -> Any:
        """Get local explanation for a specific instance.
        
        Parameters
        ----------
        index : int
            Instance index.
            
        Returns
        -------
        Any
            Local explanation for the instance.
        """
        if self.local_explanations is None:
            raise ValueError("No local explanations available")
        if index not in self.local_explanations:
            raise ValueError(f"No explanation for index {index}")
        return self.local_explanations[index]
    
    def has_global_importance(self) -> bool:
        """Check if global importance is available.
        
        Returns
        -------
        bool
            True if global importance is available.
        """
        return self.global_importance is not None
    
    def has_local_explanations(self) -> bool:
        """Check if local explanations are available.
        
        Returns
        -------
        bool
            True if local explanations are available.
        """
        return self.local_explanations is not None and len(self.local_explanations) > 0


class Interpreter(MLPYObject, ABC):
    """Abstract base class for model interpreters.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    """
    
    def __init__(self, id: Optional[str] = None):
        """Initialize interpreter."""
        super().__init__(id=id or self.__class__.__name__.lower())
        
    @abstractmethod
    def interpret(
        self,
        learner: Learner,
        task: Task,
        indices: Optional[List[int]] = None,
        **kwargs
    ) -> InterpretationResult:
        """Interpret the model.
        
        Parameters
        ----------
        learner : Learner
            Trained learner to interpret.
        task : Task
            Task with data for interpretation.
        indices : list of int, optional
            Indices of instances to explain locally.
        **kwargs
            Additional interpreter-specific arguments.
            
        Returns
        -------
        InterpretationResult
            Interpretation results.
        """
        pass
    
    def check_learner_compatibility(self, learner: Learner) -> bool:
        """Check if learner is compatible with this interpreter.
        
        Parameters
        ----------
        learner : Learner
            Learner to check.
            
        Returns
        -------
        bool
            True if compatible.
        """
        # Default: assume all learners are compatible
        # Subclasses can override for specific requirements
        return True
    
    def get_feature_names(self, task: Task) -> List[str]:
        """Get feature names from task.
        
        Parameters
        ----------
        task : Task
            Task object.
            
        Returns
        -------
        list of str
            Feature names.
        """
        return task.feature_names
    
    def validate_indices(self, indices: List[int], n_samples: int) -> List[int]:
        """Validate and normalize indices.
        
        Parameters
        ----------
        indices : list of int
            Indices to validate.
        n_samples : int
            Total number of samples.
            
        Returns
        -------
        list of int
            Validated indices.
        """
        if not indices:
            return []
            
        # Convert negative indices
        validated = []
        for idx in indices:
            if idx < 0:
                idx = n_samples + idx
            if 0 <= idx < n_samples:
                validated.append(idx)
            else:
                raise ValueError(f"Index {idx} out of range [0, {n_samples})")
                
        return validated