"""Base learner class for MLPY.

This module provides the abstract base class for all learners
in the MLPY framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict

from ..base import MLPYObject
from ..tasks import Task
from ..predictions import Prediction


class Learner(MLPYObject, ABC):
    """Abstract base class for all learners.
    
    A learner encapsulates a machine learning algorithm that can be
    trained on data and used to make predictions.
    
    Parameters
    ----------
    id : str
        Unique identifier for the learner.
    param_set : dict, optional
        Parameters for the learner.
    predict_type : str, optional
        Type of prediction ("response", "prob", "se").
    feature_names : list of str, optional
        Names of features the learner can use.
    predict_sets : list of str, optional
        Which sets the learner can predict on ("train", "test").
    properties : set of str, optional
        Additional properties of the learner.
    packages : list of str, optional
        Required packages for the learner.
    """
    
    def __init__(
        self,
        id: str,
        param_set: Optional[Dict[str, Any]] = None,
        predict_type: str = "response",
        feature_names: Optional[List[str]] = None,
        predict_sets: Optional[List[str]] = None,
        properties: Optional[set] = None,
        packages: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        
        self.param_set = param_set if param_set is not None else {}
        self.predict_type = predict_type
        self.feature_names = feature_names
        self.predict_sets = predict_sets or ["test"]
        self.properties = properties or set()
        self.packages = packages or []
        
        # State
        self._model = None
        self._train_task = None
        self._train_time = None
        
    @property
    @abstractmethod
    def task_type(self) -> str:
        """Type of task this learner can handle."""
        pass
        
    @abstractmethod
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "Learner":
        """Train the learner on a task.
        
        Parameters
        ----------
        task : Task
            The task to train on.
        row_ids : list of int, optional
            Subset of rows to use for training.
            If None, all rows in use are used.
            
        Returns
        -------
        self : Learner
            The trained learner.
        """
        pass
        
    @abstractmethod
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> Prediction:
        """Make predictions on a task.
        
        Parameters
        ----------
        task : Task
            The task to predict on.
        row_ids : list of int, optional
            Subset of rows to predict.
            If None, all rows in use are predicted.
            
        Returns
        -------
        Prediction
            The predictions.
        """
        pass
        
    @property
    def is_trained(self) -> bool:
        """Whether the learner has been trained."""
        return self._model is not None
        
    def reset(self) -> "Learner":
        """Reset the learner to untrained state.
        
        Returns
        -------
        self : Learner
            The reset learner.
        """
        self._model = None
        self._train_task = None
        self._train_time = None
        return self
        
    def clone(self, deep: bool = True) -> "Learner":
        """Create a copy of the learner.
        
        Parameters
        ----------
        deep : bool, default=True
            Whether to make a deep copy.
            
        Returns
        -------
        Learner
            A copy of the learner.
        """
        # Subclasses should override this method
        cls = self.__class__
        cloned = cls(**self.__dict__)
        cloned.reset()  # Don't copy trained state
        return cloned
        
    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.is_trained else "untrained"
        return f"<{self.__class__.__name__}:{self.id}> ({status})"