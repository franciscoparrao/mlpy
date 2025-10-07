"""Base classes for resampling strategies in MLPY."""

from abc import abstractmethod
from typing import Optional, List, Tuple, Dict, Any, Iterator
import numpy as np

from ..core.base import MLPYObject
from ..tasks import Task
from ..utils.registry import mlpy_resamplings


class Resampling(MLPYObject):
    """Abstract base class for resampling strategies.
    
    A resampling strategy defines how to split data into training and test sets
    for model evaluation. It can be instantiated with a task to create fixed
    splits that can be iterated over.
    
    Parameters
    ----------
    id : str
        Unique identifier for the resampling strategy.
    param_set : dict, optional
        Parameters for the resampling strategy.
    iters : int
        Number of resampling iterations.
    duplicated_ids : bool, default=False
        Whether IDs can be duplicated in train/test sets (e.g., bootstrap).
    """
    
    def __init__(
        self,
        id: str,
        param_set: Optional[Dict[str, Any]] = None,
        iters: int = 1,
        duplicated_ids: bool = False
    ):
        super().__init__(id=id)
        self.param_set = param_set if param_set is not None else {}
        self.iters = iters
        self.duplicated_ids = duplicated_ids
        self._task = None
        self._instance = None
        
    @property
    def _properties(self):
        """Return resampling properties for hashing."""
        props = set()
        if self.duplicated_ids:
            props.add("duplicated_ids")
        if hasattr(self, 'stratify') and self.stratify:
            props.add("stratified")
        return props
        
    @property
    def is_instantiated(self) -> bool:
        """Check if resampling has been instantiated with a task."""
        return self._instance is not None
        
    def instantiate(self, task: Task) -> "Resampling":
        """Instantiate the resampling with a specific task.
        
        This fixes the train/test splits based on the task's data.
        
        Parameters
        ----------
        task : Task
            The task to instantiate the resampling for.
            
        Returns
        -------
        Resampling
            Self for method chaining.
        """
        self._task = task
        self._instance = self._materialize(task)
        return self
        
    @abstractmethod
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Materialize the resampling splits for a task.
        
        Parameters
        ----------
        task : Task
            The task to create splits for.
            
        Returns
        -------
        dict
            Dictionary containing the materialized splits.
        """
        pass
        
    def train_set(self, i: int) -> np.ndarray:
        """Get training indices for iteration i.
        
        Parameters
        ----------
        i : int
            Iteration number (0-based).
            
        Returns
        -------
        np.ndarray
            Array of row indices for training.
        """
        if not self.is_instantiated:
            raise RuntimeError("Resampling must be instantiated before accessing splits")
        if not 0 <= i < self.iters:
            raise IndexError(f"Iteration {i} out of range [0, {self.iters})")
            
        return self._get_train_set(i)
        
    def test_set(self, i: int) -> np.ndarray:
        """Get test indices for iteration i.
        
        Parameters
        ----------
        i : int
            Iteration number (0-based).
            
        Returns
        -------
        np.ndarray
            Array of row indices for testing.
        """
        if not self.is_instantiated:
            raise RuntimeError("Resampling must be instantiated before accessing splits")
        if not 0 <= i < self.iters:
            raise IndexError(f"Iteration {i} out of range [0, {self.iters})")
            
        return self._get_test_set(i)
        
    @abstractmethod
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices for iteration i (internal).
        
        Parameters
        ----------
        i : int
            Iteration number.
            
        Returns
        -------
        np.ndarray
            Training indices.
        """
        pass
        
    @abstractmethod
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test indices for iteration i (internal).
        
        Parameters
        ----------
        i : int
            Iteration number.
            
        Returns
        -------
        np.ndarray
            Test indices.
        """
        pass
        
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over train/test splits.
        
        Yields
        ------
        train_indices : np.ndarray
            Training set indices.
        test_indices : np.ndarray
            Test set indices.
        """
        if not self.is_instantiated:
            raise RuntimeError("Resampling must be instantiated before iteration")
            
        for i in range(self.iters):
            yield self.train_set(i), self.test_set(i)
            
    def clone(self, deep: bool = True) -> "Resampling":
        """Clone the resampling strategy.
        
        Parameters
        ----------
        deep : bool, default=True
            Whether to deep copy internal state.
            
        Returns
        -------
        Resampling
            Cloned resampling (not instantiated).
        """
        cloned = super().clone(deep=deep)
        # Reset instantiation
        cloned._task = None
        cloned._instance = None
        return cloned
        
    def __repr__(self):
        status = "instantiated" if self.is_instantiated else "not instantiated"
        return f"<{self.__class__.__name__}[{status}]:{self.id}>"


def register_resampling(cls):
    """Decorator to register a resampling class.
    
    Parameters
    ----------
    cls : type
        The resampling class to register.
        
    Returns
    -------
    type
        The unchanged class.
    """
    # Create instance with default parameters
    instance = cls()
    mlpy_resamplings[instance.id] = instance
    return cls