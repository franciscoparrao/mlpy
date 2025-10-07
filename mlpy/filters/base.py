"""
Base classes for feature filters in MLPY.

Inspired by mlr3filters, providing a comprehensive framework
for feature selection and filtering.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import warnings

from ..tasks.base import Task
from ..tasks.supervised import TaskClassif, TaskRegr


@dataclass
class FilterResult:
    """Result of a filter calculation.
    
    Attributes
    ----------
    scores : pd.Series
        Feature scores indexed by feature names.
    features : List[str]
        Ordered list of features (best to worst).
    method : str
        Filter method used.
    task_type : str
        Type of task (classif/regr).
    params : Dict[str, Any]
        Parameters used for filtering.
    """
    scores: pd.Series
    features: List[str]
    method: str
    task_type: str
    params: Dict[str, Any] = None
    
    def select_top_k(self, k: int) -> List[str]:
        """Select top k features."""
        return self.features[:k]
    
    def select_threshold(self, threshold: float, above: bool = True) -> List[str]:
        """Select features by score threshold."""
        if above:
            mask = self.scores >= threshold
        else:
            mask = self.scores <= threshold
        return self.scores[mask].index.tolist()
    
    def select_percentile(self, percentile: float) -> List[str]:
        """Select top percentile of features."""
        threshold = np.percentile(self.scores, percentile)
        return self.select_threshold(threshold, above=True)


class Filter(ABC):
    """Abstract base class for feature filters.
    
    Parameters
    ----------
    id : str
        Unique identifier for the filter.
    properties : set
        Properties of the filter (e.g., {'missings', 'weights'}).
    packages : set
        Required packages.
    task_types : set
        Supported task types {'classif', 'regr', 'any'}.
    feature_types : set
        Supported feature types {'numeric', 'factor', 'any'}.
    """
    
    def __init__(
        self,
        id: str,
        properties: Optional[set] = None,
        packages: Optional[set] = None,
        task_types: Optional[set] = None,
        feature_types: Optional[set] = None
    ):
        self.id = id
        self.properties = properties or set()
        self.packages = packages or set()
        self.task_types = task_types or {'any'}
        self.feature_types = feature_types or {'any'}
        
    def check_task_compatibility(self, task: Task) -> None:
        """Check if task is compatible with filter."""
        # Check task type
        if 'any' not in self.task_types:
            if isinstance(task, TaskClassif) and 'classif' not in self.task_types:
                raise ValueError(f"Filter {self.id} does not support classification tasks")
            elif isinstance(task, TaskRegr) and 'regr' not in self.task_types:
                raise ValueError(f"Filter {self.id} does not support regression tasks")
                
        # Check for required properties
        if 'weights' in self.properties and not hasattr(task, 'weights'):
            raise ValueError(f"Filter {self.id} requires task weights")
            
    @abstractmethod
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        """Calculate feature scores.
        
        Parameters
        ----------
        task : Task
            The task to filter features from.
        features : List[str], optional
            Subset of features to score. If None, use all.
            
        Returns
        -------
        FilterResult
            Filter calculation results.
        """
        pass
    
    def filter(
        self, 
        task: Task, 
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        percentile: Optional[float] = None,
        features: Optional[List[str]] = None
    ) -> List[str]:
        """Filter features using various strategies.
        
        Parameters
        ----------
        task : Task
            The task to filter.
        k : int, optional
            Select top k features.
        threshold : float, optional
            Select by score threshold.
        percentile : float, optional
            Select top percentile.
        features : List[str], optional
            Subset of features to consider.
            
        Returns
        -------
        List[str]
            Selected feature names.
        """
        result = self.calculate(task, features)
        
        if k is not None:
            return result.select_top_k(k)
        elif threshold is not None:
            return result.select_threshold(threshold)
        elif percentile is not None:
            return result.select_percentile(percentile)
        else:
            raise ValueError("Must specify k, threshold, or percentile")


class FilterRegistry:
    """Registry for filters."""
    
    def __init__(self):
        self._filters = {}
        
    def register(self, filter_class: type, id: str = None) -> None:
        """Register a filter class."""
        id = id or filter_class.__name__.lower()
        self._filters[id] = filter_class
        
    def get(self, id: str) -> type:
        """Get filter class by id."""
        if id not in self._filters:
            raise KeyError(f"Unknown filter: {id}")
        return self._filters[id]
        
    def create(self, id: str, **kwargs) -> Filter:
        """Create filter instance."""
        filter_class = self.get(id)
        return filter_class(**kwargs)
        
    def list(self) -> List[str]:
        """List available filters."""
        return list(self._filters.keys())


# Global registry
filter_registry = FilterRegistry()