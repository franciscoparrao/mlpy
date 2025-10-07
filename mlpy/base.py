"""Base classes for MLPY framework.

This module provides the foundational classes that all MLPY
components inherit from.
"""

from abc import ABC
from typing import Optional, Dict, Any, Set
import copy


class MLPYObject(ABC):
    """Base class for all MLPY objects.
    
    Provides common functionality like ID management, cloning,
    and string representation.
    
    Parameters
    ----------
    id : str
        Unique identifier for the object.
    label : str, optional
        Human-readable label for the object.
    """
    
    def __init__(self, id: str, label: Optional[str] = None, **kwargs):
        self.id = id
        self.label = label if label is not None else id
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def clone(self, deep: bool = True) -> "MLPYObject":
        """Create a copy of the object.
        
        Parameters
        ----------
        deep : bool, default=True
            Whether to make a deep copy.
            
        Returns
        -------
        MLPYObject
            A copy of the object.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
            
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"<{self.__class__.__name__}:{self.id}>"
        
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.label
        
    def __eq__(self, other):
        """Check equality based on ID."""
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id
        
    def __hash__(self):
        """Hash based on ID."""
        return hash(self.id)