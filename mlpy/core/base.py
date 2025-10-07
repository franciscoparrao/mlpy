"""
Base classes for MLPY framework.

This module provides the foundational classes that all MLPY components inherit from.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional, Set, Type, TypeVar, Union
from uuid import uuid4

T = TypeVar("T", bound="MLPYObject")


class MLPYObject(ABC):
    """
    Base class for all MLPY objects.
    
    This class provides common functionality like hashing, cloning, parameter management,
    and string representations.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier for the object. If None, a random ID is generated.
    label : str, optional
        Human-readable label for the object
    """
    
    def __init__(self, id: Optional[str] = None, label: Optional[str] = None):
        self._id = id or self._generate_id()
        self._label = label or self._id
        self._hash: Optional[str] = None
        self._params: Dict[str, Any] = {}
        self._dirty = True  # Flag to track if hash needs recalculation
    
    @property
    def id(self) -> str:
        """Unique identifier for the object."""
        return self._id
    
    @property
    def label(self) -> str:
        """Human-readable label for the object."""
        return self._label
    
    @label.setter
    def label(self, value: str) -> None:
        """Set the label for the object."""
        self._label = value
        self._dirty = True
    
    @property
    def hash(self) -> str:
        """
        Unique hash of the object state.
        
        The hash is calculated based on the object's parameters and state.
        It is cached and only recalculated when the object changes.
        """
        if self._dirty or self._hash is None:
            self._hash = self._calculate_hash()
            self._dirty = False
        return self._hash
    
    @property
    @abstractmethod
    def _properties(self) -> Set[str]:
        """
        Set of properties that characterize this object type.
        
        Returns
        -------
        Set[str]
            Set of property names
        """
        pass
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the object."""
        class_name = self.__class__.__name__.lower()
        unique_suffix = str(uuid4())[:8]
        return f"{class_name}_{unique_suffix}"
    
    def _calculate_hash(self) -> str:
        """
        Calculate a hash based on the object's state.
        
        Returns
        -------
        str
            Hexadecimal hash string
        """
        # Collect all relevant state information
        state_dict = {
            "class": self.__class__.__name__,
            "id": self._id,
            "params": self._get_params_for_hash(),
        }
        
        # Convert to JSON string for consistent hashing
        state_json = json.dumps(state_dict, sort_keys=True)
        
        # Calculate SHA256 hash
        return hashlib.sha256(state_json.encode()).hexdigest()
    
    def _get_params_for_hash(self) -> Dict[str, Any]:
        """
        Get parameters that should be included in hash calculation.
        
        Override this method in subclasses to customize hash calculation.
        
        Returns
        -------
        Dict[str, Any]
            Parameters to include in hash
        """
        return self._params.copy()
    
    def _mark_dirty(self) -> None:
        """Mark the object as dirty, requiring hash recalculation."""
        self._dirty = True
    
    def clone(self: T, deep: bool = True) -> T:
        """
        Create a copy of the object.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, create a deep copy. If False, create a shallow copy.
        
        Returns
        -------
        MLPYObject
            A copy of the object
        """
        if deep:
            return deepcopy(self)
        else:
            # Create a new instance with same parameters
            cls = self.__class__
            new_obj = cls.__new__(cls)
            new_obj.__dict__.update(self.__dict__.copy())
            new_obj._id = self._generate_id()  # New object gets new ID
            new_obj._dirty = True
            return new_obj
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value.
        
        Parameters
        ----------
        name : str
            Parameter name
        default : Any, optional
            Default value if parameter not found
        
        Returns
        -------
        Any
            Parameter value
        """
        return self._params.get(name, default)
    
    def set_params(self, **params: Any) -> None:
        """
        Set multiple parameters at once.
        
        Parameters
        ----------
        **params : Any
            Parameter names and values
        """
        self._params.update(params)
        self._mark_dirty()
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get all parameters.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of all parameters
        """
        return self._params.copy()
    
    def has_property(self, property: str) -> bool:
        """
        Check if this object has a specific property.
        
        Parameters
        ----------
        property : str
            Property name to check
        
        Returns
        -------
        bool
            True if object has the property
        """
        return property in self._properties
    
    def __repr__(self) -> str:
        """String representation of the object."""
        class_name = self.__class__.__name__
        return f"<{class_name}:{self._id}>"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.__class__.__name__}({self._label})"
    
    def __eq__(self, other: Any) -> bool:
        """Check equality based on hash."""
        if not isinstance(other, MLPYObject):
            return False
        return self.hash == other.hash
    
    def __hash__(self) -> int:
        """Make object hashable."""
        return hash(self.hash)


class ParamSet:
    """
    Parameter set for validating and managing parameters.
    
    This class defines the valid parameters for an MLPY object,
    including their types, defaults, and constraints.
    
    Parameters
    ----------
    params : Dict[str, ParamDef]
        Dictionary mapping parameter names to their definitions
    """
    
    def __init__(self, params: Optional[Dict[str, "ParamDef"]] = None):
        self.params = params or {}
    
    def validate(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameter values.
        
        Parameters
        ----------
        values : Dict[str, Any]
            Parameter values to validate
        
        Returns
        -------
        Dict[str, Any]
            Validated parameter values with defaults applied
        
        Raises
        ------
        ValueError
            If validation fails
        """
        validated = {}
        
        # Check all provided values
        for name, value in values.items():
            if name not in self.params:
                raise ValueError(f"Unknown parameter: {name}")
            
            param_def = self.params[name]
            validated[name] = param_def.validate(value)
        
        # Apply defaults for missing parameters
        for name, param_def in self.params.items():
            if name not in validated and param_def.default is not None:
                validated[name] = param_def.default
        
        return validated
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return {
            name: param.default 
            for name, param in self.params.items() 
            if param.default is not None
        }


class ParamDef:
    """
    Definition of a single parameter.
    
    Parameters
    ----------
    type : Type or tuple of Types
        Valid type(s) for the parameter
    default : Any, optional
        Default value
    description : str, optional
        Description of the parameter
    lower : float, optional
        Lower bound for numeric parameters
    upper : float, optional
        Upper bound for numeric parameters
    values : list, optional
        Valid values for categorical parameters
    """
    
    def __init__(
        self,
        type: Union[Type, tuple[Type, ...]],
        default: Any = None,
        description: Optional[str] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        values: Optional[list] = None,
    ):
        self.type = type
        self.default = default
        self.description = description
        self.lower = lower
        self.upper = upper
        self.values = values
    
    def validate(self, value: Any) -> Any:
        """
        Validate a value against this parameter definition.
        
        Parameters
        ----------
        value : Any
            Value to validate
        
        Returns
        -------
        Any
            The validated value
        
        Raises
        ------
        ValueError
            If validation fails
        """
        # Type check
        if not isinstance(value, self.type):
            type_str = self.type.__name__ if hasattr(self.type, '__name__') else str(self.type)
            raise ValueError(f"Expected type {type_str}, got {type(value).__name__}")
        
        # Numeric bounds check
        if self.lower is not None and value < self.lower:
            raise ValueError(f"Value {value} is below lower bound {self.lower}")
        
        if self.upper is not None and value > self.upper:
            raise ValueError(f"Value {value} is above upper bound {self.upper}")
        
        # Categorical values check
        if self.values is not None and value not in self.values:
            raise ValueError(f"Value {value} not in allowed values: {self.values}")
        
        return value


__all__ = ["MLPYObject", "ParamSet", "ParamDef"]