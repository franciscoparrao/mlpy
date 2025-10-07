"""
Registry system for MLPY components.

This module provides a generic registry system for registering and retrieving
MLPY components like tasks, learners, measures, and resamplings.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union, Callable
from collections.abc import MutableMapping
import warnings

T = TypeVar("T")


class Registry(MutableMapping):
    """
    A registry for storing and retrieving objects by key.
    
    This class provides a dictionary-like interface with additional features
    for managing ML components.
    
    Parameters
    ----------
    name : str
        Name of the registry (e.g., "learners", "tasks")
    """
    
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Any] = {}
        self._aliases: Dict[str, str] = {}
    
    def __getitem__(self, key: str) -> Any:
        """Get an item from the registry."""
        # Check if it's an alias
        if key in self._aliases:
            key = self._aliases[key]
        
        if key not in self._items:
            raise KeyError(f"'{key}' not found in {self.name} registry")
        
        return self._items[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Add an item to the registry."""
        if key in self._items:
            warnings.warn(
                f"Overwriting existing item '{key}' in {self.name} registry",
                UserWarning,
                stacklevel=2
            )
        self._items[key] = value
    
    def __delitem__(self, key: str) -> None:
        """Remove an item from the registry."""
        # Remove any aliases pointing to this key
        aliases_to_remove = [alias for alias, target in self._aliases.items() if target == key]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        del self._items[key]
    
    def __iter__(self):
        """Iterate over registry keys."""
        return iter(self._items)
    
    def __len__(self) -> int:
        """Return the number of items in the registry."""
        return len(self._items)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the registry or aliases."""
        return key in self._items or key in self._aliases
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        n_items = len(self._items)
        n_aliases = len(self._aliases)
        items_str = f"{n_items} item{'s' if n_items != 1 else ''}"
        aliases_str = f"{n_aliases} alias{'es' if n_aliases != 1 else ''}"
        return f"<Registry '{self.name}' ({items_str}, {aliases_str})>"
    
    def register(
        self, 
        key: str, 
        value: Any = None, 
        *, 
        aliases: Optional[list[str]] = None,
        force: bool = False
    ) -> Union[None, Callable]:
        """
        Register an item in the registry.
        
        Can be used as a decorator or called directly.
        
        Parameters
        ----------
        key : str
            The key to register the item under
        value : Any, optional
            The value to register. If None, returns a decorator
        aliases : list[str], optional
            Alternative names for the item
        force : bool, default=False
            If True, overwrite existing items without warning
        
        Returns
        -------
        None or Callable
            None if value is provided, otherwise a decorator
        
        Examples
        --------
        >>> registry = Registry("models")
        >>> 
        >>> # Direct registration
        >>> registry.register("my_model", MyModel)
        >>> 
        >>> # As a decorator
        >>> @registry.register("my_model")
        ... class MyModel:
        ...     pass
        """
        def decorator(obj: T) -> T:
            if not force and key in self._items:
                raise ValueError(f"Key '{key}' already exists in {self.name} registry")
            
            self._items[key] = obj
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    if not force and alias in self._aliases:
                        raise ValueError(f"Alias '{alias}' already exists in {self.name} registry")
                    self._aliases[alias] = key
            
            return obj
        
        if value is None:
            return decorator
        else:
            return decorator(value)
    
    def add_alias(self, alias: str, key: str) -> None:
        """
        Add an alias for an existing key.
        
        Parameters
        ----------
        alias : str
            The alias to add
        key : str
            The existing key to alias
        """
        if key not in self._items:
            raise KeyError(f"Key '{key}' not found in {self.name} registry")
        
        if alias in self._aliases:
            warnings.warn(
                f"Overwriting existing alias '{alias}' in {self.name} registry",
                UserWarning,
                stacklevel=2
            )
        
        self._aliases[alias] = key
    
    def get_keys(self, include_aliases: bool = False) -> list[str]:
        """
        Get all keys in the registry.
        
        Parameters
        ----------
        include_aliases : bool, default=False
            If True, include aliases in the returned list
        
        Returns
        -------
        list[str]
            List of keys (and optionally aliases)
        """
        keys = list(self._items.keys())
        if include_aliases:
            keys.extend(self._aliases.keys())
        return sorted(keys)
    
    def clear(self) -> None:
        """Clear all items and aliases from the registry."""
        self._items.clear()
        self._aliases.clear()


# Create global registries
mlpy_tasks = Registry("tasks")
mlpy_learners = Registry("learners")
mlpy_measures = Registry("measures")
mlpy_resamplings = Registry("resamplings")
mlpy_filters = Registry("filters")


__all__ = [
    "Registry",
    "mlpy_tasks",
    "mlpy_learners", 
    "mlpy_measures",
    "mlpy_resamplings",
    "mlpy_filters",
]