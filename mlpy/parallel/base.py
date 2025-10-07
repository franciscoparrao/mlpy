"""Base classes for parallel execution backends.

This module provides the abstract base class for parallel backends
and a sequential backend for baseline execution.
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Any, Optional, Dict
import warnings
import time
from contextlib import contextmanager

from ..base import MLPYObject


class Backend(MLPYObject, ABC):
    """Abstract base class for parallel execution backends.
    
    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs. -1 means use all processors.
    id : str, optional
        Backend identifier.
    """
    
    def __init__(self, n_jobs: int = 1, id: Optional[str] = None):
        super().__init__(id=id or self.__class__.__name__.lower())
        self.n_jobs = self._normalize_n_jobs(n_jobs)
        
    def _normalize_n_jobs(self, n_jobs: int) -> int:
        """Normalize n_jobs parameter."""
        import os
        
        if n_jobs == -1:
            # Use all available cores
            try:
                import multiprocessing
                return multiprocessing.cpu_count()
            except:
                return os.cpu_count() or 1
        elif n_jobs <= 0:
            raise ValueError(f"n_jobs must be positive or -1, got {n_jobs}")
        else:
            return n_jobs
            
    @abstractmethod
    def map(
        self,
        func: Callable,
        iterable: List[Any],
        **kwargs
    ) -> List[Any]:
        """Apply function to each element in parallel.
        
        Parameters
        ----------
        func : callable
            Function to apply to each element.
        iterable : list
            List of elements to process.
        **kwargs
            Additional arguments passed to the backend.
            
        Returns
        -------
        list
            Results of applying func to each element.
        """
        pass
        
    @abstractmethod
    def starmap(
        self,
        func: Callable,
        iterable: List[tuple],
        **kwargs
    ) -> List[Any]:
        """Apply function with multiple arguments in parallel.
        
        Parameters
        ----------
        func : callable
            Function to apply.
        iterable : list of tuples
            List of argument tuples for func.
        **kwargs
            Additional arguments passed to the backend.
            
        Returns
        -------
        list
            Results of applying func to each argument tuple.
        """
        pass
        
    @contextmanager
    def parallel_context(self, **kwargs):
        """Context manager for parallel execution.
        
        This can be used to set up and tear down resources
        needed for parallel execution.
        """
        yield self
        
    def close(self):
        """Clean up backend resources."""
        pass
        
    def __enter__(self):
        """Enter context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
        return False


class BackendSequential(Backend):
    """Sequential backend (no parallelization).
    
    This backend executes tasks sequentially and serves as
    a baseline for comparison and debugging.
    """
    
    def __init__(self):
        super().__init__(n_jobs=1, id="sequential")
        
    def map(
        self,
        func: Callable,
        iterable: List[Any],
        verbose: int = 0,
        **kwargs
    ) -> List[Any]:
        """Apply function sequentially."""
        results = []
        n_tasks = len(iterable)
        
        for i, item in enumerate(iterable):
            if verbose > 0:
                print(f"Processing task {i+1}/{n_tasks}")
                
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Task {i} failed: {e}")
                results.append(None)
                
        return results
        
    def starmap(
        self,
        func: Callable,
        iterable: List[tuple],
        verbose: int = 0,
        **kwargs
    ) -> List[Any]:
        """Apply function with multiple arguments sequentially."""
        results = []
        n_tasks = len(iterable)
        
        for i, args in enumerate(iterable):
            if verbose > 0:
                print(f"Processing task {i+1}/{n_tasks}")
                
            try:
                result = func(*args)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Task {i} failed: {e}")
                results.append(None)
                
        return results


class ParallelConfig:
    """Global configuration for parallel execution.
    
    This class manages the default backend and provides
    utilities for parallel execution configuration.
    """
    
    def __init__(self):
        self._backend = BackendSequential()
        self._backend_stack = []
        
    def get_backend(self) -> Backend:
        """Get current backend."""
        return self._backend
        
    def set_backend(self, backend: Backend) -> None:
        """Set default backend."""
        if not isinstance(backend, Backend):
            raise TypeError(f"Expected Backend, got {type(backend)}")
        self._backend = backend
        
    @contextmanager
    def backend_context(self, backend: Backend):
        """Temporarily use a different backend."""
        self._backend_stack.append(self._backend)
        self._backend = backend
        try:
            yield backend
        finally:
            self._backend = self._backend_stack.pop()


# Global configuration instance
_parallel_config = ParallelConfig()


__all__ = [
    "Backend",
    "BackendSequential",
    "ParallelConfig",
    "_parallel_config"
]