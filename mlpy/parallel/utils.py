"""Utilities for parallel execution.

This module provides convenient functions for parallel execution
and backend management.
"""

from typing import Callable, List, Any, Optional, Union
from contextlib import contextmanager

from .base import Backend, _parallel_config


def get_backend() -> Backend:
    """Get the current default backend.
    
    Returns
    -------
    Backend
        The current default backend.
    """
    return _parallel_config.get_backend()


def set_backend(backend: Union[Backend, str]) -> None:
    """Set the default backend.
    
    Parameters
    ----------
    backend : Backend or str
        Backend instance or string identifier.
        Valid strings: 'sequential', 'threading', 'multiprocessing', 'joblib'
    """
    if isinstance(backend, str):
        backend = _create_backend(backend)
    _parallel_config.set_backend(backend)


@contextmanager
def backend_context(backend: Union[Backend, str]):
    """Context manager for temporary backend.
    
    Parameters
    ----------
    backend : Backend or str
        Backend to use within context.
        
    Examples
    --------
    >>> with backend_context('threading'):
    ...     results = parallel_map(func, data)
    """
    if isinstance(backend, str):
        backend = _create_backend(backend)
        
    with _parallel_config.backend_context(backend):
        yield backend


def parallel_map(
    func: Callable,
    iterable: List[Any],
    backend: Optional[Union[Backend, str]] = None,
    **kwargs
) -> List[Any]:
    """Apply function to elements in parallel.
    
    Parameters
    ----------
    func : callable
        Function to apply to each element.
    iterable : list
        List of elements to process.
    backend : Backend or str, optional
        Backend to use. If None, uses default backend.
    **kwargs
        Additional arguments passed to backend.map()
        
    Returns
    -------
    list
        Results in same order as input.
        
    Examples
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> results = parallel_map(square, [1, 2, 3, 4])
    >>> print(results)
    [1, 4, 9, 16]
    """
    if backend is None:
        backend = get_backend()
    elif isinstance(backend, str):
        backend = _create_backend(backend)
        
    return backend.map(func, iterable, **kwargs)


def parallel_starmap(
    func: Callable,
    iterable: List[tuple],
    backend: Optional[Union[Backend, str]] = None,
    **kwargs
) -> List[Any]:
    """Apply function with multiple arguments in parallel.
    
    Parameters
    ----------
    func : callable
        Function to apply.
    iterable : list of tuples
        List of argument tuples.
    backend : Backend or str, optional
        Backend to use. If None, uses default backend.
    **kwargs
        Additional arguments passed to backend.starmap()
        
    Returns
    -------
    list
        Results in same order as input.
        
    Examples
    --------
    >>> def add(a, b):
    ...     return a + b
    >>> results = parallel_starmap(add, [(1, 2), (3, 4), (5, 6)])
    >>> print(results)
    [3, 7, 11]
    """
    if backend is None:
        backend = get_backend()
    elif isinstance(backend, str):
        backend = _create_backend(backend)
        
    return backend.starmap(func, iterable, **kwargs)


def _create_backend(name: str, **kwargs) -> Backend:
    """Create backend from string identifier.
    
    Parameters
    ----------
    name : str
        Backend name.
    **kwargs
        Arguments passed to backend constructor.
        
    Returns
    -------
    Backend
        Backend instance.
    """
    from .base import BackendSequential
    from .threading import BackendThreading
    from .multiprocessing import BackendMultiprocessing
    
    backends = {
        'sequential': BackendSequential,
        'threading': BackendThreading,
        'multiprocessing': BackendMultiprocessing,
    }
    
    # Try joblib if available
    try:
        from .joblib import BackendJoblib
        backends['joblib'] = BackendJoblib
    except ImportError:
        pass
        
    if name not in backends:
        raise ValueError(
            f"Unknown backend '{name}'. "
            f"Available: {list(backends.keys())}"
        )
        
    return backends[name](**kwargs)


def get_n_jobs(n_jobs: Optional[int] = None) -> int:
    """Get number of jobs from parameter or environment.
    
    Parameters
    ----------
    n_jobs : int, optional
        Number of jobs. If None, checks MLPY_N_JOBS environment variable.
        
    Returns
    -------
    int
        Number of jobs to use.
    """
    if n_jobs is not None:
        return n_jobs
        
    import os
    env_n_jobs = os.environ.get('MLPY_N_JOBS')
    
    if env_n_jobs is not None:
        try:
            return int(env_n_jobs)
        except ValueError:
            pass
            
    return 1  # Default to sequential


def chunk_iterable(iterable: List[Any], n_chunks: int) -> List[List[Any]]:
    """Split iterable into approximately equal chunks.
    
    Parameters
    ----------
    iterable : list
        List to split.
    n_chunks : int
        Number of chunks.
        
    Returns
    -------
    list of lists
        List split into chunks.
    """
    n = len(iterable)
    chunk_size = n // n_chunks
    remainder = n % n_chunks
    
    chunks = []
    start = 0
    
    for i in range(n_chunks):
        # Add 1 to chunk size for first 'remainder' chunks
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(iterable[start:start + size])
        start += size
        
    return chunks


__all__ = [
    "get_backend",
    "set_backend",
    "backend_context",
    "parallel_map",
    "parallel_starmap",
    "get_n_jobs",
    "chunk_iterable"
]