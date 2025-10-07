"""Joblib backend for parallel execution.

This backend uses joblib for parallel execution, providing
a more sophisticated interface with better error handling
and memory management.
"""

from typing import List, Callable, Any, Optional
import warnings

from .base import Backend


class BackendJoblib(Backend):
    """Joblib backend for parallel execution.
    
    This backend provides advanced features like memory mapping
    and better handling of numpy arrays.
    
    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs. -1 means use all processors.
    backend : str
        Joblib backend to use ('loky', 'threading', 'multiprocessing').
    prefer : str
        Soft hint for backend ('processes' or 'threads').
    verbose : int
        Verbosity level for joblib.
    """
    
    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = "loky",
        prefer: str = "processes",
        verbose: int = 0
    ):
        super().__init__(n_jobs=n_jobs, id="joblib")
        self.backend = backend
        self.prefer = prefer
        self.verbose = verbose
        self._parallel = None
        
        # Check if joblib is available
        try:
            import joblib
            self._joblib = joblib
        except ImportError:
            raise ImportError(
                "joblib is required for BackendJoblib. "
                "Install it with: pip install joblib"
            )
            
    def _get_parallel(self):
        """Get or create Parallel instance."""
        if self._parallel is None:
            self._parallel = self._joblib.Parallel(
                n_jobs=self.n_jobs,
                backend=self.backend,
                prefer=self.prefer,
                verbose=self.verbose
            )
        return self._parallel
        
    def map(
        self,
        func: Callable,
        iterable: List[Any],
        verbose: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """Apply function to each element using joblib.
        
        Parameters
        ----------
        func : callable
            Function to apply to each element.
        iterable : list
            List of elements to process.
        verbose : int, optional
            Override verbosity level.
            
        Returns
        -------
        list
            Results in the same order as input.
        """
        parallel = self._get_parallel()
        
        # Override verbosity if specified
        if verbose is not None:
            parallel = self._joblib.Parallel(
                n_jobs=self.n_jobs,
                backend=self.backend,
                prefer=self.prefer,
                verbose=verbose
            )
            
        try:
            # Use joblib's delayed for lazy evaluation
            delayed_func = self._joblib.delayed(func)
            results = parallel(delayed_func(item) for item in iterable)
            return results
            
        except Exception as e:
            warnings.warn(f"Parallel execution failed: {e}")
            # Fall back to sequential
            results = []
            for i, item in enumerate(iterable):
                try:
                    result = func(item)
                    results.append(result)
                except Exception as task_e:
                    warnings.warn(f"Task {i} failed: {task_e}")
                    results.append(None)
            return results
            
    def starmap(
        self,
        func: Callable,
        iterable: List[tuple],
        verbose: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """Apply function with multiple arguments using joblib.
        
        Parameters
        ----------
        func : callable
            Function to apply.
        iterable : list of tuples
            List of argument tuples.
        verbose : int, optional
            Override verbosity level.
            
        Returns
        -------
        list
            Results in the same order as input.
        """
        def wrapper(args):
            return func(*args)
            
        return self.map(wrapper, iterable, verbose=verbose, **kwargs)
        
    def close(self):
        """Clean up joblib resources."""
        # Joblib handles cleanup automatically
        self._parallel = None
        
    def __repr__(self) -> str:
        return (
            f"BackendJoblib(n_jobs={self.n_jobs}, "
            f"backend='{self.backend}', prefer='{self.prefer}')"
        )


# Advanced joblib features for memory management
class BackendJoblibMemory(BackendJoblib):
    """Joblib backend with memory caching.
    
    This backend caches function results to disk to avoid
    recomputation.
    
    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs.
    cachedir : str
        Directory for cache storage.
    compress : bool or int
        Whether to compress cache data.
    **kwargs
        Additional arguments for BackendJoblib.
    """
    
    def __init__(
        self,
        n_jobs: int = -1,
        cachedir: str = ".mlpy_cache",
        compress: bool = True,
        **kwargs
    ):
        super().__init__(n_jobs=n_jobs, **kwargs)
        self.cachedir = cachedir
        self.compress = compress
        
        # Create memory object
        self.memory = self._joblib.Memory(
            location=cachedir,
            compress=compress,
            verbose=self.verbose
        )
        
    def cache_function(self, func: Callable) -> Callable:
        """Decorate function with caching.
        
        Parameters
        ----------
        func : callable
            Function to cache.
            
        Returns
        -------
        callable
            Cached version of function.
        """
        return self.memory.cache(func)
        
    def clear_cache(self):
        """Clear all cached results."""
        self.memory.clear()
        
    def reduce_cache(self, bytes_limit: int):
        """Reduce cache size to specified limit.
        
        Parameters
        ----------
        bytes_limit : int
            Maximum cache size in bytes.
        """
        self.memory.reduce_size(bytes_limit)


__all__ = ["BackendJoblib", "BackendJoblibMemory"]