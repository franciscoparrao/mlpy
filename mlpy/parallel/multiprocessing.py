"""Multiprocessing backend for parallel execution.

This backend uses Python's multiprocessing module for true
parallel execution, bypassing the GIL.
"""

from typing import List, Callable, Any, Optional, Dict
from multiprocessing import Pool, cpu_count
import warnings
import signal
from functools import partial

from .base import Backend


def _init_worker():
    """Initialize worker process to ignore SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class BackendMultiprocessing(Backend):
    """Multiprocessing backend using Pool.
    
    This backend creates separate processes for parallel execution,
    allowing true parallelism for CPU-bound tasks.
    
    Parameters
    ----------
    n_jobs : int
        Number of worker processes. -1 means use all processors.
    maxtasksperchild : int, optional
        Number of tasks a worker process completes before it's replaced.
    initializer : callable, optional
        Function to initialize worker processes.
    initargs : tuple, optional
        Arguments for initializer function.
    """
    
    def __init__(
        self,
        n_jobs: int = -1,
        maxtasksperchild: Optional[int] = None,
        initializer: Optional[Callable] = None,
        initargs: tuple = ()
    ):
        super().__init__(n_jobs=n_jobs, id="multiprocessing")
        self.maxtasksperchild = maxtasksperchild
        self.initializer = initializer or _init_worker
        self.initargs = initargs
        self._pool = None
        
    def _get_pool(self) -> Pool:
        """Get or create process pool."""
        if self._pool is None:
            self._pool = Pool(
                processes=self.n_jobs,
                initializer=self.initializer,
                initargs=self.initargs,
                maxtasksperchild=self.maxtasksperchild
            )
        return self._pool
        
    def map(
        self,
        func: Callable,
        iterable: List[Any],
        chunksize: Optional[int] = None,
        verbose: int = 0,
        **kwargs
    ) -> List[Any]:
        """Apply function to each element using processes.
        
        Parameters
        ----------
        func : callable
            Function to apply to each element.
        iterable : list
            List of elements to process.
        chunksize : int, optional
            Size of chunks sent to worker processes.
        verbose : int
            Verbosity level.
            
        Returns
        -------
        list
            Results in the same order as input.
        """
        pool = self._get_pool()
        n_tasks = len(iterable)
        
        if chunksize is None:
            # Heuristic for chunk size
            chunksize = max(1, n_tasks // (self.n_jobs * 4))
            
        if verbose > 0:
            print(f"Processing {n_tasks} tasks with {self.n_jobs} workers")
            print(f"Using chunksize={chunksize}")
            
        try:
            # Use map for ordered results
            results = pool.map(func, iterable, chunksize=chunksize)
            
            if verbose > 0:
                print(f"Completed all {n_tasks} tasks")
                
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
        chunksize: Optional[int] = None,
        verbose: int = 0,
        **kwargs
    ) -> List[Any]:
        """Apply function with multiple arguments using processes.
        
        Parameters
        ----------
        func : callable
            Function to apply.
        iterable : list of tuples
            List of argument tuples.
        chunksize : int, optional
            Size of chunks sent to worker processes.
        verbose : int
            Verbosity level.
            
        Returns
        -------
        list
            Results in the same order as input.
        """
        pool = self._get_pool()
        n_tasks = len(iterable)
        
        if chunksize is None:
            chunksize = max(1, n_tasks // (self.n_jobs * 4))
            
        if verbose > 0:
            print(f"Processing {n_tasks} tasks with {self.n_jobs} workers")
            
        try:
            results = pool.starmap(func, iterable, chunksize=chunksize)
            
            if verbose > 0:
                print(f"Completed all {n_tasks} tasks")
                
            return results
            
        except Exception as e:
            warnings.warn(f"Parallel execution failed: {e}")
            # Fall back to sequential
            results = []
            for i, args in enumerate(iterable):
                try:
                    result = func(*args)
                    results.append(result)
                except Exception as task_e:
                    warnings.warn(f"Task {i} failed: {task_e}")
                    results.append(None)
            return results
            
    def close(self):
        """Close and cleanup process pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
            
    def terminate(self):
        """Terminate process pool immediately."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            
    def __repr__(self) -> str:
        return f"BackendMultiprocessing(n_jobs={self.n_jobs})"


# Helper function for pickling
def _apply_along_axis(func, axis, arr, *args, **kwargs):
    """Apply function along axis (for numpy compatibility)."""
    import numpy as np
    return np.apply_along_axis(func, axis, arr, *args, **kwargs)


__all__ = ["BackendMultiprocessing"]