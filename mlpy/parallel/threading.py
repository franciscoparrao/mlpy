"""Threading backend for parallel execution.

This backend uses Python's threading module for parallel execution.
Best suited for I/O-bound tasks due to the GIL.
"""

from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from .base import Backend


class BackendThreading(Backend):
    """Threading backend using ThreadPoolExecutor.
    
    This backend is best suited for I/O-bound tasks where
    the GIL is released (file I/O, network requests, etc).
    
    Parameters
    ----------
    n_jobs : int
        Number of worker threads. -1 means use all processors.
    thread_name_prefix : str
        Prefix for worker thread names.
    """
    
    def __init__(
        self,
        n_jobs: int = -1,
        thread_name_prefix: str = "MLPY-Worker"
    ):
        super().__init__(n_jobs=n_jobs, id="threading")
        self.thread_name_prefix = thread_name_prefix
        self._executor = None
        
    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.n_jobs,
                thread_name_prefix=self.thread_name_prefix
            )
        return self._executor
        
    def map(
        self,
        func: Callable,
        iterable: List[Any],
        timeout: Optional[float] = None,
        verbose: int = 0,
        **kwargs
    ) -> List[Any]:
        """Apply function to each element using threads.
        
        Parameters
        ----------
        func : callable
            Function to apply to each element.
        iterable : list
            List of elements to process.
        timeout : float, optional
            Timeout in seconds for each task.
        verbose : int
            Verbosity level.
            
        Returns
        -------
        list
            Results in the same order as input.
        """
        executor = self._get_executor()
        n_tasks = len(iterable)
        
        # Submit all tasks
        future_to_idx = {}
        for i, item in enumerate(iterable):
            future = executor.submit(func, item)
            future_to_idx[future] = i
            
        # Collect results
        results = [None] * n_tasks
        completed = 0
        
        for future in as_completed(future_to_idx, timeout=timeout):
            idx = future_to_idx[future]
            completed += 1
            
            if verbose > 0:
                print(f"Completed task {completed}/{n_tasks}")
                
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                warnings.warn(f"Task {idx} failed: {e}")
                results[idx] = None
                
        return results
        
    def starmap(
        self,
        func: Callable,
        iterable: List[tuple],
        timeout: Optional[float] = None,
        verbose: int = 0,
        **kwargs
    ) -> List[Any]:
        """Apply function with multiple arguments using threads.
        
        Parameters
        ----------
        func : callable
            Function to apply.
        iterable : list of tuples
            List of argument tuples.
        timeout : float, optional
            Timeout in seconds for each task.
        verbose : int
            Verbosity level.
            
        Returns
        -------
        list
            Results in the same order as input.
        """
        def wrapper(args):
            return func(*args)
            
        return self.map(wrapper, iterable, timeout=timeout, verbose=verbose, **kwargs)
        
    def close(self):
        """Shutdown thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            
    def __repr__(self) -> str:
        return f"BackendThreading(n_jobs={self.n_jobs})"


__all__ = ["BackendThreading"]