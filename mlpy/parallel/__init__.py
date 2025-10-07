"""Parallel execution backends for MLPY.

This module provides different backends for parallel execution
of tasks, including threading, multiprocessing, and joblib.
"""

from .base import Backend, BackendSequential
from .threading import BackendThreading
from .multiprocessing import BackendMultiprocessing
from .joblib import BackendJoblib
from .utils import get_backend, set_backend, backend_context, parallel_map, parallel_starmap

__all__ = [
    "Backend",
    "BackendSequential", 
    "BackendThreading",
    "BackendMultiprocessing",
    "BackendJoblib",
    "get_backend",
    "set_backend",
    "backend_context",
    "parallel_map",
    "parallel_starmap"
]