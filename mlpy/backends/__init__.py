"""Data backend implementations for MLPY."""

from .base import DataBackend, DataBackendCbind, DataBackendRbind
from .pandas_backend import DataBackendPandas
from .numpy_backend import DataBackendNumPy

# Optional backends for large datasets
try:
    from .dask_backend import DataBackendDask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    DataBackendDask = None

try:
    from .vaex_backend import DataBackendVaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False
    DataBackendVaex = None

__all__ = [
    "DataBackend",
    "DataBackendCbind", 
    "DataBackendRbind",
    "DataBackendPandas",
    "DataBackendNumPy",
]

# Add optional backends if available
if DASK_AVAILABLE:
    __all__.append("DataBackendDask")
if VAEX_AVAILABLE:
    __all__.append("DataBackendVaex")