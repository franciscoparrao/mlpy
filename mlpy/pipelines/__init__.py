"""Pipeline operations for MLPY.

This module provides composable operations for building
machine learning pipelines.
"""

from .base import (
    PipeOp,
    PipeOpInput,
    PipeOpOutput,
    PipeOpState,
    PipeOpLearner,
    PipeOpNOP,
    mlpy_pipeops
)

from .operators import (
    PipeOpScale,
    PipeOpImpute,
    PipeOpSelect,
    PipeOpEncode
)

from .graph import (
    Graph,
    GraphLearner,
    Edge,
    linear_pipeline
)

# Advanced operators
from .advanced_operators import (
    PipeOpPCA,
    PipeOpTargetEncode,
    PipeOpOutlierDetect,
    PipeOpBin,
    PipeOpTextVectorize,
    PipeOpPolynomial
)

# Filter operators
from .filter_ops import (
    PipeOpFilter,
    PipeOpFilterRank,
    PipeOpFilterCorr
)

# Optional lazy operations
try:
    from .lazy_ops import (
        LazyPipeOp,
        LazyPipeOpScale,
        LazyPipeOpFilter,
        LazyPipeOpSample,
        LazyPipeOpCache
    )
    LAZY_OPS_AVAILABLE = True
except ImportError:
    LAZY_OPS_AVAILABLE = False


__all__ = [
    # Base classes
    "PipeOp",
    "PipeOpInput",
    "PipeOpOutput", 
    "PipeOpState",
    "PipeOpLearner",
    "PipeOpNOP",
    "mlpy_pipeops",
    
    # Basic operators
    "PipeOpScale",
    "PipeOpImpute",
    "PipeOpSelect",
    "PipeOpEncode",
    
    # Advanced operators
    "PipeOpPCA",
    "PipeOpTargetEncode",
    "PipeOpOutlierDetect",
    "PipeOpBin",
    "PipeOpTextVectorize",
    "PipeOpPolynomial",
    
    # Filter operators
    "PipeOpFilter",
    "PipeOpFilterRank",
    "PipeOpFilterCorr",
    
    # Graph
    "Graph",
    "GraphLearner",
    "Edge",
    "linear_pipeline",
]

# Add lazy operations if available
if LAZY_OPS_AVAILABLE:
    __all__.extend([
        "LazyPipeOp",
        "LazyPipeOpScale",
        "LazyPipeOpFilter",
        "LazyPipeOpSample",
        "LazyPipeOpCache"
    ])