"""
Lazy evaluation module for MLPY.

Provides computation graphs and deferred execution for optimization.
"""

from .lazy_evaluation import (
    ComputationNode,
    ComputationGraph,
    LazyArray,
    LazyDataFrame,
    lazy_operation,
    create_pipeline,
    optimize_pipeline
)

__all__ = [
    'ComputationNode',
    'ComputationGraph',
    'LazyArray',
    'LazyDataFrame',
    'lazy_operation',
    'create_pipeline',
    'optimize_pipeline'
]