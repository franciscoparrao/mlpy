"""Predictions module for MLPY.

This module provides classes for storing and manipulating predictions
from machine learning models.
"""

from .base import Prediction
from .classification import PredictionClassif
from .regression import PredictionRegr

__all__ = ["Prediction", "PredictionClassif", "PredictionRegr"]