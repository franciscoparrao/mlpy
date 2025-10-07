"""Scikit-learn learner wrappers for MLPY.

This module provides wrappers for scikit-learn models to integrate them
seamlessly into the MLPY framework.
"""

from .base import LearnerSKLearn
from .classification import (
    LearnerLogisticRegression,
    LearnerDecisionTree,
    LearnerRandomForest,
    LearnerGradientBoosting,
    LearnerSVM,
    LearnerKNN,
    LearnerNaiveBayes,
    LearnerMLPClassifier
)
from .regression import (
    LearnerLinearRegression,
    LearnerRidge,
    LearnerLasso,
    LearnerElasticNet,
    LearnerDecisionTreeRegressor,
    LearnerRandomForestRegressor,
    LearnerGradientBoostingRegressor,
    LearnerSVR,
    LearnerKNNRegressor,
    LearnerMLPRegressor
)
from .auto_wrap import auto_sklearn

# Aliases for compatibility with registry naming
LearnerRandomForestClassifier = LearnerRandomForest
LearnerGradientBoostingClassifier = LearnerGradientBoosting
LearnerSVMClassifier = LearnerSVM
LearnerKNNClassifier = LearnerKNN
LearnerDecisionTreeClassifier = LearnerDecisionTree

__all__ = [
    # Base
    "LearnerSKLearn",
    
    # Classification
    "LearnerLogisticRegression",
    "LearnerDecisionTree",
    "LearnerRandomForest",
    "LearnerGradientBoosting",
    "LearnerSVM",
    "LearnerKNN",
    "LearnerNaiveBayes",
    "LearnerMLPClassifier",
    
    # Regression
    "LearnerLinearRegression",
    "LearnerRidge",
    "LearnerLasso",
    "LearnerElasticNet",
    "LearnerDecisionTreeRegressor",
    "LearnerRandomForestRegressor",
    "LearnerGradientBoostingRegressor",
    "LearnerSVR",
    "LearnerKNNRegressor",
    "LearnerMLPRegressor",
    
    # Auto-wrapper
    "auto_sklearn",
    
    # Aliases for compatibility
    "LearnerRandomForestClassifier",
    "LearnerGradientBoostingClassifier",
    "LearnerSVMClassifier",
    "LearnerKNNClassifier",
    "LearnerDecisionTreeClassifier"
]