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
    LearnerAdaBoost,
    LearnerExtraTrees,
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
    LearnerAdaBoostRegressor,
    LearnerExtraTreesRegressor,
    LearnerSVR,
    LearnerKNNRegressor,
    LearnerMLPRegressor
)
from .auto_wrap import auto_sklearn

# Aliases for compatibility with registry naming
LearnerRandomForestClassifier = LearnerRandomForest
LearnerGradientBoostingClassifier = LearnerGradientBoosting
LearnerAdaBoostClassifier = LearnerAdaBoost
LearnerExtraTreesClassifier = LearnerExtraTrees
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
    "LearnerAdaBoost",
    "LearnerExtraTrees",
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
    "LearnerAdaBoostRegressor",
    "LearnerExtraTreesRegressor",
    "LearnerSVR",
    "LearnerKNNRegressor",
    "LearnerMLPRegressor",
    
    # Auto-wrapper
    "auto_sklearn",
    
    # Aliases for compatibility
    "LearnerRandomForestClassifier",
    "LearnerGradientBoostingClassifier",
    "LearnerAdaBoostClassifier",
    "LearnerExtraTreesClassifier",
    "LearnerSVMClassifier",
    "LearnerKNNClassifier",
    "LearnerDecisionTreeClassifier"
]