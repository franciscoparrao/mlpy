"""Model interpretability module for MLPY.

This module provides tools for explaining and interpreting machine learning models,
including integration with popular interpretability libraries like SHAP and LIME.
"""

from .base import Interpreter, InterpretationResult, FeatureImportance
from .shap_interpreter import SHAPInterpreter, SHAPExplanation
from .lime_interpreter import LIMEInterpreter, LIMEExplanation
from .utils import (
    plot_feature_importance, 
    plot_shap_summary, 
    plot_lime_explanation,
    plot_interpretation_comparison,
    create_interpretation_report
)

__all__ = [
    # Base classes
    "Interpreter",
    "InterpretationResult", 
    "FeatureImportance",
    
    # SHAP
    "SHAPInterpreter",
    "SHAPExplanation",
    
    # LIME
    "LIMEInterpreter",
    "LIMEExplanation",
    
    # Utilities
    "plot_feature_importance",
    "plot_shap_summary",
    "plot_lime_explanation",
    "plot_interpretation_comparison",
    "create_interpretation_report"
]