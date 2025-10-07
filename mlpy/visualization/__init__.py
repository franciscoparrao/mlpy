"""
Visualization module for MLPY.

Provides plotting and visualization utilities for machine learning workflows.
"""

from .plots import (
    plot_learning_curve,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_residuals,
    plot_prediction_error,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
    plot_validation_curve
)

from .dashboards import (
    create_model_dashboard,
    create_comparison_dashboard
)

__all__ = [
    # Basic plots
    "plot_learning_curve",
    "plot_feature_importance", 
    "plot_confusion_matrix",
    "plot_residuals",
    "plot_prediction_error",
    "plot_roc_curve",
    "plot_precision_recall_curve", 
    "plot_calibration_curve",
    "plot_validation_curve",
    # Dashboards
    "create_model_dashboard",
    "create_comparison_dashboard"
]