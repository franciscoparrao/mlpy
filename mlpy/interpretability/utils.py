"""Utility functions for model interpretability."""

from typing import Optional, Tuple, List, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from .base import FeatureImportance, InterpretationResult
from .shap_interpreter import SHAPExplanation
from .lime_interpreter import LIMEExplanation

# Check for optional dependencies
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def plot_feature_importance(
    importance: FeatureImportance,
    max_features: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = 'steelblue',
    title: Optional[str] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot feature importance scores.
    
    Parameters
    ----------
    importance : FeatureImportance
        Feature importance object.
    max_features : int
        Maximum number of features to display.
    figsize : tuple, optional
        Figure size.
    color : str
        Bar color.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments for plotting.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    # Get top features
    df = importance.as_dataframe().head(max_features)
    
    # Create figure
    if figsize is None:
        figsize = (10, max(6, len(df) * 0.3))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['importance'], color=color, **kwargs)
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Importance Score')
    
    # Set title
    if title is None:
        title = f'Feature Importance ({importance.method})'
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Invert y-axis to have most important at top
    ax.invert_yaxis()
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, ax


def plot_shap_summary(
    shap_explanation: SHAPExplanation,
    plot_type: str = "dot",
    max_features: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Create SHAP summary plot.
    
    Parameters
    ----------
    shap_explanation : SHAPExplanation
        SHAP explanation object.
    plot_type : str
        Type of plot ("dot", "bar", "violin").
    max_features : int
        Maximum number of features to display.
    figsize : tuple, optional
        Figure size.
    **kwargs
        Additional arguments for SHAP plotting.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if not _HAS_SHAP:
        raise ImportError("SHAP is required for this function")
        
    # Create figure
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
        ax = plt.gca()
        
    # Create SHAP plot
    if plot_type == "dot":
        shap.summary_plot(
            shap_explanation.values,
            shap_explanation.data,
            feature_names=shap_explanation.feature_names,
            max_display=max_features,
            show=False,
            **kwargs
        )
    elif plot_type == "bar":
        shap.summary_plot(
            shap_explanation.values,
            feature_names=shap_explanation.feature_names,
            plot_type="bar",
            max_display=max_features,
            show=False,
            **kwargs
        )
    elif plot_type == "violin":
        shap.summary_plot(
            shap_explanation.values,
            shap_explanation.data,
            feature_names=shap_explanation.feature_names,
            plot_type="violin",
            max_display=max_features,
            show=False,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
        
    return fig, ax


def plot_lime_explanation(
    lime_explanation: LIMEExplanation,
    num_features: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    positive_color: str = 'green',
    negative_color: str = 'red',
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot LIME explanation.
    
    Parameters
    ----------
    lime_explanation : LIMEExplanation
        LIME explanation object.
    num_features : int
        Number of features to display.
    figsize : tuple, optional
        Figure size.
    positive_color : str
        Color for positive contributions.
    negative_color : str
        Color for negative contributions.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    # Get top features
    top_features = lime_explanation.get_top_features(num_features)
    
    # Prepare data
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    colors = [positive_color if imp >= 0 else negative_color for imp in importances]
    
    # Create figure
    if figsize is None:
        figsize = (10, max(6, len(features) * 0.4))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, color=colors, **kwargs)
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Contribution')
    
    # Add title
    ax.set_title(f'LIME Explanation for Instance {lime_explanation.instance_idx}')
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Invert y-axis
    ax.invert_yaxis()
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, ax


def plot_interpretation_comparison(
    results: List[InterpretationResult],
    max_features: int = 15,
    figsize: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare feature importance across different interpretation methods.
    
    Parameters
    ----------
    results : list of InterpretationResult
        Results from different interpreters.
    max_features : int
        Maximum number of features to display.
    figsize : tuple, optional
        Figure size.
    normalize : bool
        Whether to normalize importance scores.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    # Collect all feature importances
    all_features = set()
    importance_data = {}
    
    for result in results:
        if not result.has_global_importance():
            warnings.warn(f"Result from {result.method} has no global importance")
            continue
            
        imp = result.global_importance
        df = imp.as_dataframe()
        
        # Store importance scores
        importance_data[result.method] = dict(zip(df['feature'], df['importance']))
        all_features.update(df['feature'])
        
    if not importance_data:
        raise ValueError("No results have global importance")
        
    # Create DataFrame for comparison
    comparison_df = pd.DataFrame(importance_data, index=list(all_features))
    comparison_df = comparison_df.fillna(0)
    
    # Normalize if requested
    if normalize:
        for col in comparison_df.columns:
            max_val = comparison_df[col].abs().max()
            if max_val > 0:
                comparison_df[col] = comparison_df[col] / max_val
                
    # Calculate mean importance and sort
    comparison_df['mean_importance'] = comparison_df.abs().mean(axis=1)
    comparison_df = comparison_df.sort_values('mean_importance', ascending=False)
    comparison_df = comparison_df.drop('mean_importance', axis=1)
    
    # Select top features
    comparison_df = comparison_df.head(max_features)
    
    # Create figure
    if figsize is None:
        figsize = (10, max(6, len(comparison_df) * 0.4))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grouped bar plot
    x = np.arange(len(comparison_df))
    width = 0.8 / len(comparison_df.columns)
    
    for i, method in enumerate(comparison_df.columns):
        offset = (i - len(comparison_df.columns)/2 + 0.5) * width
        ax.barh(x + offset, comparison_df[method], width, label=method)
        
    # Set labels
    ax.set_yticks(x)
    ax.set_yticklabels(comparison_df.index)
    ax.set_xlabel('Importance Score' + (' (Normalized)' if normalize else ''))
    ax.set_title('Feature Importance Comparison')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Invert y-axis
    ax.invert_yaxis()
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, ax


def create_interpretation_report(
    result: InterpretationResult,
    output_path: Optional[str] = None,
    include_local: bool = True,
    max_features: int = 20
) -> str:
    """Create a text report of interpretation results.
    
    Parameters
    ----------
    result : InterpretationResult
        Interpretation result.
    output_path : str, optional
        Path to save report.
    include_local : bool
        Whether to include local explanations.
    max_features : int
        Maximum features to include.
        
    Returns
    -------
    str
        Report text.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"Model Interpretation Report")
    lines.append("=" * 60)
    lines.append(f"Method: {result.method.upper()}")
    lines.append(f"Learner: {result.learner.id}")
    lines.append(f"Task: {result.task.id}")
    lines.append("")
    
    # Global importance
    if result.has_global_importance():
        lines.append("Global Feature Importance:")
        lines.append("-" * 40)
        
        df = result.global_importance.as_dataframe().head(max_features)
        for _, row in df.iterrows():
            lines.append(f"{row['feature']:30s} {row['importance']:10.4f}")
        lines.append("")
        
    # Local explanations
    if include_local and result.has_local_explanations():
        lines.append("Local Explanations:")
        lines.append("-" * 40)
        
        for idx, explanation in result.local_explanations.items():
            lines.append(f"\nInstance {idx}:")
            
            if isinstance(explanation, LIMEExplanation):
                top_features = explanation.get_top_features(10)
                for feat, imp in top_features:
                    lines.append(f"  {feat:28s} {imp:10.4f}")
                    
            elif isinstance(explanation, SHAPExplanation):
                importance = explanation.get_feature_importance()
                df = importance.as_dataframe().head(10)
                for _, row in df.iterrows():
                    lines.append(f"  {row['feature']:28s} {row['importance']:10.4f}")
                    
    # Metadata
    if result.metadata:
        lines.append("\nMetadata:")
        lines.append("-" * 40)
        for key, value in result.metadata.items():
            if key not in ['explainer', 'shap_explanation']:  # Skip complex objects
                lines.append(f"{key}: {value}")
                
    report = "\n".join(lines)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
            
    return report


# Add convenience functions to InterpretationResult
def _add_result_methods():
    """Add plotting methods to InterpretationResult."""
    
    def plot_importance(self, **kwargs):
        """Plot global feature importance."""
        if not self.has_global_importance():
            raise ValueError("No global importance available")
        return plot_feature_importance(self.global_importance, **kwargs)
        
    def create_report(self, **kwargs):
        """Create interpretation report."""
        return create_interpretation_report(self, **kwargs)
        
    InterpretationResult.plot_importance = plot_importance
    InterpretationResult.create_report = create_report


# Auto-register methods
_add_result_methods()


__all__ = [
    "plot_feature_importance",
    "plot_shap_summary",
    "plot_lime_explanation",
    "plot_interpretation_comparison",
    "create_interpretation_report"
]