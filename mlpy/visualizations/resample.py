"""Visualization functions for ResampleResult objects."""

from typing import Optional, Tuple, List, Union, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from ..resample import ResampleResult
from .base import PlotBase, get_colors
from .utils import (
    annotate_bars, add_reference_line, format_axis_labels,
    add_grid, despine, adjust_legend, set_axis_limits
)


class ResampleBoxplot(PlotBase):
    """Boxplot visualization for ResampleResult scores."""
    
    def create(
        self,
        result: ResampleResult,
        measure_id: Optional[str] = None,
        show_points: bool = True,
        colors: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create boxplot of resampling scores.
        
        Parameters
        ----------
        result : ResampleResult
            Result object to visualize.
        measure_id : str, optional
            Measure to plot. If None, uses first measure.
        show_points : bool
            Whether to overlay individual points.
        colors : list of str, optional
            Colors to use.
        **kwargs
            Additional arguments for boxplot.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        # Setup figure
        fig, ax = self._setup_figure()
        
        # Get measure
        if measure_id is None:
            measure_id = result.measures[0].id
            
        # Get scores
        scores = result.scores[measure_id]
        valid_scores = [s for s in scores if not np.isnan(s)]
        
        if not valid_scores:
            warnings.warn(f"No valid scores for measure '{measure_id}'")
            return fig, ax
            
        # Create boxplot
        bp = ax.boxplot(
            valid_scores,
            patch_artist=True,
            widths=0.6,
            **kwargs
        )
        
        # Style boxplot
        if colors is None:
            colors = get_colors(1)
        color = colors[0]
        
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        
        # Add individual points
        if show_points:
            x = np.random.normal(1, 0.04, len(valid_scores))
            ax.scatter(x, valid_scores, alpha=0.5, s=30, color=color)
            
        # Add mean line
        mean_score = np.mean(valid_scores)
        ax.axhline(mean_score, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_score:.4f}')
        
        # Format axes
        ax.set_xticklabels([result.learner.id])
        ax.set_ylabel(f'{measure_id} score')
        self._add_title(ax, f'Resampling Results: {result.learner.id}')
        
        # Add grid and legend
        add_grid(ax, axis='y')
        adjust_legend(ax)
        despine(ax)
        
        return fig, ax


def plot_resample_boxplot(
    result: ResampleResult,
    measure_id: Optional[str] = None,
    show_points: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Create boxplot of resampling scores.
    
    Parameters
    ----------
    result : ResampleResult
        Result object to visualize.
    measure_id : str, optional
        Measure to plot. If None, uses first measure.
    show_points : bool
        Whether to overlay individual points.
    figsize : tuple, optional
        Figure size.
    **kwargs
        Additional arguments for plotting.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    plotter = ResampleBoxplot(figsize=figsize)
    return plotter.create(result, measure_id, show_points, **kwargs)


class ResampleROCCurve(PlotBase):
    """ROC curve visualization for classification ResampleResult."""
    
    def create(
        self,
        result: ResampleResult,
        average: bool = True,
        confidence: bool = True,
        colors: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create ROC curve plot.
        
        Parameters
        ----------
        result : ResampleResult
            Result object with classification predictions.
        average : bool
            Whether to show average ROC curve.
        confidence : bool
            Whether to show confidence bands.
        colors : list of str, optional
            Colors to use.
        **kwargs
            Additional arguments.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        # Setup figure
        fig, ax = self._setup_figure()
        
        # Check if we have probability predictions
        if not result.predictions or not hasattr(result.predictions[0], 'prob'):
            warnings.warn("No probability predictions available for ROC curve")
            return fig, ax
            
        # Get colors
        if colors is None:
            colors = get_colors(2)
            
        # Collect ROC data from each iteration
        all_fpr = []
        all_tpr = []
        all_auc = []
        
        for pred in result.predictions:
            if pred is None or pred.prob is None:
                continue
                
            # Calculate ROC for binary classification
            # This is simplified - full implementation would handle multiclass
            from sklearn.metrics import roc_curve, auc
            
            try:
                # Assume binary classification with positive class
                y_true = (pred.truth == pred.task.positive)
                y_score = pred.prob.iloc[:, 1]  # Probability of positive class
                
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                
                all_fpr.append(fpr)
                all_tpr.append(tpr)
                all_auc.append(roc_auc)
                
                # Plot individual curves
                if not average:
                    ax.plot(fpr, tpr, alpha=0.3, color=colors[0])
                    
            except Exception as e:
                warnings.warn(f"Could not compute ROC curve: {e}")
                continue
                
        if not all_fpr:
            warnings.warn("Could not compute any ROC curves")
            return fig, ax
            
        # Plot average ROC curve
        if average and len(all_fpr) > 1:
            # Interpolate all curves to common FPR points
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            
            for fpr, tpr in zip(all_fpr, all_tpr):
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[0] = 0.0
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(all_auc)
            
            # Plot mean curve
            ax.plot(
                mean_fpr, mean_tpr,
                color=colors[1],
                linewidth=2,
                label=f'Mean ROC (AUC = {mean_auc:.3f})'
            )
            
            # Add confidence bands
            if confidence:
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                ax.fill_between(
                    mean_fpr, tprs_lower, tprs_upper,
                    color=colors[1], alpha=0.2
                )
                
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        # Format axes
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        self._add_title(ax, 'ROC Curve')
        
        # Add grid and legend
        add_grid(ax)
        adjust_legend(ax, loc='lower right')
        ax.set_aspect('equal')
        
        return fig, ax


def plot_resample_roc(
    result: ResampleResult,
    average: bool = True,
    confidence: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Create ROC curve plot for classification results.
    
    Parameters
    ----------
    result : ResampleResult
        Result object with classification predictions.
    average : bool
        Whether to show average ROC curve.
    confidence : bool
        Whether to show confidence bands.
    figsize : tuple, optional
        Figure size.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    plotter = ResampleROCCurve(figsize=figsize)
    return plotter.create(result, average, confidence, **kwargs)


class ResampleIterations(PlotBase):
    """Plot scores across resampling iterations."""
    
    def create(
        self,
        result: ResampleResult,
        measure_ids: Optional[List[str]] = None,
        show_mean: bool = True,
        colors: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create iteration plot.
        
        Parameters
        ----------
        result : ResampleResult
            Result object to visualize.
        measure_ids : list of str, optional
            Measures to plot. If None, plots all.
        show_mean : bool
            Whether to show mean lines.
        colors : list of str, optional
            Colors to use.
        **kwargs
            Additional arguments.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        """
        # Setup figure
        fig, ax = self._setup_figure()
        
        # Get measures to plot
        if measure_ids is None:
            measure_ids = list(result.scores.keys())
            
        # Get colors
        if colors is None:
            colors = get_colors(len(measure_ids))
            
        # Plot each measure
        for i, measure_id in enumerate(measure_ids):
            scores = result.scores[measure_id]
            iterations = range(len(scores))
            
            # Plot line
            ax.plot(
                iterations, scores,
                marker='o',
                color=colors[i],
                label=measure_id,
                alpha=0.8,
                **kwargs
            )
            
            # Add mean line
            if show_mean:
                valid_scores = [s for s in scores if not np.isnan(s)]
                if valid_scores:
                    mean_score = np.mean(valid_scores)
                    ax.axhline(
                        mean_score,
                        color=colors[i],
                        linestyle='--',
                        alpha=0.5
                    )
                    
        # Format axes
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(result.iterations)))
        self._add_title(ax, 'Scores Across Iterations')
        
        # Add grid and legend
        add_grid(ax)
        adjust_legend(ax)
        despine(ax)
        
        return fig, ax


def plot_resample_iterations(
    result: ResampleResult,
    measure_ids: Optional[List[str]] = None,
    show_mean: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot scores across resampling iterations.
    
    Parameters
    ----------
    result : ResampleResult
        Result object to visualize.
    measure_ids : list of str, optional
        Measures to plot. If None, plots all.
    show_mean : bool
        Whether to show mean lines.
    figsize : tuple, optional
        Figure size.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    plotter = ResampleIterations(figsize=figsize)
    return plotter.create(result, measure_ids, show_mean, **kwargs)


# Add plot methods to ResampleResult
def _add_resample_plot_methods():
    """Add plotting methods to ResampleResult class."""
    
    def plot_boxplot(self, **kwargs):
        """Create boxplot of scores."""
        return plot_resample_boxplot(self, **kwargs)
        
    def plot_roc(self, **kwargs):
        """Create ROC curve plot."""
        return plot_resample_roc(self, **kwargs)
        
    def plot_iterations(self, **kwargs):
        """Plot scores across iterations."""
        return plot_resample_iterations(self, **kwargs)
        
    ResampleResult.plot_boxplot = plot_boxplot
    ResampleResult.plot_roc = plot_roc
    ResampleResult.plot_iterations = plot_iterations


# Auto-register methods
_add_resample_plot_methods()


__all__ = [
    "ResampleBoxplot",
    "ResampleROCCurve", 
    "ResampleIterations",
    "plot_resample_boxplot",
    "plot_resample_roc",
    "plot_resample_iterations"
]