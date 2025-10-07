"""Visualization functions for BenchmarkResult objects."""

from typing import Optional, Tuple, List, Union, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from ..benchmark import BenchmarkResult
from .base import PlotBase, get_colors
from .utils import (
    annotate_bars, format_axis_labels, add_grid, 
    despine, adjust_legend, set_axis_limits
)


class BenchmarkBoxplot(PlotBase):
    """Boxplot comparison of benchmark results."""
    
    def create(
        self,
        result: BenchmarkResult,
        measure_id: Optional[str] = None,
        by: str = "learner",
        show_points: bool = False,
        colors: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create comparative boxplot.
        
        Parameters
        ----------
        result : BenchmarkResult
            Benchmark result to visualize.
        measure_id : str, optional
            Measure to plot. If None, uses first measure.
        by : str
            Group by 'learner' or 'task'.
        show_points : bool
            Whether to show individual points.
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
            
        # Get data in long format
        df = result.to_long_format()
        score_col = f'score_{measure_id}'
        
        if score_col not in df.columns:
            warnings.warn(f"No scores for measure '{measure_id}'")
            return fig, ax
            
        # Filter valid scores
        df_valid = df[df[score_col].notna()].copy()
        
        if df_valid.empty:
            warnings.warn("No valid scores to plot")
            return fig, ax
            
        # Determine grouping
        if by == "learner":
            x_col = "learner_id"
            group_col = "task_id"
            x_label = "Learner"
        else:
            x_col = "task_id"
            group_col = "learner_id"
            x_label = "Task"
            
        # Get unique groups and positions
        unique_x = df_valid[x_col].unique()
        positions = np.arange(len(unique_x))
        
        # Get colors
        if colors is None:
            n_groups = df_valid[group_col].nunique()
            colors = get_colors(n_groups)
            
        # Create grouped boxplots
        data_by_x = []
        labels = []
        
        for i, x_val in enumerate(unique_x):
            x_data = df_valid[df_valid[x_col] == x_val][score_col].values
            data_by_x.append(x_data)
            labels.append(x_val)
            
        # Create boxplot
        bp = ax.boxplot(
            data_by_x,
            positions=positions,
            patch_artist=True,
            widths=0.6,
            **kwargs
        )
        
        # Style boxplots
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
            box.set_alpha(0.7)
            
        # Add individual points if requested
        if show_points:
            for i, (x_val, data) in enumerate(zip(unique_x, data_by_x)):
                x = np.random.normal(i, 0.04, len(data))
                ax.scatter(x, data, alpha=0.5, s=20, color=colors[i % len(colors)])
                
        # Format axes
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f'{measure_id} score')
        self._add_title(ax, f'Benchmark Comparison: {measure_id}')
        
        # Rotate labels if many
        if len(labels) > 5:
            format_axis_labels(ax, rotation=45)
            
        # Add grid
        add_grid(ax, axis='y')
        despine(ax)
        
        return fig, ax


def plot_benchmark_boxplot(
    result: BenchmarkResult,
    measure_id: Optional[str] = None,
    by: str = "learner",
    show_points: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Create comparative boxplot of benchmark results.
    
    Parameters
    ----------
    result : BenchmarkResult
        Benchmark result to visualize.
    measure_id : str, optional
        Measure to plot.
    by : str
        Group by 'learner' or 'task'.
    show_points : bool
        Whether to show individual points.
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
    plotter = BenchmarkBoxplot(figsize=figsize)
    return plotter.create(result, measure_id, by, show_points, **kwargs)


class BenchmarkHeatmap(PlotBase):
    """Heatmap visualization of benchmark results."""
    
    def create(
        self,
        result: BenchmarkResult,
        measure_id: Optional[str] = None,
        aggr: str = "mean",
        annotate: bool = True,
        cmap: str = "viridis",
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create heatmap of scores.
        
        Parameters
        ----------
        result : BenchmarkResult
            Benchmark result to visualize.
        measure_id : str, optional
            Measure to plot.
        aggr : str
            Aggregation method.
        annotate : bool
            Whether to annotate cells with values.
        cmap : str
            Colormap name.
        **kwargs
            Additional arguments for heatmap.
            
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
            
        # Get aggregated scores
        scores_df = result.aggregate(measure_id, aggr)
        
        # Create heatmap
        sns.heatmap(
            scores_df,
            annot=annotate,
            fmt='.3f',
            cmap=cmap,
            cbar_kws={'label': f'{measure_id} ({aggr})'},
            ax=ax,
            **kwargs
        )
        
        # Format
        ax.set_xlabel('Learner')
        ax.set_ylabel('Task')
        self._add_title(ax, f'Benchmark Results: {measure_id}')
        
        # Rotate labels
        format_axis_labels(ax, 'x', rotation=45)
        format_axis_labels(ax, 'y', rotation=0)
        
        return fig, ax


def plot_benchmark_heatmap(
    result: BenchmarkResult,
    measure_id: Optional[str] = None,
    aggr: str = "mean",
    annotate: bool = True,
    cmap: str = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Create heatmap of benchmark scores.
    
    Parameters
    ----------
    result : BenchmarkResult
        Benchmark result to visualize.
    measure_id : str, optional
        Measure to plot.
    aggr : str
        Aggregation method.
    annotate : bool
        Whether to annotate cells.
    cmap : str
        Colormap name.
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
    plotter = BenchmarkHeatmap(figsize=figsize)
    return plotter.create(result, measure_id, aggr, annotate, cmap, **kwargs)


class BenchmarkCriticalDifference(PlotBase):
    """Critical difference diagram for benchmark results."""
    
    def create(
        self,
        result: BenchmarkResult,
        measure_id: Optional[str] = None,
        alpha: float = 0.05,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create critical difference diagram.
        
        This is a simplified version. A full implementation would
        include statistical tests (Friedman, Nemenyi).
        
        Parameters
        ----------
        result : BenchmarkResult
            Benchmark result to visualize.
        measure_id : str, optional
            Measure to use for ranking.
        alpha : float
            Significance level.
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
        fig, ax = self._setup_figure(figsize=(10, 6))
        
        # Get measure
        if measure_id is None:
            measure_id = result.measures[0].id
            
        # Get rankings
        rankings_df = result.rank_learners(measure_id)
        n_learners = len(rankings_df)
        
        # Plot ranks
        y_positions = np.arange(n_learners)
        
        for i, row in rankings_df.iterrows():
            learner = row['learner']
            rank = row['rank']
            score = row['mean_score']
            
            # Plot rank line
            ax.plot([rank, rank], [i - 0.1, i + 0.1], 'k-', linewidth=2)
            
            # Add learner name
            ax.text(
                rank - 0.1, i, learner,
                ha='right', va='center', fontsize=10
            )
            
            # Add score
            ax.text(
                rank + 0.1, i, f'{score:.3f}',
                ha='left', va='center', fontsize=9, color='gray'
            )
            
        # Add rank axis at top
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Rank')
        
        # Format main axes
        ax.set_xlim(0.5, n_learners + 0.5)
        ax.set_ylim(-0.5, n_learners - 0.5)
        ax.set_xlabel('Average Rank')
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)
        
        # Remove spines
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
            
        self._add_title(ax, f'Critical Difference Diagram: {measure_id}')
        
        # Add note about statistical tests
        ax.text(
            0.5, -0.05, 
            'Note: Statistical significance tests not shown',
            transform=ax.transAxes,
            ha='center', fontsize=8, style='italic'
        )
        
        return fig, ax


def plot_benchmark_critical_difference(
    result: BenchmarkResult,
    measure_id: Optional[str] = None,
    alpha: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Create critical difference diagram.
    
    Parameters
    ----------
    result : BenchmarkResult
        Benchmark result to visualize.
    measure_id : str, optional
        Measure to use for ranking.
    alpha : float
        Significance level.
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
    plotter = BenchmarkCriticalDifference(figsize=figsize)
    return plotter.create(result, measure_id, alpha, **kwargs)


# Add plot methods to BenchmarkResult
def _add_benchmark_plot_methods():
    """Add plotting methods to BenchmarkResult class."""
    
    def plot_boxplot(self, **kwargs):
        """Create comparative boxplot."""
        return plot_benchmark_boxplot(self, **kwargs)
        
    def plot_heatmap(self, **kwargs):
        """Create heatmap of scores."""
        return plot_benchmark_heatmap(self, **kwargs)
        
    def plot_critical_difference(self, **kwargs):
        """Create critical difference diagram."""
        return plot_benchmark_critical_difference(self, **kwargs)
        
    BenchmarkResult.plot_boxplot = plot_boxplot
    BenchmarkResult.plot_heatmap = plot_heatmap
    BenchmarkResult.plot_critical_difference = plot_critical_difference


# Auto-register methods
_add_benchmark_plot_methods()


__all__ = [
    "BenchmarkBoxplot",
    "BenchmarkHeatmap",
    "BenchmarkCriticalDifference",
    "plot_benchmark_boxplot",
    "plot_benchmark_heatmap",
    "plot_benchmark_critical_difference"
]