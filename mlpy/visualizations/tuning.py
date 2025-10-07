"""Visualization functions for TuneResult objects."""

from typing import Optional, Tuple, List, Union, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import warnings

from ..automl.tuning import TuneResult
from .base import PlotBase, get_colors
from .utils import (
    add_reference_line, format_axis_labels, add_grid,
    despine, adjust_legend, set_axis_limits
)


class TuningPerformance(PlotBase):
    """Performance visualization for hyperparameter tuning."""
    
    def create(
        self,
        result: TuneResult,
        param: Optional[str] = None,
        show_best: bool = True,
        colors: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create tuning performance plot.
        
        Parameters
        ----------
        result : TuneResult
            Tuning result to visualize.
        param : str, optional
            Parameter to plot against score. If None, plots all configs.
        show_best : bool
            Whether to highlight best configuration.
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
        
        # Get colors
        if colors is None:
            colors = get_colors(2)
            
        # Convert to DataFrame
        df = result.as_data_frame()
        score_col = f"{result.measure.id}_score"
        
        if param is None:
            # Plot all configurations
            x = range(len(df))
            y = df[score_col].values
            
            # Create scatter plot
            scatter = ax.scatter(
                x, y,
                c=y,
                cmap='viridis',
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Highlight best
            if show_best:
                best_idx = df[df['is_best']].index[0]
                ax.scatter(
                    best_idx, y[best_idx],
                    color='red',
                    s=200,
                    marker='*',
                    edgecolors='black',
                    linewidth=1,
                    label='Best',
                    zorder=10
                )
                
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(score_col)
            
            ax.set_xlabel('Configuration Index')
            ax.set_ylabel(score_col)
            
        else:
            # Plot against specific parameter
            if param not in df.columns:
                warnings.warn(f"Parameter '{param}' not found")
                return fig, ax
                
            x = df[param].values
            y = df[score_col].values
            
            # Handle categorical parameters
            if df[param].dtype == 'object':
                # Convert to numeric for plotting
                unique_vals = df[param].unique()
                x_numeric = pd.Categorical(df[param]).codes
                
                ax.scatter(
                    x_numeric, y,
                    color=colors[0],
                    s=50,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                # Set tick labels
                ax.set_xticks(range(len(unique_vals)))
                ax.set_xticklabels(unique_vals)
                
            else:
                # Numeric parameter
                ax.scatter(
                    x, y,
                    color=colors[0],
                    s=50,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                # Add trend line
                if len(np.unique(x)) > 1:
                    z = np.polyfit(x, y, 2)  # Quadratic fit
                    p = np.poly1d(z)
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    ax.plot(
                        x_smooth, p(x_smooth),
                        color=colors[1],
                        linestyle='--',
                        alpha=0.7,
                        label='Trend'
                    )
                    
            # Highlight best
            if show_best:
                best_row = df[df['is_best']].iloc[0]
                if df[param].dtype == 'object':
                    best_x = pd.Categorical([best_row[param]], categories=unique_vals).codes[0]
                else:
                    best_x = best_row[param]
                    
                ax.scatter(
                    best_x, best_row[score_col],
                    color='red',
                    s=200,
                    marker='*',
                    edgecolors='black',
                    linewidth=1,
                    label='Best',
                    zorder=10
                )
                
            ax.set_xlabel(param)
            ax.set_ylabel(score_col)
            
        # Add title
        self._add_title(ax, f'Tuning Performance: {result.learner.id}')
        
        # Format
        add_grid(ax)
        adjust_legend(ax)
        despine(ax)
        
        return fig, ax


def plot_tuning_performance(
    result: TuneResult,
    param: Optional[str] = None,
    show_best: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Create tuning performance plot.
    
    Parameters
    ----------
    result : TuneResult
        Tuning result to visualize.
    param : str, optional
        Parameter to plot against score.
    show_best : bool
        Whether to highlight best configuration.
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
    plotter = TuningPerformance(figsize=figsize)
    return plotter.create(result, param, show_best, **kwargs)


class TuningParallelCoordinates(PlotBase):
    """Parallel coordinates plot for hyperparameter configurations."""
    
    def create(
        self,
        result: TuneResult,
        color_by_score: bool = True,
        highlight_best: bool = True,
        alpha: float = 0.5,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create parallel coordinates plot.
        
        Parameters
        ----------
        result : TuneResult
            Tuning result to visualize.
        color_by_score : bool
            Whether to color lines by score.
        highlight_best : bool
            Whether to highlight best configuration.
        alpha : float
            Line transparency.
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
        fig, ax = self._setup_figure(figsize=(12, 6))
        
        # Convert to DataFrame
        df = result.as_data_frame()
        score_col = f"{result.measure.id}_score"
        
        # Get parameter columns
        param_cols = [col for col in df.columns 
                     if col not in [score_col, 'is_best']]
        
        if not param_cols:
            warnings.warn("No parameters to plot")
            return fig, ax
            
        # Normalize data for plotting
        df_norm = df.copy()
        
        for col in param_cols:
            if df[col].dtype == 'object':
                # Handle categorical
                df_norm[col] = pd.Categorical(df[col]).codes
                df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
            else:
                # Normalize numeric
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                
        # Normalize scores for coloring
        scores_norm = (df[score_col] - df[score_col].min()) / (df[score_col].max() - df[score_col].min())
        
        # Create parallel coordinates
        x = np.arange(len(param_cols))
        
        # Plot each configuration
        for idx, row in df_norm.iterrows():
            y = [row[col] for col in param_cols]
            
            if color_by_score:
                color = plt.cm.viridis(scores_norm.iloc[idx])
            else:
                color = 'blue'
                
            # Highlight best
            if highlight_best and df.iloc[idx]['is_best']:
                ax.plot(x, y, color='red', linewidth=3, alpha=1.0, zorder=10, label='Best')
            else:
                ax.plot(x, y, color=color, linewidth=1, alpha=alpha)
                
        # Add axes for each parameter
        for i, col in enumerate(param_cols):
            ax.axvline(i, color='black', linewidth=1, alpha=0.5)
            
            # Add labels for categorical parameters
            if df[col].dtype == 'object':
                unique_vals = df[col].unique()
                n_unique = len(unique_vals)
                positions = np.linspace(0, 1, n_unique)
                
                for j, val in enumerate(unique_vals):
                    ax.text(
                        i + 0.05, positions[j], str(val),
                        fontsize=8, va='center'
                    )
                    
        # Format axes
        ax.set_xticks(x)
        ax.set_xticklabels(param_cols, rotation=45, ha='right')
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Normalized Value')
        
        # Add colorbar if coloring by score
        if color_by_score:
            sm = plt.cm.ScalarMappable(
                cmap='viridis',
                norm=plt.Normalize(vmin=df[score_col].min(), vmax=df[score_col].max())
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(score_col)
            
        # Add title
        self._add_title(ax, 'Parallel Coordinates: Hyperparameter Configurations')
        
        # Format
        ax.grid(True, axis='y', alpha=0.3)
        adjust_legend(ax)
        
        # Remove y-axis labels (normalized values not meaningful)
        ax.set_yticklabels([])
        
        return fig, ax


def plot_tuning_parallel_coordinates(
    result: TuneResult,
    color_by_score: bool = True,
    highlight_best: bool = True,
    alpha: float = 0.5,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Create parallel coordinates plot for tuning results.
    
    Parameters
    ----------
    result : TuneResult
        Tuning result to visualize.
    color_by_score : bool
        Whether to color lines by score.
    highlight_best : bool
        Whether to highlight best configuration.
    alpha : float
        Line transparency.
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
    plotter = TuningParallelCoordinates(figsize=figsize)
    return plotter.create(result, color_by_score, highlight_best, alpha, **kwargs)


# Add plot methods to TuneResult
def _add_tune_plot_methods():
    """Add plotting methods to TuneResult class."""
    
    # Import here to avoid circular imports
    from ..automl.tuning import TuneResult
    
    # Already has a plot method, so we'll add specific ones
    def plot_performance(self, **kwargs):
        """Create performance plot."""
        return plot_tuning_performance(self, **kwargs)
        
    def plot_parallel_coordinates(self, **kwargs):
        """Create parallel coordinates plot."""
        return plot_tuning_parallel_coordinates(self, **kwargs)
        
    TuneResult.plot_performance = plot_performance
    TuneResult.plot_parallel_coordinates = plot_parallel_coordinates


# Auto-register methods
_add_tune_plot_methods()


__all__ = [
    "TuningPerformance",
    "TuningParallelCoordinates",
    "plot_tuning_performance",
    "plot_tuning_parallel_coordinates"
]