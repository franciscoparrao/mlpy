"""Base classes and utilities for MLPY visualizations.

This module provides the foundation for all plotting functionality in MLPY,
including themes, base classes, and common utilities.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import matplotlib.pyplot as plt
import matplotlib as mpl
from abc import ABC, abstractmethod
import seaborn as sns
import warnings

from ..base import MLPYObject


# Default plot themes
THEMES = {
    "default": {
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
    "minimal": {
        "figure.figsize": (10, 6),
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
    },
    "publication": {
        "figure.figsize": (8, 6),
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    }
}

# Current theme
_current_theme = "default"


def plot_theme(theme: Optional[str] = None) -> Dict[str, Any]:
    """Get or set the current plot theme.
    
    Parameters
    ----------
    theme : str, optional
        Theme name to set. If None, returns current theme settings.
        Available themes: 'default', 'minimal', 'publication'.
        
    Returns
    -------
    dict
        Current theme settings.
    """
    global _current_theme
    
    if theme is not None:
        if theme not in THEMES:
            raise ValueError(f"Unknown theme '{theme}'. Available: {list(THEMES.keys())}")
        _current_theme = theme
        
    return THEMES[_current_theme].copy()


def set_plot_theme(theme: Union[str, Dict[str, Any]]) -> None:
    """Set the matplotlib plotting theme.
    
    Parameters
    ----------
    theme : str or dict
        Either a theme name ('default', 'minimal', 'publication')
        or a dictionary of matplotlib rc parameters.
    """
    if isinstance(theme, str):
        if theme not in THEMES:
            raise ValueError(f"Unknown theme '{theme}'. Available: {list(THEMES.keys())}")
        settings = THEMES[theme]
    else:
        settings = theme
        
    # Apply settings to matplotlib
    for key, value in settings.items():
        mpl.rcParams[key] = value


class PlotBase(MLPYObject, ABC):
    """Abstract base class for MLPY plots.
    
    All MLPY plotting classes should inherit from this base class.
    It provides common functionality and enforces a consistent API.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier for the plot.
    figsize : tuple, optional
        Figure size as (width, height).
    title : str, optional
        Plot title.
    theme : str, optional
        Theme to use for this plot.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        theme: Optional[str] = None
    ):
        super().__init__(id=id or self.__class__.__name__.lower())
        self.figsize = figsize
        self.title = title
        self.theme = theme
        self._fig = None
        self._axes = None
        
    @abstractmethod
    def create(self, *args, **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
        """Create the plot.
        
        This method must be implemented by subclasses.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes or list of Axes
            The axes object(s).
        """
        pass
        
    def _setup_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        **subplot_kwargs
    ) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
        """Setup figure and axes with theme.
        
        Parameters
        ----------
        nrows : int
            Number of subplot rows.
        ncols : int
            Number of subplot columns.
        **subplot_kwargs
            Additional arguments for plt.subplots.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes or array of Axes
            The axes object(s).
        """
        # Apply theme if specified
        if self.theme:
            original_params = mpl.rcParams.copy()
            set_plot_theme(self.theme)
            
        # Get figsize
        if self.figsize is None:
            figsize = plot_theme()[self.theme or _current_theme]["figure.figsize"]
        else:
            figsize = self.figsize
            
        # Create figure
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **subplot_kwargs)
        
        # Restore original params if theme was temporary
        if self.theme:
            mpl.rcParams.update(original_params)
            
        self._fig = fig
        self._axes = ax
        
        return fig, ax
        
    def _add_title(self, ax: plt.Axes, title: Optional[str] = None) -> None:
        """Add title to axes.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to add title to.
        title : str, optional
            Title text. Uses self.title if not provided.
        """
        title = title or self.title
        if title:
            ax.set_title(title)
            
    def _format_axes(
        self,
        ax: plt.Axes,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xscale: Optional[str] = None,
        yscale: Optional[str] = None,
        grid: Optional[bool] = None
    ) -> None:
        """Format axes with common settings.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to format.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        xlim : tuple, optional
            X-axis limits.
        ylim : tuple, optional
            Y-axis limits.
        xscale : str, optional
            X-axis scale ('linear', 'log', etc.).
        yscale : str, optional
            Y-axis scale ('linear', 'log', etc.).
        grid : bool, optional
            Whether to show grid.
        """
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
        if grid is not None:
            ax.grid(grid, alpha=0.3)
            
    def show(self) -> None:
        """Display the plot."""
        if self._fig is None:
            raise ValueError("Plot has not been created yet. Call create() first.")
        plt.show()
        
    def save(
        self,
        filename: str,
        dpi: Optional[int] = None,
        bbox_inches: str = "tight",
        **kwargs
    ) -> None:
        """Save the plot to file.
        
        Parameters
        ----------
        filename : str
            Output filename.
        dpi : int, optional
            Resolution in dots per inch.
        bbox_inches : str
            Bounding box setting.
        **kwargs
            Additional arguments for savefig.
        """
        if self._fig is None:
            raise ValueError("Plot has not been created yet. Call create() first.")
            
        if dpi is None:
            dpi = mpl.rcParams.get("figure.dpi", 100)
            
        self._fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)


# Color palettes for consistent styling
COLORS = {
    "default": sns.color_palette("deep"),
    "muted": sns.color_palette("muted"),
    "bright": sns.color_palette("bright"),
    "pastel": sns.color_palette("pastel"),
    "dark": sns.color_palette("dark"),
    "colorblind": sns.color_palette("colorblind")
}


def get_colors(
    n_colors: int,
    palette: str = "default",
    as_hex: bool = False
) -> List[Union[Tuple[float, float, float], str]]:
    """Get a list of colors from a palette.
    
    Parameters
    ----------
    n_colors : int
        Number of colors needed.
    palette : str
        Palette name.
    as_hex : bool
        Whether to return colors as hex strings.
        
    Returns
    -------
    list
        List of colors as RGB tuples or hex strings.
    """
    if palette not in COLORS:
        # Try to use it as a seaborn palette
        try:
            colors = sns.color_palette(palette, n_colors)
        except:
            warnings.warn(f"Unknown palette '{palette}', using default")
            colors = COLORS["default"]
    else:
        colors = COLORS[palette]
        
    # Cycle if we need more colors
    colors = list(colors) * (n_colors // len(colors) + 1)
    colors = colors[:n_colors]
    
    if as_hex:
        colors = [mpl.colors.to_hex(c) for c in colors]
        
    return colors


__all__ = [
    "PlotBase",
    "plot_theme",
    "set_plot_theme",
    "THEMES",
    "COLORS",
    "get_colors"
]