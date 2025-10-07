"""Utility functions for MLPY visualizations."""

from typing import Optional, Tuple, List, Union, Dict, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings
from pathlib import Path


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """Create a figure with subplots.
    
    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    figsize : tuple, optional
        Figure size. If None, uses current theme default.
    **kwargs
        Additional arguments for plt.subplots.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes or array of Axes
        Axes object(s).
    """
    if figsize is None:
        figsize = mpl.rcParams.get("figure.figsize", (10, 6))
        
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def save_plot(
    fig: plt.Figure,
    filename: Union[str, Path],
    dpi: Optional[int] = None,
    formats: Optional[List[str]] = None,
    bbox_inches: str = "tight",
    **kwargs
) -> None:
    """Save a plot to one or more formats.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str or Path
        Base filename (without extension).
    dpi : int, optional
        Resolution. If None, uses current setting.
    formats : list of str, optional
        Output formats. If None, infers from filename or uses ['png'].
    bbox_inches : str
        Bounding box setting.
    **kwargs
        Additional arguments for savefig.
    """
    filename = Path(filename)
    
    # Determine formats
    if formats is None:
        if filename.suffix:
            formats = [filename.suffix[1:]]  # Remove dot
            filename = filename.with_suffix("")  # Remove extension
        else:
            formats = ["png"]
            
    # Default DPI
    if dpi is None:
        dpi = mpl.rcParams.get("savefig.dpi", 100)
        
    # Save in each format
    for fmt in formats:
        output_file = filename.with_suffix(f".{fmt}")
        fig.savefig(output_file, dpi=dpi, bbox_inches=bbox_inches, format=fmt, **kwargs)


def annotate_bars(
    ax: plt.Axes,
    bars: Union[Any, List],
    fmt: str = ".2f",
    offset: float = 0.02,
    fontsize: Optional[int] = None,
    ha: str = "center",
    va: str = "bottom"
) -> None:
    """Add value annotations to bar plots.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the bars.
    bars : BarContainer or list
        Bar objects to annotate.
    fmt : str
        Format string for values.
    offset : float
        Vertical offset as fraction of y-range.
    fontsize : int, optional
        Font size for annotations.
    ha : str
        Horizontal alignment.
    va : str
        Vertical alignment.
    """
    if fontsize is None:
        fontsize = mpl.rcParams.get("font.size", 10) - 2
        
    # Get y-range for offset calculation
    ymin, ymax = ax.get_ylim()
    offset_value = offset * (ymax - ymin)
    
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height) and not np.isinf(height):
            label = f"{height:{fmt}}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset_value if height >= 0 else -offset_value),
                textcoords="offset points",
                ha=ha,
                va=va if height >= 0 else "top",
                fontsize=fontsize
            )


def add_reference_line(
    ax: plt.Axes,
    value: float,
    axis: str = "y",
    color: str = "red",
    linestyle: str = "--",
    alpha: float = 0.7,
    label: Optional[str] = None,
    **kwargs
) -> plt.Line2D:
    """Add a reference line to axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add line to.
    value : float
        Position of the line.
    axis : str
        'x' for vertical line, 'y' for horizontal line.
    color : str
        Line color.
    linestyle : str
        Line style.
    alpha : float
        Line transparency.
    label : str, optional
        Label for legend.
    **kwargs
        Additional arguments for axhline/axvline.
        
    Returns
    -------
    line : matplotlib.lines.Line2D
        The line object.
    """
    if axis == "y":
        line = ax.axhline(value, color=color, linestyle=linestyle, alpha=alpha, label=label, **kwargs)
    elif axis == "x":
        line = ax.axvline(value, color=color, linestyle=linestyle, alpha=alpha, label=label, **kwargs)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
        
    return line


def format_axis_labels(
    ax: plt.Axes,
    axis: str = "x",
    rotation: float = 0,
    ha: Optional[str] = None,
    fontsize: Optional[int] = None,
    max_labels: Optional[int] = None
) -> None:
    """Format axis tick labels.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.
    axis : str
        Which axis to format ('x' or 'y').
    rotation : float
        Label rotation angle.
    ha : str, optional
        Horizontal alignment.
    fontsize : int, optional
        Font size.
    max_labels : int, optional
        Maximum number of labels to show.
    """
    if axis == "x":
        labels = ax.get_xticklabels()
        set_labels = ax.set_xticklabels
    else:
        labels = ax.get_yticklabels()
        set_labels = ax.set_yticklabels
        
    # Determine ha based on rotation if not specified
    if ha is None:
        if rotation > 0:
            ha = "right"
        elif rotation < 0:
            ha = "left"
        else:
            ha = "center"
            
    # Limit number of labels if requested
    if max_labels is not None and len(labels) > max_labels:
        step = len(labels) // max_labels
        for i, label in enumerate(labels):
            if i % step != 0:
                label.set_visible(False)
                
    # Apply formatting
    set_labels(labels, rotation=rotation, ha=ha, fontsize=fontsize)


def add_grid(
    ax: plt.Axes,
    axis: str = "both",
    which: str = "major",
    alpha: float = 0.3,
    linestyle: str = "-",
    **kwargs
) -> None:
    """Add grid to axes with custom styling.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add grid to.
    axis : str
        Which axis to show grid for ('x', 'y', or 'both').
    which : str
        'major', 'minor', or 'both'.
    alpha : float
        Grid transparency.
    linestyle : str
        Grid line style.
    **kwargs
        Additional arguments for grid.
    """
    ax.grid(True, axis=axis, which=which, alpha=alpha, linestyle=linestyle, **kwargs)


def set_axis_limits(
    ax: plt.Axes,
    data: Optional[np.ndarray] = None,
    axis: str = "y",
    pad: float = 0.1,
    symmetric: bool = False,
    include_zero: bool = False
) -> None:
    """Set axis limits with padding.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to set limits for.
    data : array-like, optional
        Data to base limits on. If None, uses current data.
    axis : str
        Which axis ('x' or 'y').
    pad : float
        Padding as fraction of range.
    symmetric : bool
        Whether to make limits symmetric around zero.
    include_zero : bool
        Whether to include zero in range.
    """
    if data is not None:
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data) & ~np.isinf(data)]
        
        if len(data) == 0:
            return
            
        dmin, dmax = data.min(), data.max()
    else:
        if axis == "x":
            dmin, dmax = ax.get_xlim()
        else:
            dmin, dmax = ax.get_ylim()
            
    # Include zero if requested
    if include_zero:
        dmin = min(dmin, 0)
        dmax = max(dmax, 0)
        
    # Add padding
    drange = dmax - dmin
    if drange == 0:
        drange = 1
    dmin -= pad * drange
    dmax += pad * drange
    
    # Make symmetric if requested
    if symmetric:
        abs_max = max(abs(dmin), abs(dmax))
        dmin, dmax = -abs_max, abs_max
        
    # Set limits
    if axis == "x":
        ax.set_xlim(dmin, dmax)
    else:
        ax.set_ylim(dmin, dmax)


def despine(
    ax: plt.Axes,
    top: bool = True,
    right: bool = True,
    left: bool = False,
    bottom: bool = False,
    offset: Optional[int] = None
) -> None:
    """Remove spines from axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify.
    top : bool
        Whether to remove top spine.
    right : bool
        Whether to remove right spine.
    left : bool
        Whether to remove left spine.
    bottom : bool
        Whether to remove bottom spine.
    offset : int, optional
        Points to move spines outward.
    """
    # Remove spines
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        
    # Offset remaining spines
    if offset is not None:
        for spine in ["top", "right", "left", "bottom"]:
            if ax.spines[spine].get_visible():
                ax.spines[spine].set_position(("outward", offset))
                
    # Remove ticks from hidden spines
    if top or bottom:
        ax.xaxis.set_ticks_position("bottom" if not bottom else "top")
    if left or right:
        ax.yaxis.set_ticks_position("left" if not left else "right")


def adjust_legend(
    ax: plt.Axes,
    loc: Optional[Union[str, int]] = None,
    bbox_to_anchor: Optional[Tuple[float, float]] = None,
    ncol: int = 1,
    fontsize: Optional[int] = None,
    title: Optional[str] = None,
    frameon: bool = True,
    fancybox: bool = True,
    shadow: bool = False,
    **kwargs
) -> Optional[Any]:
    """Adjust legend with common settings.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes with legend.
    loc : str or int, optional
        Legend location.
    bbox_to_anchor : tuple, optional
        Anchor point for legend.
    ncol : int
        Number of columns.
    fontsize : int, optional
        Font size.
    title : str, optional
        Legend title.
    frameon : bool
        Whether to show frame.
    fancybox : bool
        Whether to use fancy box.
    shadow : bool
        Whether to add shadow.
    **kwargs
        Additional arguments for legend.
        
    Returns
    -------
    legend : matplotlib.legend.Legend or None
        The legend object.
    """
    # Check if there are any labeled items
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
        
    # Default location
    if loc is None:
        loc = "best"
        
    # Create legend
    legend = ax.legend(
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        fontsize=fontsize,
        title=title,
        frameon=frameon,
        fancybox=fancybox,
        shadow=shadow,
        **kwargs
    )
    
    return legend


__all__ = [
    "create_figure",
    "save_plot",
    "annotate_bars",
    "add_reference_line",
    "format_axis_labels",
    "add_grid",
    "set_axis_limits",
    "despine",
    "adjust_legend"
]