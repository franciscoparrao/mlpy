"""Visualization module for MLPY.

This module provides plotting functionality for MLPY objects including:
- ResampleResult plots
- BenchmarkResult plots
- TuneResult plots
- Task data visualization
- Measure performance plots
"""

from .base import PlotBase, plot_theme, set_plot_theme
from .resample import plot_resample_boxplot, plot_resample_roc, plot_resample_iterations
from .benchmark import plot_benchmark_boxplot, plot_benchmark_heatmap, plot_benchmark_critical_difference
from .tuning import plot_tuning_performance, plot_tuning_parallel_coordinates
from .utils import save_plot, create_figure

__all__ = [
    # Base
    "PlotBase",
    "plot_theme", 
    "set_plot_theme",
    
    # Resample plots
    "plot_resample_boxplot",
    "plot_resample_roc", 
    "plot_resample_iterations",
    
    # Benchmark plots
    "plot_benchmark_boxplot",
    "plot_benchmark_heatmap",
    "plot_benchmark_critical_difference",
    
    # Tuning plots
    "plot_tuning_performance",
    "plot_tuning_parallel_coordinates",
    
    # Utils
    "save_plot",
    "create_figure"
]