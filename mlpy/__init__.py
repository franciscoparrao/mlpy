"""
MLPY - A Machine Learning Framework for Python inspired by mlr3

MLPY provides a unified, object-oriented interface for machine learning in Python,
inspired by the mlr3 framework for R. It features:

- Consistent API across different ML tasks
- Modular and extensible design
- First-class support for pipelines and AutoML
- Comprehensive evaluation and resampling strategies
- Integration with popular ML libraries (scikit-learn, etc.)
"""

__version__ = "0.1.0-dev"
__author__ = "MLPY Contributors"
__license__ = "MIT"

# Core imports
from .resample import resample, ResampleResult
from .benchmark import benchmark, BenchmarkResult

# Pipeline imports
from .pipelines import (
    PipeOp, PipeOpLearner, PipeOpScale, PipeOpImpute,
    PipeOpSelect, PipeOpEncode, GraphLearner, linear_pipeline
)

# Visualization imports (optional)
try:
    from .visualizations import (
        plot_resample_boxplot, plot_resample_roc, plot_resample_iterations,
        plot_benchmark_boxplot, plot_benchmark_heatmap, plot_benchmark_critical_difference,
        plot_tuning_performance, plot_tuning_parallel_coordinates,
        set_plot_theme
    )
    _has_viz = True
except ImportError:
    _has_viz = False

# Interpretability imports (optional)
try:
    from .interpretability import (
        Interpreter, InterpretationResult, FeatureImportance,
        SHAPInterpreter, LIMEInterpreter,
        plot_feature_importance, create_interpretation_report
    )
    _has_interpret = True
except ImportError:
    _has_interpret = False

# Persistence imports
try:
    from .persistence import (
        save_model, load_model, ModelSerializer,
        PickleSerializer, JoblibSerializer, JSONSerializer,
        ModelRegistry, export_model_package
    )
    _has_persistence = True
except ImportError:
    _has_persistence = False

# Validation system imports
from .validation import ValidatedTask, validate_task_data
from .validation.errors import MLPYValidationError, ErrorContext

# Enhanced serialization and lazy evaluation (optional)
try:
    from .serialization.robust_serializer import RobustSerializer, save_pipeline, load_pipeline
    _has_serialization = True
except ImportError:
    _has_serialization = False

try:
    from .lazy.lazy_evaluation import lazy, LazyResult, ComputationGraph
    _has_lazy = True
except ImportError:
    _has_lazy = False

__all__ = [
    "__version__", 
    "resample", "ResampleResult", 
    "benchmark", "BenchmarkResult",
    "PipeOp", "PipeOpLearner", "PipeOpScale", "PipeOpImpute",
    "PipeOpSelect", "PipeOpEncode", "GraphLearner", "linear_pipeline",
    # Validation system
    "ValidatedTask", "validate_task_data", "MLPYValidationError", "ErrorContext"
]

# Add visualization exports if available
if _has_viz:
    __all__.extend([
        "plot_resample_boxplot", "plot_resample_roc", "plot_resample_iterations",
        "plot_benchmark_boxplot", "plot_benchmark_heatmap", "plot_benchmark_critical_difference",
        "plot_tuning_performance", "plot_tuning_parallel_coordinates",
        "set_plot_theme"
    ])

# Add interpretability exports if available
if _has_interpret:
    __all__.extend([
        "Interpreter", "InterpretationResult", "FeatureImportance",
        "SHAPInterpreter", "LIMEInterpreter",
        "plot_feature_importance", "create_interpretation_report"
    ])

# Add persistence exports if available
if _has_persistence:
    __all__.extend([
        "save_model", "load_model", "ModelSerializer",
        "PickleSerializer", "JoblibSerializer", "JSONSerializer",
        "ModelRegistry", "export_model_package"
    ])

# Add serialization exports if available
if _has_serialization:
    __all__.extend([
        "RobustSerializer", "save_pipeline", "load_pipeline"
    ])

# Add lazy evaluation exports if available
if _has_lazy:
    __all__.extend([
        "lazy", "LazyResult", "ComputationGraph"
    ])