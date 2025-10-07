"""
Feature filtering module for MLPY.

Provides comprehensive feature selection methods inspired by mlr3filters.
"""

from .base import Filter, FilterResult

# Import mutual information based filters
from .mutual_information import (
    MRMR,
    CMIM,
    JMI,
    JMIM,
    MIM,
    InformationGain
)

# Import statistical filters
from .statistical import (
    Relief,
    ReliefF,
    DISR,
    ANOVA,
    VarianceThreshold
)

# Import ensemble ranking
from .ensemble_ranking import (
    EnsembleRanking,
    CumulativeRanking,
    quick_feature_selection
)

# Try to import existing filters if they exist
try:
    from .univariate import (
        FilterANOVA,
        FilterFRegression,
        FilterMutualInformation,
        FilterChiSquared,
        FilterCorrelation,
        FilterVariance
    )
    UNIVARIATE_AVAILABLE = True
except ImportError:
    UNIVARIATE_AVAILABLE = False

try:
    from .multivariate import (
        FilterImportance,
        FilterRFE,
        FilterMRMR,
        FilterRelief
    )
    MULTIVARIATE_AVAILABLE = True
except ImportError:
    MULTIVARIATE_AVAILABLE = False

try:
    from .ensemble import (
        FilterEnsemble,
        FilterStability,
        FilterAutoSelect
    )
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from .information_theory import (
        FilterInformationGain,
        FilterInformationGainRatio,
        FilterSymmetricalUncertainty,
        FilterJMIM
    )
    INFO_THEORY_AVAILABLE = True
except ImportError:
    INFO_THEORY_AVAILABLE = False

# Convenience functions
def create_filter(method: str, **kwargs) -> Filter:
    """Create a filter instance.
    
    Parameters
    ----------
    method : str
        Filter method name.
    **kwargs
        Arguments for the filter.
        
    Returns
    -------
    Filter
        Filter instance.
        
    Examples
    --------
    >>> from mlpy.filters import create_filter
    >>> 
    >>> # Create ANOVA filter
    >>> anova = create_filter('anova')
    >>> 
    >>> # Create MRMR filter
    >>> mrmr = create_filter('mrmr')
    >>> 
    >>> # Create Relief filter
    >>> relief = create_filter('relief')
    """
    from ..utils.registry import mlpy_filters
    
    # Map method names to classes
    method_map = {
        'mrmr': MRMR,
        'cmim': CMIM,
        'jmi': JMI,
        'jmim': JMIM,
        'mim': MIM,
        'information_gain': InformationGain,
        'relief': Relief,
        'relieff': ReliefF,
        'disr': DISR,
        'anova': ANOVA,
        'variance': VarianceThreshold,
        'cumulative': CumulativeRanking
    }
    
    if method in method_map:
        return method_map[method](**kwargs)
    elif method in mlpy_filters:
        return mlpy_filters[method](**kwargs)
    else:
        raise ValueError(f"Unknown filter method: {method}")


def list_filters() -> list:
    """List all available filters.
    
    Returns
    -------
    list
        List of filter IDs.
    """
    from ..utils.registry import mlpy_filters
    
    builtin_filters = [
        'mrmr', 'cmim', 'jmi', 'jmim', 'mim', 'information_gain',
        'relief', 'relieff', 'disr', 'anova', 'variance', 'cumulative'
    ]
    
    registry_filters = mlpy_filters.get_keys()
    
    return sorted(set(builtin_filters + registry_filters))


def filter_features(
    task,
    method: str = 'auto',
    k: int = None,
    threshold: float = None,
    percentile: float = None,
    **kwargs
) -> list:
    """Quick feature filtering.
    
    Parameters
    ----------
    task : Task
        The task to filter.
    method : str, default='auto'
        Filter method to use.
    k : int, optional
        Number of features to select.
    threshold : float, optional
        Score threshold for selection.
    percentile : float, optional
        Percentile threshold for selection.
    **kwargs
        Additional arguments for the filter.
        
    Returns
    -------
    list
        Selected feature names.
        
    Examples
    --------
    >>> from mlpy.filters import filter_features
    >>> 
    >>> # Select top 10 features using ANOVA
    >>> features = filter_features(task, method='anova', k=10)
    >>> 
    >>> # Select features with variance > 0.1
    >>> features = filter_features(task, method='variance', threshold=0.1)
    >>> 
    >>> # Auto-select best features
    >>> features = filter_features(task, method='auto', k=20)
    """
    filter_obj = create_filter(method, **kwargs)
    return filter_obj.filter(task, k=k, threshold=threshold, percentile=percentile)


# Export main classes and functions
__all__ = [
    # Base classes
    'Filter',
    'FilterResult',
    
    # Mutual Information filters
    'MRMR',
    'CMIM',
    'JMI', 
    'JMIM',
    'MIM',
    'InformationGain',
    
    # Statistical filters
    'Relief',
    'ReliefF',
    'DISR',
    'ANOVA',
    'VarianceThreshold',
    
    # Ensemble ranking
    'EnsembleRanking',
    'CumulativeRanking',
    'quick_feature_selection',
    
    # Functions
    'create_filter',
    'list_filters',
    'filter_features'
]

# Add existing filters if available
if UNIVARIATE_AVAILABLE:
    __all__.extend([
        'FilterANOVA',
        'FilterFRegression',
        'FilterMutualInformation',
        'FilterChiSquared',
        'FilterCorrelation',
        'FilterVariance'
    ])

if MULTIVARIATE_AVAILABLE:
    __all__.extend([
        'FilterImportance',
        'FilterRFE',
        'FilterMRMR',
        'FilterRelief'
    ])

if ENSEMBLE_AVAILABLE:
    __all__.extend([
        'FilterEnsemble',
        'FilterStability',
        'FilterAutoSelect'
    ])

if INFO_THEORY_AVAILABLE:
    __all__.extend([
        'FilterInformationGain',
        'FilterInformationGainRatio',
        'FilterSymmetricalUncertainty',
        'FilterJMIM'
    ])