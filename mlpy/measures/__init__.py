"""Measures module for MLPY."""

from .base import Measure, MeasureClassif, MeasureRegr, MeasureSimple
from .classification import (
    MeasureClassifAccuracy,
    MeasureClassifCE,
    MeasureClassifAUC,
    MeasureClassifLogLoss,
    MeasureClassifF1,
    MeasureClassifPrecision,
    MeasureClassifRecall,
    MeasureClassifBalancedAccuracy,
    MeasureClassifMCC
)
from .regression import (
    MeasureRegrMSE,
    MeasureRegrRMSE, 
    MeasureRegrMAE,
    MeasureRegrMAPE,
    MeasureRegrR2,
    MeasureRegrMedianAE,
    MeasureRegrMSLE,
    MeasureRegrRMSLE,
    MeasureRegrMaxAbsoluteError,
    MeasureRegrBias
)
from .spatial import (
    MeasureSpatialAccuracy,
    MeasureSpatialAUC,
    MeasureSpatialF1,
    MeasureSpatialPrecision,
    MeasureSpatialRecall,
    MeasureSpatialMCC
)


def create_measure(measure_id: str, **kwargs) -> Measure:
    """Create a measure instance from a string identifier.
    
    Parameters
    ----------
    measure_id : str
        Measure identifier
    **kwargs
        Additional arguments for the measure
        
    Returns
    -------
    Measure
        Measure instance
    """
    measure_map = {
        'accuracy': MeasureClassifAccuracy,
        'auc': MeasureClassifAUC,
        'f1': MeasureClassifF1,
        'precision': MeasureClassifPrecision,
        'recall': MeasureClassifRecall,
        'logloss': MeasureClassifLogLoss,
        'ce': MeasureClassifCE,
        'balanced_accuracy': MeasureClassifBalancedAccuracy,
        'mcc': MeasureClassifMCC,
        # Spatial measures
        'spatial.acc': MeasureSpatialAccuracy,
        'spatial.auc': MeasureSpatialAUC,
        'spatial.f1': MeasureSpatialF1,
        'spatial.precision': MeasureSpatialPrecision,
        'spatial.recall': MeasureSpatialRecall,
        'spatial.mcc': MeasureSpatialMCC,
        # Regression measures
        'mse': MeasureRegrMSE,
        'rmse': MeasureRegrRMSE,
        'mae': MeasureRegrMAE,
        'mape': MeasureRegrMAPE,
        'r2': MeasureRegrR2,
        'median_ae': MeasureRegrMedianAE,
        'msle': MeasureRegrMSLE,
        'rmsle': MeasureRegrRMSLE,
        'max_ae': MeasureRegrMaxAbsoluteError,
        'bias': MeasureRegrBias
    }
    
    if measure_id in measure_map:
        return measure_map[measure_id](**kwargs)
    else:
        raise ValueError(f"Unknown measure: {measure_id}")


def list_measures() -> list:
    """List all available measures.
    
    Returns
    -------
    list
        List of measure identifiers
    """
    return [
        'accuracy', 'auc', 'f1', 'precision', 'recall', 'logloss', 'ce',
        'balanced_accuracy', 'mcc', 
        'spatial.acc', 'spatial.auc', 'spatial.f1', 'spatial.precision',
        'spatial.recall', 'spatial.mcc',
        'mse', 'rmse', 'mae', 'mape', 'r2',
        'median_ae', 'msle', 'rmsle', 'max_ae', 'bias'
    ]

__all__ = [
    # Base classes
    'Measure',
    'MeasureClassif',
    'MeasureRegr',
    'MeasureSimple',
    # Classification measures
    'MeasureClassifAccuracy',
    'MeasureClassifCE',
    'MeasureClassifAUC',
    'MeasureClassifLogLoss',
    'MeasureClassifF1',
    'MeasureClassifPrecision',
    'MeasureClassifRecall',
    'MeasureClassifBalancedAccuracy',
    'MeasureClassifMCC',
    # Spatial measures
    'MeasureSpatialAccuracy',
    'MeasureSpatialAUC',
    'MeasureSpatialF1',
    'MeasureSpatialPrecision',
    'MeasureSpatialRecall',
    'MeasureSpatialMCC',
    # Regression measures
    'MeasureRegrMSE',
    'MeasureRegrRMSE',
    'MeasureRegrMAE',
    'MeasureRegrMAPE',
    'MeasureRegrR2',
    'MeasureRegrMedianAE',
    'MeasureRegrMSLE',
    'MeasureRegrRMSLE',
    'MeasureRegrMaxAbsoluteError',
    'MeasureRegrBias',
    # Functions
    'create_measure',
    'list_measures'
]