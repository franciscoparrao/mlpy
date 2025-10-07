"""Regression measures for MLPY."""

import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics

from .base import MeasureRegr, MeasureSimple, register_measure
from ..predictions import PredictionRegr


@register_measure
class MeasureRegrMSE(MeasureRegr):
    """Mean Squared Error.
    
    Calculates the average of squared differences between predictions and true values.
    
    Parameters
    ----------
    squared : bool, default=True
        If True, returns MSE. If False, returns RMSE.
    """
    
    def __init__(self, squared: bool = True):
        super().__init__(
            id='regr.mse' if squared else 'regr.rmse',
            minimize=True,
            range=(0, np.inf),
            predict_type='response'
        )
        self.squared = squared
        
    def _score(self, prediction: PredictionRegr, task=None, **kwargs) -> float:
        """Calculate MSE or RMSE."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
            
        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        if not mask.any():
            return np.nan
            
        mse = sklearn_metrics.mean_squared_error(
            prediction.truth[mask],
            prediction.response[mask]
        )
        
        return mse if self.squared else np.sqrt(mse)


@register_measure
class MeasureRegrRMSE(MeasureRegrMSE):
    """Root Mean Squared Error."""
    
    def __init__(self):
        super().__init__(squared=False)


@register_measure
class MeasureRegrMAE(MeasureRegr):
    """Mean Absolute Error.
    
    Calculates the average of absolute differences between predictions and true values.
    """
    
    def __init__(self):
        super().__init__(
            id='regr.mae',
            minimize=True,
            range=(0, np.inf),
            predict_type='response'
        )
        
    def _score(self, prediction: PredictionRegr, task=None, **kwargs) -> float:
        """Calculate MAE."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
            
        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        if not mask.any():
            return np.nan
            
        return sklearn_metrics.mean_absolute_error(
            prediction.truth[mask],
            prediction.response[mask]
        )


@register_measure
class MeasureRegrMAPE(MeasureRegr):
    """Mean Absolute Percentage Error.
    
    Calculates the average of absolute percentage differences.
    Note: undefined for true values of 0.
    
    Parameters
    ----------
    eps : float, default=1e-8
        Small value to avoid division by zero.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__(
            id='regr.mape',
            minimize=True,
            range=(0, np.inf),
            predict_type='response'
        )
        self.eps = eps
        
    def _score(self, prediction: PredictionRegr, task=None, **kwargs) -> float:
        """Calculate MAPE."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
            
        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        # Also exclude values too close to zero
        mask = mask & (np.abs(prediction.truth) > self.eps)
        
        if not mask.any():
            return np.nan
            
        truth = prediction.truth[mask]
        response = prediction.response[mask]
        
        return np.mean(np.abs((truth - response) / truth)) * 100


@register_measure
class MeasureRegrR2(MeasureRegr):
    """Coefficient of determination (R²).
    
    Proportion of variance in the dependent variable predictable from the independent variable(s).
    """
    
    def __init__(self):
        super().__init__(
            id='regr.rsq',
            minimize=False,
            range=(-np.inf, 1),
            predict_type='response'
        )
        
    def _score(self, prediction: PredictionRegr, task=None, **kwargs) -> float:
        """Calculate R² score."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
            
        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        if not mask.any():
            return np.nan
            
        # Need at least 2 samples for R²
        if mask.sum() < 2:
            return np.nan
            
        return sklearn_metrics.r2_score(
            prediction.truth[mask],
            prediction.response[mask]
        )


@register_measure
class MeasureRegrMedianAE(MeasureRegr):
    """Median Absolute Error.
    
    Robust to outliers compared to MAE.
    """
    
    def __init__(self):
        super().__init__(
            id='regr.medae',
            minimize=True,
            range=(0, np.inf),
            predict_type='response'
        )
        
    def _score(self, prediction: PredictionRegr, task=None, **kwargs) -> float:
        """Calculate Median AE."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
            
        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        if not mask.any():
            return np.nan
            
        return sklearn_metrics.median_absolute_error(
            prediction.truth[mask],
            prediction.response[mask]
        )


@register_measure
class MeasureRegrMSLE(MeasureRegr):
    """Mean Squared Logarithmic Error.
    
    Useful when targets have exponential growth.
    Note: requires non-negative values.
    
    Parameters
    ----------
    squared : bool, default=True
        If True, returns MSLE. If False, returns RMSLE.
    """
    
    def __init__(self, squared: bool = True):
        super().__init__(
            id='regr.msle' if squared else 'regr.rmsle',
            minimize=True,
            range=(0, np.inf),
            predict_type='response'
        )
        self.squared = squared
        
    def _score(self, prediction: PredictionRegr, task=None, **kwargs) -> float:
        """Calculate MSLE or RMSLE."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
            
        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        # MSLE requires non-negative values
        mask = mask & (prediction.truth >= 0) & (prediction.response >= 0)
        
        if not mask.any():
            return np.nan
            
        try:
            msle = sklearn_metrics.mean_squared_log_error(
                prediction.truth[mask],
                prediction.response[mask]
            )
            return msle if self.squared else np.sqrt(msle)
        except ValueError:
            # Can happen if values are slightly negative due to numerical errors
            return np.nan


@register_measure
class MeasureRegrRMSLE(MeasureRegrMSLE):
    """Root Mean Squared Logarithmic Error."""
    
    def __init__(self):
        super().__init__(squared=False)


# Additional useful measures
register_measure(lambda: MeasureSimple(
    'regr.maxae',
    score_func=lambda y_true, y_pred: np.max(np.abs(y_true - y_pred)),
    task_type='regr',
    minimize=True,
    range=(0, np.inf),
    predict_type='response'
))()

register_measure(lambda: MeasureSimple(
    'regr.bias',
    score_func=lambda y_true, y_pred: np.mean(y_pred - y_true),
    task_type='regr',
    minimize=True,  # We want bias close to 0
    range=(-np.inf, np.inf),
    predict_type='response'
))()


# Aliases
from ..utils.registry import mlpy_measures as _mlpy_measures  
MeasureRegrMaxAbsoluteError = lambda: _mlpy_measures['regr.maxae']
MeasureRegrBias = lambda: _mlpy_measures['regr.bias']