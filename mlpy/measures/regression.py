"""Regression measures for MLPY."""

import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics

from .base import MeasureRegr, MeasureSimple, register_measure
from ..predictions import PredictionRegr


@register_measure
class MeasureRegrMSE(MeasureRegr):
    """Mean Squared Error.
    
    Wraps sklearn's mean_squared_error with MLPY's unified interface. Average
    squared difference between predictions and truth; lower is better.
    
    Parameters
    ----------
    squared : bool, default=True
        If True, returns MSE. If False, returns RMSE.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.regression import MeasureRegrMSE
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5,  0.0, 2, 8])
    >>> round(MeasureRegrMSE().score(y_true, y_pred), 3)
    0.375

    Notes
    -----
    Requires response predictions (numeric values).

    See Also
    --------
    MeasureRegrRMSE : Root mean squared error
    MeasureRegrMAE : Mean absolute error
    MeasureRegrR2 : Coefficient of determination
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
    """Root Mean Squared Error.

    Square root of MSE; preserves units of the target.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.regression import MeasureRegrRMSE
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5,  0.0, 2, 8])
    >>> float(round(MeasureRegrRMSE().score(y_true, y_pred), 3))
    0.612

    Notes
    -----
    Equivalent to ``sqrt(MSE)``.
    """
    
    def __init__(self):
        super().__init__(squared=False)


@register_measure
class MeasureRegrMAE(MeasureRegr):
    """Mean Absolute Error.
    
    Wraps sklearn's mean_absolute_error. Average absolute difference between
    predictions and truth; robust to a few large errors compared to MSE.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.regression import MeasureRegrMAE
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5,  0.0, 2, 8])
    >>> MeasureRegrMAE().score(y_true, y_pred)
    0.5

    Notes
    -----
    Requires response predictions (numeric values).

    See Also
    --------
    MeasureRegrMedianAE : Median absolute error (more robust)
    MeasureRegrMSE : Mean squared error
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
    
    Wraps the mean absolute percentage error. Average absolute percentage
    difference; undefined at true=0 (entries near 0 are excluded by eps).
    
    Parameters
    ----------
    eps : float, default=1e-8
        Small value to avoid division by zero.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.regression import MeasureRegrMAPE
    >>> y_true = np.array([1.0, 1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 0.9, 2.2, 2.7])
    >>> float(round(MeasureRegrMAPE().score(y_true, y_pred), 1))
    10.0

    Notes
    -----
    Values with |truth| <= eps are ignored to avoid division by zero.

    See Also
    --------
    MeasureRegrMAE : Absolute error in original units
    MeasureRegrMSLE : Squared log error for relative differences
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
    
    Proportion of variance in the target explained by the model.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.regression import MeasureRegrR2
    >>> y_true = np.array([1, 2, 3, 4])
    >>> y_pred = np.array([0.6, 1.9, 3.1, 4.2])
    >>> round(MeasureRegrR2().score(y_true, y_pred), 3)
    0.956

    Notes
    -----
    Requires at least 2 valid samples; can be negative for poor fits.
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
    
    Wraps sklearn's median_absolute_error. Median absolute difference; more
    robust to outliers than MAE.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.regression import MeasureRegrMedianAE
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5,  0.0, 2, 8])
    >>> MeasureRegrMedianAE().score(y_true, y_pred)
    0.5
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
    
    Useful when relative differences matter or targets grow exponentially.
    Requires non-negative values (entries < 0 are ignored).
    
    Parameters
    ----------
    squared : bool, default=True
        If True, returns MSLE. If False, returns RMSLE.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.regression import MeasureRegrMSLE
    >>> y_true = np.array([1., 10.])
    >>> y_pred = np.array([2., 20.])
    >>> round(MeasureRegrMSLE().score(y_true, y_pred), 3)
    0.291

    Notes
    -----
    Negative values are excluded from the calculation.

    See Also
    --------
    MeasureRegrRMSLE : Root mean squared logarithmic error
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
    """Root Mean Squared Logarithmic Error.

    Square root of MSLE; preserves interpretability in log scale.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.regression import MeasureRegrRMSLE
    >>> y_true = np.array([1., 10.])
    >>> y_pred = np.array([2., 20.])
    >>> float(round(MeasureRegrRMSLE().score(y_true, y_pred), 2))
    0.54
    """
    
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