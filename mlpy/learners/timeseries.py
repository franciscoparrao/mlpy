"""
Time series learners for MLPY.

This module provides learners specialized for time series forecasting,
including ARIMA, Prophet, Exponential Smoothing, and other temporal models.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from copy import deepcopy
import warnings

from .base import Learner
from ..tasks import Task, TaskTimeSeries, TaskForecasting
from ..predictions import PredictionRegr
from ..utils.registry import mlpy_learners


class LearnerTimeSeriesBase(Learner):
    """Base class for time series learners.
    
    This class provides common functionality for all time series learners
    including validation, data preparation, and prediction formatting.
    """
    
    def __init__(
        self,
        id: str,
        horizon: int = 1,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.horizon = horizon
        self._fitted_model = None
        self._time_col = None
        self._freq = None
        
    @property
    def task_type(self) -> str:
        """This learner handles time series tasks."""
        return "timeseries"
        
    def _validate_task(self, task: Task):
        """Validate that task is appropriate for time series learning."""
        # Check for time series capabilities rather than exact type
        # This allows TaskForecasting and other TaskTimeSeries subclasses
        if not (hasattr(task, 'time_col') and hasattr(task, 'freq')):
            raise TypeError(f"Expected time series task with time_col attribute, got {type(task)}")
            
    def _prepare_data(self, task: Task, row_ids: Optional[List[int]] = None):
        """Prepare time series data for modeling."""
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        # Get data and sort by time column
        data = task.data(rows=row_ids)
        if task.time_col in data.columns:
            data = data.sort_values(task.time_col).reset_index(drop=True)
            
        # Extract time series
        y = task.truth(rows=row_ids)
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store metadata
        self._time_col = task.time_col
        self._freq = getattr(task, 'freq', None)
        
        return data, y
        
    def _create_prediction(
        self, 
        task: Task, 
        row_ids: List[int], 
        predictions: np.ndarray,
        truth: Optional[np.ndarray] = None
    ) -> PredictionRegr:
        """Create prediction object for time series results."""
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth if truth is not None else task.truth(rows=row_ids),
            response=predictions
        )


@mlpy_learners.register('timeseries.arima')
class LearnerARIMA(LearnerTimeSeriesBase):
    """ARIMA learner for time series forecasting.
    
    This learner wraps ARIMA models from statsmodels for integration
    with the MLPY framework.
    
    Parameters
    ----------
    order : tuple, default=(1,1,1)
        The (p,d,q) order of the ARIMA model.
    seasonal_order : tuple, optional
        The (P,D,Q,s) seasonal order for SARIMA models.
    horizon : int, default=1
        Number of steps to forecast ahead.
    trend : str, optional
        Trend component ('n', 'c', 't', 'ct').
    
    Examples
    --------
    >>> from mlpy.learners import LearnerARIMA
    >>> from mlpy.tasks import TaskForecasting
    >>> 
    >>> learner = LearnerARIMA(order=(2,1,1), horizon=5)
    >>> learner.train(task)
    >>> predictions = learner.predict(task)
    """
    
    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: Optional[tuple] = None,
        horizon: int = 1,
        trend: Optional[str] = None,
        id: Optional[str] = None,
        **kwargs
    ):
        if id is None:
            id = f"arima_{order[0]}_{order[1]}_{order[2]}"
            if seasonal_order:
                id += f"_s{seasonal_order[0]}_{seasonal_order[1]}_{seasonal_order[2]}_{seasonal_order[3]}"
                
        super().__init__(id=id, horizon=horizon, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        
        # Set properties
        self.properties.add('autoregressive')
        self.properties.add('time_series')
        self.packages = ['statsmodels']
        
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "LearnerARIMA":
        """Train ARIMA model."""
        self._validate_task(task)
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError(
                "statsmodels is required for ARIMA models. "
                "Install with: pip install statsmodels"
            )
            
        # Prepare data
        data, y = self._prepare_data(task, row_ids)
        
        # Create and fit model
        try:
            if self.seasonal_order:
                model = SARIMAX(
                    y,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    trend=self.trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = ARIMA(
                    y,
                    order=self.order,
                    trend=self.trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
            self._fitted_model = model.fit(disp=False)
            self._model = self._fitted_model
            
        except Exception as e:
            raise RuntimeError(f"ARIMA training failed: {e}") from e
            
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Make ARIMA predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        self._validate_task(task)
        
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        try:
            # Generate forecasts
            forecast = self._fitted_model.forecast(steps=self.horizon)
            
            # Handle single vs multiple horizons
            if self.horizon == 1:
                predictions = np.array([forecast])
            else:
                predictions = np.array(forecast)
                
            # Extend predictions to match row_ids if needed
            if len(predictions) < len(row_ids):
                # Repeat last prediction
                predictions = np.pad(
                    predictions, 
                    (0, len(row_ids) - len(predictions)),
                    mode='edge'
                )
            elif len(predictions) > len(row_ids):
                predictions = predictions[:len(row_ids)]
                
            return self._create_prediction(task, row_ids, predictions)
            
        except Exception as e:
            raise RuntimeError(f"ARIMA prediction failed: {e}") from e


@mlpy_learners.register('timeseries.prophet')            
class LearnerProphet(LearnerTimeSeriesBase):
    """Prophet learner for time series forecasting.
    
    This learner wraps Facebook Prophet for robust time series forecasting
    with trend and seasonality detection.
    
    Parameters
    ----------
    horizon : int, default=1
        Number of steps to forecast ahead.
    yearly_seasonality : bool or int, default='auto'
        Fit yearly seasonality.
    weekly_seasonality : bool or int, default='auto'  
        Fit weekly seasonality.
    daily_seasonality : bool or int, default='auto'
        Fit daily seasonality.
    growth : str, default='linear'
        Growth model ('linear' or 'logistic').
    changepoint_prior_scale : float, default=0.05
        Flexibility of trend changes.
        
    Examples
    --------
    >>> from mlpy.learners import LearnerProphet
    >>> 
    >>> learner = LearnerProphet(
    ...     horizon=30,
    ...     yearly_seasonality=True,
    ...     growth='linear'
    ... )
    >>> learner.train(task)
    >>> predictions = learner.predict(task)
    """
    
    def __init__(
        self,
        horizon: int = 1,
        yearly_seasonality: Union[bool, str] = 'auto',
        weekly_seasonality: Union[bool, str] = 'auto',
        daily_seasonality: Union[bool, str] = 'auto',
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        id: Optional[str] = None,
        **kwargs
    ):
        if id is None:
            id = f"prophet_h{horizon}"
            
        super().__init__(id=id, horizon=horizon, **kwargs)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality  
        self.daily_seasonality = daily_seasonality
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        
        # Set properties
        self.properties.add('seasonal')
        self.properties.add('time_series')
        self.properties.add('trend')
        self.packages = ['prophet']
        
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "LearnerProphet":
        """Train Prophet model."""
        self._validate_task(task)
        
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "prophet is required for Prophet models. "
                "Install with: pip install prophet"
            )
            
        # Prepare data
        data, y = self._prepare_data(task, row_ids)
        
        # Create Prophet dataframe format
        if self._time_col and self._time_col in data.columns:
            prophet_df = pd.DataFrame({
                'ds': data[self._time_col],
                'y': y
            })
        else:
            # Create artificial time index
            prophet_df = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(y), freq='D'),
                'y': y
            })
            
        try:
            # Create and fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model = Prophet(
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    growth=self.growth,
                    changepoint_prior_scale=self.changepoint_prior_scale
                )
                
                self._fitted_model = model.fit(prophet_df)
                self._model = self._fitted_model
                
        except Exception as e:
            raise RuntimeError(f"Prophet training failed: {e}") from e
            
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Make Prophet predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        self._validate_task(task)
        
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        try:
            # Create future dataframe
            future = self._fitted_model.make_future_dataframe(periods=self.horizon)
            
            # Generate forecasts
            forecast = self._fitted_model.predict(future)
            
            # Extract predictions (take last `horizon` values)
            predictions = forecast['yhat'].tail(self.horizon).values
            
            # Extend predictions to match row_ids if needed
            if len(predictions) < len(row_ids):
                predictions = np.pad(
                    predictions,
                    (0, len(row_ids) - len(predictions)),
                    mode='edge'
                )
            elif len(predictions) > len(row_ids):
                predictions = predictions[:len(row_ids)]
                
            return self._create_prediction(task, row_ids, predictions)
            
        except Exception as e:
            raise RuntimeError(f"Prophet prediction failed: {e}") from e


@mlpy_learners.register('timeseries.exponential_smoothing')
class LearnerExponentialSmoothing(LearnerTimeSeriesBase):
    """Exponential Smoothing learner for time series forecasting.
    
    This learner implements Holt-Winters exponential smoothing methods
    for time series with trend and seasonality.
    
    Parameters
    ----------
    trend : str, optional
        Type of trend component ('add', 'mul', None).
    seasonal : str, optional  
        Type of seasonal component ('add', 'mul', None).
    seasonal_periods : int, optional
        Length of the seasonal cycle.
    horizon : int, default=1
        Number of steps to forecast ahead.
    
    Examples
    --------
    >>> from mlpy.learners import LearnerExponentialSmoothing
    >>> 
    >>> learner = LearnerExponentialSmoothing(
    ...     trend='add',
    ...     seasonal='add', 
    ...     seasonal_periods=12,
    ...     horizon=6
    ... )
    >>> learner.train(task)
    >>> predictions = learner.predict(task)
    """
    
    def __init__(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        horizon: int = 1,
        id: Optional[str] = None,
        **kwargs
    ):
        if id is None:
            id = f"exp_smooth_{trend or 'none'}_{seasonal or 'none'}"
            
        super().__init__(id=id, horizon=horizon, **kwargs)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        
        # Set properties
        self.properties.add('exponential_smoothing')
        self.properties.add('time_series')
        if trend:
            self.properties.add('trend')
        if seasonal:
            self.properties.add('seasonal')
        self.packages = ['statsmodels']
        
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "LearnerExponentialSmoothing":
        """Train Exponential Smoothing model."""
        self._validate_task(task)
        
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            raise ImportError(
                "statsmodels is required for Exponential Smoothing models. "
                "Install with: pip install statsmodels"
            )
            
        # Prepare data
        data, y = self._prepare_data(task, row_ids)
        
        try:
            # Create and fit model
            model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )
            
            self._fitted_model = model.fit()
            self._model = self._fitted_model
            
        except Exception as e:
            raise RuntimeError(f"Exponential Smoothing training failed: {e}") from e
            
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Make Exponential Smoothing predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        self._validate_task(task)
        
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        try:
            # Generate forecasts
            forecast = self._fitted_model.forecast(steps=self.horizon)
            
            # Handle single vs multiple horizons
            if self.horizon == 1:
                predictions = np.array([forecast])
            else:
                predictions = np.array(forecast)
                
            # Extend predictions to match row_ids if needed
            if len(predictions) < len(row_ids):
                predictions = np.pad(
                    predictions,
                    (0, len(row_ids) - len(predictions)), 
                    mode='edge'
                )
            elif len(predictions) > len(row_ids):
                predictions = predictions[:len(row_ids)]
                
            return self._create_prediction(task, row_ids, predictions)
            
        except Exception as e:
            raise RuntimeError(f"Exponential Smoothing prediction failed: {e}") from e


# Convenience functions
def learner_arima(order=(1,1,1), **kwargs) -> LearnerARIMA:
    """Create an ARIMA learner with specified order."""
    return LearnerARIMA(order=order, **kwargs)

def learner_sarima(order=(1,1,1), seasonal_order=(1,1,1,12), **kwargs) -> LearnerARIMA:
    """Create a seasonal ARIMA learner."""  
    return LearnerARIMA(order=order, seasonal_order=seasonal_order, **kwargs)

def learner_prophet(horizon=1, **kwargs) -> LearnerProphet:
    """Create a Prophet learner with specified horizon."""
    return LearnerProphet(horizon=horizon, **kwargs)

def learner_exponential_smoothing(trend=None, seasonal=None, **kwargs) -> LearnerExponentialSmoothing:
    """Create an Exponential Smoothing learner."""
    return LearnerExponentialSmoothing(trend=trend, seasonal=seasonal, **kwargs)


__all__ = [
    'LearnerTimeSeriesBase',
    'LearnerARIMA',
    'LearnerProphet', 
    'LearnerExponentialSmoothing',
    'learner_arima',
    'learner_sarima',
    'learner_prophet',
    'learner_exponential_smoothing'
]