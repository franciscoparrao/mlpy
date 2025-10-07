"""
Tests for time series learners.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from mlpy.tasks import TaskTimeSeries, TaskForecasting
from mlpy.predictions import PredictionRegr


class TestLearnerTimeSeriesBase:
    """Test base time series learner functionality."""
    
    @pytest.fixture
    def simple_ts_task(self):
        """Create a simple time series task."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        trend = np.linspace(0, 10, 100)
        seasonal = np.sin(np.linspace(0, 4*np.pi, 100)) * 2
        noise = np.random.normal(0, 0.5, 100)
        values = trend + seasonal + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'feature1': np.random.randn(100)
        })
        
        return TaskForecasting(df, time_col='date', target='value')
    
    @pytest.fixture
    def forecasting_task(self):
        """Create a forecasting task."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        values = np.cumsum(np.random.randn(50)) + 100
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        return TaskForecasting(df, time_col='date', target='value')

    def test_timeseries_base_validation(self, simple_ts_task):
        """Test base class task validation."""
        from mlpy.learners.timeseries import LearnerTimeSeriesBase
        
        # Mock implementation
        class MockTSLearner(LearnerTimeSeriesBase):
            def train(self, task, row_ids=None):
                self._validate_task(task)
                return self
                
            def predict(self, task, row_ids=None):
                return self._create_prediction(
                    task, 
                    [0, 1, 2], 
                    np.array([1.0, 2.0, 3.0])
                )
        
        learner = MockTSLearner(id="test", horizon=3)
        assert learner.task_type == "timeseries"
        
        # Should not raise for correct task type
        learner._validate_task(simple_ts_task)
        
        # Should raise for incorrect task type
        from mlpy.tasks import TaskRegr
        X = np.random.randn(10, 3)
        df = pd.DataFrame(X, columns=['a', 'b', 'c'])
        df['target'] = np.random.randn(10)
        wrong_task = TaskRegr(df, target='target')
        
        with pytest.raises(TypeError):
            learner._validate_task(wrong_task)


class TestLearnerARIMA:
    """Test ARIMA learner."""
    
    @pytest.fixture
    def arima_task(self):
        """Create task suitable for ARIMA."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Generate AR(1) process
        y = np.zeros(100)
        y[0] = np.random.normal()
        for i in range(1, 100):
            y[i] = 0.7 * y[i-1] + np.random.normal(0, 0.5)
        
        df = pd.DataFrame({
            'date': dates,
            'value': y
        })
        
        return TaskForecasting(df, time_col='date', target='value')
    
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_arima_basic(self, mock_arima, arima_task):
        """Test basic ARIMA functionality with mocking."""
        from mlpy.learners.timeseries import LearnerARIMA
        
        # Mock the fitted model
        mock_fitted = Mock()
        mock_fitted.forecast.return_value = np.array([1.5])
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_fitted
        mock_arima.return_value = mock_model
        
        # Create and train learner
        learner = LearnerARIMA(order=(1, 0, 0), horizon=1)
        assert learner.order == (1, 0, 0)
        assert learner.horizon == 1
        assert 'autoregressive' in learner.properties
        
        # Train
        trained = learner.train(arima_task)
        assert trained.is_trained
        
        # Predict
        pred = trained.predict(arima_task)
        assert isinstance(pred, PredictionRegr)
        assert len(pred.response) == arima_task.nrow
    
    @patch('statsmodels.tsa.statespace.sarimax.SARIMAX')
    def test_sarima_seasonal(self, mock_sarimax, arima_task):
        """Test seasonal ARIMA."""
        from mlpy.learners.timeseries import LearnerARIMA
        
        # Mock the fitted model
        mock_fitted = Mock()
        mock_fitted.forecast.return_value = np.array([2.1, 2.2, 2.3])
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_fitted
        mock_sarimax.return_value = mock_model
        
        # Create seasonal learner
        learner = LearnerARIMA(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            horizon=3
        )
        
        # Train and predict
        trained = learner.train(arima_task)
        pred = trained.predict(arima_task)
        
        assert isinstance(pred, PredictionRegr)
        assert len(pred.response) == arima_task.nrow
    
    def test_arima_convenience_functions(self):
        """Test convenience functions."""
        from mlpy.learners.timeseries import learner_arima, learner_sarima
        
        # Test arima convenience
        arima = learner_arima(order=(2, 1, 1), horizon=5)
        assert arima.order == (2, 1, 1)
        assert arima.horizon == 5
        
        # Test sarima convenience  
        sarima = learner_sarima(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        assert sarima.order == (1, 1, 1)
        assert sarima.seasonal_order == (1, 1, 1, 12)
    
    def test_arima_missing_dependency(self, arima_task):
        """Test error when statsmodels not available."""
        from mlpy.learners.timeseries import LearnerARIMA
        
        with patch('statsmodels.tsa.arima.model.ARIMA', side_effect=ImportError):
            learner = LearnerARIMA()
            with pytest.raises(ImportError, match="statsmodels is required"):
                learner.train(arima_task)


class TestLearnerProphet:
    """Test Prophet learner."""
    
    @pytest.fixture
    def prophet_task(self):
        """Create task suitable for Prophet."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        
        # Generate time series with trend and seasonality
        t = np.arange(365)
        trend = 0.01 * t
        yearly = np.sin(2 * np.pi * t / 365.25) * 3
        weekly = np.sin(2 * np.pi * t / 7) * 1
        noise = np.random.normal(0, 0.5, 365)
        
        values = 100 + trend + yearly + weekly + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        return TaskForecasting(df, time_col='date', target='value')
    
    @patch('prophet.Prophet')
    def test_prophet_basic(self, mock_prophet_class, prophet_task):
        """Test basic Prophet functionality."""
        from mlpy.learners.timeseries import LearnerProphet
        
        # Mock Prophet model and forecast
        mock_forecast_df = pd.DataFrame({
            'yhat': [101.5, 102.1, 102.8, 103.2, 103.9]
        })
        
        mock_fitted = Mock()
        mock_fitted.make_future_dataframe.return_value = pd.DataFrame()
        mock_fitted.predict.return_value = mock_forecast_df
        
        mock_prophet = Mock()
        mock_prophet.fit.return_value = mock_fitted
        mock_prophet_class.return_value = mock_prophet
        
        # Create and train learner
        learner = LearnerProphet(
            horizon=5,
            yearly_seasonality=True,
            growth='linear'
        )
        
        assert learner.horizon == 5
        assert learner.yearly_seasonality is True
        assert learner.growth == 'linear'
        assert 'seasonal' in learner.properties
        assert 'trend' in learner.properties
        
        # Train
        trained = learner.train(prophet_task)
        assert trained.is_trained
        
        # Predict
        pred = trained.predict(prophet_task)
        assert isinstance(pred, PredictionRegr)
        assert len(pred.response) == prophet_task.nrow
    
    def test_prophet_convenience_function(self):
        """Test Prophet convenience function."""
        from mlpy.learners.timeseries import learner_prophet
        
        prophet = learner_prophet(horizon=14, yearly_seasonality=False)
        assert prophet.horizon == 14
        assert prophet.yearly_seasonality is False
    
    def test_prophet_missing_dependency(self, prophet_task):
        """Test error when prophet not available.""" 
        from mlpy.learners.timeseries import LearnerProphet
        
        with patch('prophet.Prophet', side_effect=ImportError):
            learner = LearnerProphet()
            with pytest.raises(ImportError, match="prophet is required"):
                learner.train(prophet_task)


class TestLearnerExponentialSmoothing:
    """Test Exponential Smoothing learner."""
    
    @pytest.fixture
    def exp_smooth_task(self):
        """Create task suitable for exponential smoothing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=120, freq='ME')
        
        # Generate monthly data with trend and seasonality
        t = np.arange(120)
        trend = 100 + 0.5 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 2, 120)
        
        values = trend + seasonal + noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        return TaskForecasting(df, time_col='date', target='value')
    
    @patch('statsmodels.tsa.holtwinters.ExponentialSmoothing')
    def test_exponential_smoothing_basic(self, mock_exp_smooth, exp_smooth_task):
        """Test basic Exponential Smoothing functionality."""
        from mlpy.learners.timeseries import LearnerExponentialSmoothing
        
        # Mock fitted model
        mock_fitted = Mock()
        mock_fitted.forecast.return_value = np.array([105.2, 106.1, 106.8])
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_fitted
        mock_exp_smooth.return_value = mock_model
        
        # Create learner
        learner = LearnerExponentialSmoothing(
            trend='add',
            seasonal='add',
            seasonal_periods=12,
            horizon=3
        )
        
        assert learner.trend == 'add'
        assert learner.seasonal == 'add'
        assert learner.seasonal_periods == 12
        assert learner.horizon == 3
        assert 'trend' in learner.properties
        assert 'seasonal' in learner.properties
        
        # Train
        trained = learner.train(exp_smooth_task)
        assert trained.is_trained
        
        # Predict
        pred = trained.predict(exp_smooth_task)
        assert isinstance(pred, PredictionRegr)
        assert len(pred.response) == exp_smooth_task.nrow
    
    @patch('statsmodels.tsa.holtwinters.ExponentialSmoothing')
    def test_exponential_smoothing_simple(self, mock_exp_smooth, exp_smooth_task):
        """Test simple exponential smoothing (no trend/seasonal)."""
        from mlpy.learners.timeseries import LearnerExponentialSmoothing
        
        # Mock fitted model
        mock_fitted = Mock()
        mock_fitted.forecast.return_value = 104.5
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_fitted
        mock_exp_smooth.return_value = mock_model
        
        # Create simple learner
        learner = LearnerExponentialSmoothing(
            trend=None,
            seasonal=None,
            horizon=1
        )
        
        trained = learner.train(exp_smooth_task)
        pred = trained.predict(exp_smooth_task)
        
        assert isinstance(pred, PredictionRegr)
        assert len(pred.response) == exp_smooth_task.nrow
    
    def test_exponential_smoothing_convenience(self):
        """Test convenience function."""
        from mlpy.learners.timeseries import learner_exponential_smoothing
        
        learner = learner_exponential_smoothing(
            trend='mul',
            seasonal='add',
            seasonal_periods=4
        )
        
        assert learner.trend == 'mul'
        assert learner.seasonal == 'add'
        assert learner.seasonal_periods == 4
    
    def test_exponential_smoothing_missing_dependency(self, exp_smooth_task):
        """Test error when statsmodels not available."""
        from mlpy.learners.timeseries import LearnerExponentialSmoothing
        
        with patch('statsmodels.tsa.holtwinters.ExponentialSmoothing', side_effect=ImportError):
            learner = LearnerExponentialSmoothing()
            with pytest.raises(ImportError, match="statsmodels is required"):
                learner.train(exp_smooth_task)


class TestTimeSeriesLearnerIntegration:
    """Test integration between time series learners and tasks."""
    
    @pytest.fixture
    def multi_step_task(self):
        """Create task for multi-step forecasting."""
        dates = pd.date_range('2020-01-01', periods=200, freq='h')
        
        # Generate hourly data
        t = np.arange(200)
        daily = np.sin(2 * np.pi * t / 24) * 2
        trend = 0.01 * t  
        noise = np.random.normal(0, 0.3, 200)
        
        values = 50 + trend + daily + noise
        
        df = pd.DataFrame({
            'datetime': dates,
            'temperature': values,
            'humidity': np.random.uniform(30, 90, 200)
        })
        
        return TaskForecasting(df, time_col='datetime', target='temperature')
    
    def test_different_horizons(self, multi_step_task):
        """Test learners with different forecast horizons."""
        from mlpy.learners.timeseries import LearnerARIMA
        
        # Mock to avoid actual fitting
        with patch('statsmodels.tsa.arima.model.ARIMA') as mock_arima:
            mock_fitted = Mock()
            mock_fitted.forecast.return_value = np.array([51.2, 51.5, 51.8, 52.1, 52.4])
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_fitted
            mock_arima.return_value = mock_model
            
            # Test different horizons
            for horizon in [1, 3, 5, 10]:
                learner = LearnerARIMA(horizon=horizon)
                trained = learner.train(multi_step_task)
                pred = trained.predict(multi_step_task)
                
                assert isinstance(pred, PredictionRegr)
                # Predictions should be padded to match task size
                assert len(pred.response) == multi_step_task.nrow
    
    def test_prediction_with_different_row_ids(self, multi_step_task):
        """Test prediction with subset of row IDs."""
        from mlpy.learners.timeseries import LearnerARIMA
        
        with patch('statsmodels.tsa.arima.model.ARIMA') as mock_arima:
            mock_fitted = Mock()
            mock_fitted.forecast.return_value = np.array([51.2, 51.5])
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_fitted 
            mock_arima.return_value = mock_model
            
            learner = LearnerARIMA(horizon=2)
            trained = learner.train(multi_step_task)
            
            # Predict on subset
            subset_ids = [0, 5, 10, 15, 20]
            pred = trained.predict(multi_step_task, row_ids=subset_ids)
            
            assert len(pred.response) == len(subset_ids)
            assert list(pred.row_ids) == subset_ids
    
    def test_learner_properties_and_packages(self):
        """Test that learners have correct properties and package requirements."""
        from mlpy.learners.timeseries import (
            LearnerARIMA, LearnerProphet, LearnerExponentialSmoothing
        )
        
        # ARIMA
        arima = LearnerARIMA()
        assert 'autoregressive' in arima.properties
        assert 'time_series' in arima.properties
        assert 'statsmodels' in arima.packages
        
        # Prophet
        prophet = LearnerProphet()
        assert 'seasonal' in prophet.properties
        assert 'time_series' in prophet.properties
        assert 'trend' in prophet.properties
        assert 'prophet' in prophet.packages
        
        # Exponential Smoothing
        exp_smooth = LearnerExponentialSmoothing(trend='add', seasonal='add')
        assert 'exponential_smoothing' in exp_smooth.properties
        assert 'time_series' in exp_smooth.properties
        assert 'trend' in exp_smooth.properties
        assert 'seasonal' in exp_smooth.properties
        assert 'statsmodels' in exp_smooth.packages
    
    def test_error_handling(self, multi_step_task):
        """Test error handling in time series learners."""
        from mlpy.learners.timeseries import LearnerARIMA
        
        learner = LearnerARIMA()
        
        # Should raise error when predicting before training
        with pytest.raises(RuntimeError, match="must be trained"):
            learner.predict(multi_step_task)
        
        # Test training failure
        with patch('statsmodels.tsa.arima.model.ARIMA', side_effect=Exception("Model failed")):
            with pytest.raises(RuntimeError, match="ARIMA training failed"):
                learner.train(multi_step_task)
        
        # Test prediction failure
        with patch('statsmodels.tsa.arima.model.ARIMA') as mock_arima:
            mock_fitted = Mock()
            mock_fitted.forecast.side_effect = Exception("Forecast failed")
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_fitted
            mock_arima.return_value = mock_model
            
            trained = learner.train(multi_step_task)
            
            with pytest.raises(RuntimeError, match="ARIMA prediction failed"):
                trained.predict(multi_step_task)