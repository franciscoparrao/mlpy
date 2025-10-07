"""
Time series tasks for MLPY.

This module provides task types for time series analysis including
forecasting and temporal classification.
"""

from abc import ABC
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .base import Task
from .supervised import TaskSupervised
from mlpy.backends.base import DataBackend
from mlpy.backends.pandas_backend import DataBackendPandas
from mlpy.utils.registry import mlpy_tasks


class TaskTimeSeries(Task, ABC):
    """
    Abstract base class for time series tasks.
    
    Time series tasks have temporal structure and require special handling
    for train/test splits, feature engineering, and evaluation.
    
    Parameters
    ----------
    data : pd.DataFrame, DataBackend, or dict
        The data for the task
    time_col : str
        Name of the time/date column
    target : str or List[str], optional
        Name(s) of the target column(s) for supervised learning
    freq : str, optional
        Frequency of the time series (e.g., 'D', 'H', 'M')
        If None, will be inferred from data
    id : str, optional
        Task identifier
    label : str, optional
        Task label
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, DataBackend, Dict[str, Any]],
        time_col: str,
        target: Optional[Union[str, List[str]]] = None,
        freq: Optional[str] = None,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        # Convert data to backend if needed
        if isinstance(data, pd.DataFrame):
            backend = DataBackendPandas(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
            backend = DataBackendPandas(df)
        elif isinstance(data, DataBackend):
            backend = data
        else:
            raise TypeError(
                f"data must be DataFrame, DataBackend, or dict, got {type(data)}"
            )
        
        super().__init__(backend=backend, id=id, label=label, **kwargs)
        
        # Validate time column exists
        if time_col not in backend.colnames:
            raise ValueError(f"Time column '{time_col}' not found in data")
        
        self._time_col = time_col
        self._freq = freq
        
        # Set column roles
        time_cols = {time_col}
        target_cols = set()
        if target is not None:
            if isinstance(target, str):
                target = [target]
            target_cols = set(target)
            
            # Validate target columns exist
            missing_targets = target_cols - set(backend.colnames)
            if missing_targets:
                raise ValueError(f"Target columns not found: {missing_targets}")
        
        all_cols = set(backend.colnames)
        feature_cols = all_cols - time_cols - target_cols
        
        # Initialize properties before setting roles (to avoid validation errors)
        self._is_regular = True  # Will be properly set in _analyze_time_series
        self._time_range = None
        self._n_periods = None
        
        self.set_col_roles({
            "order": list(time_cols),  # Use 'order' role for time columns
            "target": list(target_cols),
            "feature": list(feature_cols),
        })
        
        # Analyze time series properties
        self._analyze_time_series()
    
    def _analyze_time_series(self) -> None:
        """Analyze time series properties and infer frequency if needed."""
        # Get time data
        time_data = self.data(cols=[self._time_col])
        time_series = pd.to_datetime(time_data[self._time_col])
        
        # Sort by time to ensure proper ordering
        time_sorted = time_series.sort_values()
        
        # Infer frequency if not provided
        if self._freq is None:
            try:
                self._freq = pd.infer_freq(time_sorted)
                if self._freq is None:
                    # Try to infer from differences
                    diffs = time_sorted.diff().dropna()
                    most_common_diff = diffs.mode()
                    if not most_common_diff.empty:
                        common_diff = most_common_diff.iloc[0]
                        if common_diff == pd.Timedelta(days=1):
                            self._freq = 'D'
                        elif common_diff == pd.Timedelta(hours=1):
                            self._freq = 'H'
                        elif common_diff <= pd.Timedelta(minutes=1):
                            self._freq = 'T'  # Minute
                        else:
                            warnings.warn(
                                "Could not infer frequency from time series. "
                                "Some functionality may be limited."
                            )
            except Exception as e:
                warnings.warn(f"Could not infer frequency: {e}")
        
        # Store time series properties
        self._time_range = (time_sorted.min(), time_sorted.max())
        self._n_periods = len(time_sorted)
        self._is_regular = self._check_regularity(time_sorted)
        
    def _check_regularity(self, time_series: pd.Series) -> bool:
        """Check if time series has regular intervals."""
        if len(time_series) < 3:
            return True
            
        diffs = time_series.diff().dropna()
        # Allow for small variations (e.g., due to DST)
        return diffs.std() / diffs.mean() < 0.1
    
    def _validate_task(self) -> None:
        """Validate time series task requirements."""
        # Must have a time column
        if not self.time_names:
            raise ValueError("Time series task requires a time column")
        
        # Time series should have at least 2 observations
        if self.nrow < 2:
            raise ValueError(
                f"Time series requires at least 2 observations, got {self.nrow}"
            )
        
        # Warn if irregular time series
        if not self._is_regular:
            warnings.warn(
                "Time series appears to have irregular intervals. "
                "Some methods may not work as expected."
            )
    
    @property
    def time_names(self) -> List[str]:
        """Names of time columns."""
        return sorted(self._col_roles.get("order", set()))
    
    @property
    def time_col(self) -> str:
        """Primary time column name."""
        return self._time_col
    
    @property
    def freq(self) -> Optional[str]:
        """Time series frequency."""
        return self._freq
    
    @property
    def time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Time range of the series (min, max)."""
        return self._time_range
    
    @property
    def n_periods(self) -> int:
        """Number of time periods."""
        return self._n_periods
    
    @property
    def is_regular(self) -> bool:
        """Whether the time series has regular intervals."""
        return self._is_regular
    
    def get_time_data(self, rows: Optional[List[int]] = None) -> pd.Series:
        """
        Get time column data.
        
        Parameters
        ----------
        rows : List[int], optional
            Row indices. If None, all rows in use.
            
        Returns
        -------
        pd.Series
            Time column data as datetime series
        """
        time_data = self.data(rows=rows, cols=[self._time_col])
        return pd.to_datetime(time_data[self._time_col])
    
    def temporal_split(
        self,
        train_size: Optional[Union[int, float]] = None,
        test_size: Optional[Union[int, float]] = None,
        gap: int = 0
    ) -> Tuple[List[int], List[int]]:
        """
        Create temporal train/test split.
        
        Unlike random splits, this respects temporal order by using
        earlier observations for training and later ones for testing.
        
        Parameters
        ----------
        train_size : int or float, optional
            Size of training set. If float, proportion of data.
            If None, will use all data except test_size.
        test_size : int or float, optional
            Size of test set. If float, proportion of data.
            If None, will use remaining data after train_size.
        gap : int, optional
            Number of observations to skip between train and test.
            Useful to avoid data leakage in forecasting.
            
        Returns
        -------
        tuple of (train_indices, test_indices)
        """
        # Get time-sorted indices
        time_data = self.get_time_data()
        sorted_indices = time_data.sort_values().index.tolist()
        n_total = len(sorted_indices)
        
        if train_size is None and test_size is None:
            train_size = 0.8  # Default 80/20 split
        
        if train_size is not None:
            if isinstance(train_size, float):
                train_size = int(train_size * n_total)
            train_end = min(train_size, n_total - gap)
        else:
            if isinstance(test_size, float):
                test_size = int(test_size * n_total)
            train_end = n_total - gap - test_size
        
        if test_size is not None:
            if isinstance(test_size, float):
                test_size = int(test_size * n_total)
            test_start = max(train_end + gap, n_total - test_size)
        else:
            test_start = train_end + gap
        
        train_indices = sorted_indices[:train_end]
        test_indices = sorted_indices[test_start:]
        
        return train_indices, test_indices
    
    def create_lags(
        self,
        columns: Optional[List[str]] = None,
        lags: Union[int, List[int]] = 1,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Create lagged features for time series modeling.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to create lags for. If None, uses all numeric features.
        lags : int or List[int]
            Number of lags to create. If int, creates lags 1 to lags.
            If list, creates specified lags.
        drop_na : bool
            Whether to drop rows with NaN values from lagging.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with original data plus lagged features
        """
        if isinstance(lags, int):
            lags = list(range(1, lags + 1))
        
        data = self.data()
        time_data = self.get_time_data()
        
        # Sort by time
        sort_idx = time_data.sort_values().index
        sorted_data = data.loc[sort_idx].copy()
        
        if columns is None:
            # Use numeric feature columns by default
            numeric_features = []
            for col in self.feature_names:
                if data[col].dtype in ['int64', 'float64']:
                    numeric_features.append(col)
            columns = numeric_features
        
        # Create lagged features
        for col in columns:
            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                sorted_data[lag_col] = sorted_data[col].shift(lag)
        
        if drop_na:
            sorted_data = sorted_data.dropna()
        
        return sorted_data
    
    def create_rolling_features(
        self,
        columns: Optional[List[str]] = None,
        window: int = 3,
        functions: List[str] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to create rolling features for.
        window : int
            Size of the rolling window.
        functions : List[str], optional
            Rolling functions to apply. Default: ['mean', 'std', 'min', 'max']
            
        Returns
        -------
        pd.DataFrame
            DataFrame with rolling features
        """
        if functions is None:
            functions = ['mean', 'std', 'min', 'max']
        
        data = self.data()
        time_data = self.get_time_data()
        
        # Sort by time
        sort_idx = time_data.sort_values().index
        sorted_data = data.loc[sort_idx].copy()
        
        if columns is None:
            # Use numeric feature columns by default
            columns = [col for col in self.feature_names 
                      if data[col].dtype in ['int64', 'float64']]
        
        # Create rolling features
        for col in columns:
            rolling = sorted_data[col].rolling(window=window, min_periods=1)
            
            for func in functions:
                if hasattr(rolling, func):
                    new_col = f"{col}_roll_{window}_{func}"
                    sorted_data[new_col] = getattr(rolling, func)()
        
        return sorted_data
    
    @property
    def _properties(self) -> set[str]:
        """Task properties."""
        props = super()._properties
        props.add("timeseries")
        
        if self._is_regular:
            props.add("regular")
        else:
            props.add("irregular")
            
        if self._freq:
            props.add(f"freq_{self._freq}")
            
        return props


class TaskForecasting(TaskTimeSeries):
    """
    Time series forecasting task.
    
    For predicting future values of a time series based on historical data.
    
    Parameters
    ----------
    data : pd.DataFrame, DataBackend, or dict
        The data for the task
    time_col : str
        Name of the time/date column
    target : str
        Name of the target column to forecast
    horizon : int, optional
        Forecasting horizon (number of periods to predict)
    freq : str, optional
        Frequency of the time series
    id : str, optional
        Task identifier
    label : str, optional
        Task label
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, DataBackend, Dict[str, Any]],
        time_col: str,
        target: str,
        horizon: int = 1,
        freq: Optional[str] = None,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            data=data,
            time_col=time_col,
            target=target,
            freq=freq,
            id=id,
            label=label,
            **kwargs
        )
        
        self._horizon = horizon
        self._validate_forecasting_task()
    
    @property
    def task_type(self) -> str:
        """Task type identifier."""
        return "forecasting"
    
    @property
    def horizon(self) -> int:
        """Forecasting horizon."""
        return self._horizon
    
    def _validate_forecasting_task(self) -> None:
        """Validate forecasting task requirements."""
        # Must have exactly one target
        if len(self.target_names) != 1:
            raise ValueError(
                f"Forecasting requires exactly one target, got {len(self.target_names)}"
            )
        
        # Target should be numeric
        target_col = self.target_names[0]
        col_info = self._backend.col_info()
        target_type = col_info[target_col]["type"]
        
        if target_type not in ("numeric", "integer"):
            raise ValueError(
                f"Forecasting target must be numeric, got type '{target_type}'"
            )
        
        # Horizon should be positive
        if self._horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {self._horizon}")
    
    def create_supervised_features(
        self,
        lags: Union[int, List[int]] = 5,
        include_rolling: bool = True,
        rolling_windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Transform time series into supervised learning format.
        
        Parameters
        ----------
        lags : int or List[int]
            Lags to create for the target variable
        include_rolling : bool
            Whether to include rolling features
        rolling_windows : List[int], optional
            Rolling window sizes. Default: [3, 7, 14]
            
        Returns
        -------
        pd.DataFrame
            Supervised learning dataset
        """
        target_col = self.target_names[0]
        
        # Create lagged target features
        data = self.create_lags(columns=[target_col], lags=lags, drop_na=False)
        
        # Add rolling features if requested
        if include_rolling:
            if rolling_windows is None:
                rolling_windows = [3, 7, 14]
            
            for window in rolling_windows:
                if window < len(data):
                    rolling_data = self.create_rolling_features(
                        columns=[target_col],
                        window=window,
                        functions=['mean', 'std']
                    )
                    # Merge rolling features
                    rolling_cols = [col for col in rolling_data.columns 
                                  if f"_roll_{window}_" in col]
                    for col in rolling_cols:
                        data[col] = rolling_data[col]
        
        return data.dropna()
    
    def truth(self, rows: Optional[List[int]] = None) -> np.ndarray:
        """
        Get true target values.
        
        Parameters
        ----------
        rows : List[int], optional
            Row indices. If None, all rows in use.
            
        Returns
        -------
        np.ndarray
            True target values
        """
        data = self.data(rows=rows, cols=self.target_names, data_format="array")
        return data.ravel()


class TaskTimeSeriesClassification(TaskTimeSeries):
    """
    Time series classification task.
    
    For classifying entire time series or time windows based on temporal patterns.
    
    Parameters
    ----------
    data : pd.DataFrame, DataBackend, or dict
        The data for the task
    time_col : str
        Name of the time/date column
    target : str
        Name of the target column (categorical)
    series_id : str, optional
        Column identifying different time series (for multiple series)
    freq : str, optional
        Frequency of the time series
    id : str, optional
        Task identifier
    label : str, optional
        Task label
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, DataBackend, Dict[str, Any]],
        time_col: str,
        target: str,
        series_id: Optional[str] = None,
        freq: Optional[str] = None,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            data=data,
            time_col=time_col,
            target=target,
            freq=freq,
            id=id,
            label=label,
            **kwargs
        )
        
        self._series_id = series_id
        if series_id:
            # Add series_id to group roles
            if series_id in self._backend.colnames:
                self.set_col_roles({"group": [series_id]})
        
        self._validate_classification_task()
    
    @property
    def task_type(self) -> str:
        """Task type identifier."""
        return "tsclassif"
    
    @property
    def series_id(self) -> Optional[str]:
        """Series identifier column."""
        return self._series_id
    
    @property
    def class_names(self) -> List[str]:
        """Unique class labels in the target, sorted."""
        target_col = self.target_names[0]
        rows_in_use = sorted(self._row_roles["use"])
        
        distinct = self._backend.distinct([target_col], rows=rows_in_use)
        classes = distinct[target_col]
        
        return sorted(str(c) for c in classes)
    
    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(self.class_names)
    
    def _validate_classification_task(self) -> None:
        """Validate time series classification task."""
        # Must have exactly one target
        if len(self.target_names) != 1:
            raise ValueError(
                f"TS Classification requires exactly one target, "
                f"got {len(self.target_names)}"
            )
        
        # Must have at least 2 classes
        if self.n_classes < 2:
            raise ValueError(
                f"Target must have at least 2 classes, got {self.n_classes}"
            )
    
    def extract_features(
        self,
        methods: List[str] = None,
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract time series features for classification.
        
        Parameters
        ----------
        methods : List[str], optional
            Feature extraction methods. Default: statistical features
        window_size : int, optional
            Window size for windowed features. If None, uses entire series.
            
        Returns
        -------
        pd.DataFrame
            Extracted features
        """
        if methods is None:
            methods = ['mean', 'std', 'min', 'max', 'skew', 'kurt']
        
        data = self.data()
        numeric_features = [col for col in self.feature_names 
                           if data[col].dtype in ['int64', 'float64']]
        
        extracted_features = []
        
        # Group by series if series_id is provided
        if self._series_id and self._series_id in data.columns:
            groups = data.groupby(self._series_id)
        else:
            # Single series - treat entire dataset as one group
            groups = [(None, data)]
        
        for group_name, group_data in groups:
            features = {}
            if group_name is not None:
                features[self._series_id] = group_name
            
            for col in numeric_features:
                series = group_data[col].dropna()
                
                if len(series) == 0:
                    continue
                
                for method in methods:
                    if method == 'mean':
                        features[f"{col}_{method}"] = series.mean()
                    elif method == 'std':
                        features[f"{col}_{method}"] = series.std()
                    elif method == 'min':
                        features[f"{col}_{method}"] = series.min()
                    elif method == 'max':
                        features[f"{col}_{method}"] = series.max()
                    elif method == 'skew':
                        features[f"{col}_{method}"] = series.skew()
                    elif method == 'kurt':
                        features[f"{col}_{method}"] = series.kurtosis()
                    elif method == 'median':
                        features[f"{col}_{method}"] = series.median()
                    elif method == 'q25':
                        features[f"{col}_{method}"] = series.quantile(0.25)
                    elif method == 'q75':
                        features[f"{col}_{method}"] = series.quantile(0.75)
            
            # Add target if it exists for this group
            if self.target_names:
                target_col = self.target_names[0]
                if target_col in group_data.columns:
                    # For classification, use the most common class in the group
                    features[target_col] = group_data[target_col].mode().iloc[0] if not group_data[target_col].mode().empty else group_data[target_col].iloc[0]
            
            extracted_features.append(features)
        
        return pd.DataFrame(extracted_features)
    
    def truth(self, rows: Optional[List[int]] = None) -> np.ndarray:
        """
        Get true target values.
        
        Parameters
        ----------
        rows : List[int], optional
            Row indices. If None, all rows in use.
            
        Returns
        -------
        np.ndarray
            True target values
        """
        data = self.data(rows=rows, cols=self.target_names, data_format="array")
        return data.ravel()


# Register task types
mlpy_tasks.register("forecasting", TaskForecasting)
mlpy_tasks.register("tsclassif", TaskTimeSeriesClassification, aliases=["ts_classification"])


__all__ = ["TaskTimeSeries", "TaskForecasting", "TaskTimeSeriesClassification"]