"""
Model monitoring functionality.

This module provides tools for monitoring model performance,
data quality, and generating alerts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
import json
from pathlib import Path

from ..learners import Learner
from ..tasks import Task
from ..measures import Measure
from .drift import DataDriftDetector, DriftResult


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert for monitoring issues.
    
    Attributes
    ----------
    level : AlertLevel
        Severity level of the alert.
    message : str
        Alert message.
    metric : str
        Metric that triggered the alert.
    value : float
        Current value of the metric.
    threshold : float
        Threshold that was exceeded.
    timestamp : datetime
        When the alert was generated.
    details : Dict[str, Any]
        Additional alert details.
    """
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }


class ModelMonitor(ABC):
    """Abstract base class for model monitoring."""
    
    def __init__(self, model: Learner, name: str = "model_monitor"):
        """Initialize monitor.
        
        Parameters
        ----------
        model : Learner
            Model to monitor.
        name : str
            Name of the monitor.
        """
        self.model = model
        self.name = name
        self.alerts: List[Alert] = []
        self.history: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
    
    @abstractmethod
    def check(self, data: Union[Task, pd.DataFrame], **kwargs) -> List[Alert]:
        """Check for issues and generate alerts.
        
        Parameters
        ----------
        data : Union[Task, pd.DataFrame]
            Data to check.
        **kwargs
            Additional parameters.
            
        Returns
        -------
        List[Alert]
            List of generated alerts.
        """
        pass
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        since: Optional[datetime] = None
    ) -> List[Alert]:
        """Get filtered alerts.
        
        Parameters
        ----------
        level : Optional[AlertLevel]
            Filter by alert level.
        since : Optional[datetime]
            Get alerts since this time.
            
        Returns
        -------
        List[Alert]
            Filtered alerts.
        """
        alerts = self.alerts
        
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        
        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
    
    def save_state(self, path: Union[str, Path]):
        """Save monitor state to file.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save state.
        """
        path = Path(path)
        state = {
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "alerts": [a.to_dict() for a in self.alerts],
            "history": self.history
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Union[str, Path]):
        """Load monitor state from file.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load state from.
        """
        path = Path(path)
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.name = state["name"]
        self.start_time = datetime.fromisoformat(state["start_time"])
        self.history = state["history"]
        
        # Reconstruct alerts
        self.alerts = []
        for alert_dict in state["alerts"]:
            self.alerts.append(Alert(
                level=AlertLevel(alert_dict["level"]),
                message=alert_dict["message"],
                metric=alert_dict["metric"],
                value=alert_dict["value"],
                threshold=alert_dict["threshold"],
                timestamp=datetime.fromisoformat(alert_dict["timestamp"]),
                details=alert_dict["details"]
            ))


class PerformanceMonitor(ModelMonitor):
    """Monitor model performance metrics.
    
    Tracks model performance over time and alerts on degradation.
    """
    
    def __init__(
        self,
        model: Learner,
        measures: List[Measure],
        thresholds: Optional[Dict[str, float]] = None,
        window_size: int = 100,
        name: str = "performance_monitor"
    ):
        """Initialize performance monitor.
        
        Parameters
        ----------
        model : Learner
            Model to monitor.
        measures : List[Measure]
            Performance measures to track.
        thresholds : Optional[Dict[str, float]]
            Alert thresholds for each measure.
        window_size : int
            Size of sliding window for metrics.
        name : str
            Name of the monitor.
        """
        super().__init__(model, name)
        self.measures = measures
        self.thresholds = thresholds or {}
        self.window_size = window_size
        self.metric_history: Dict[str, List[float]] = {m.id: [] for m in measures}
        self.baseline_metrics: Dict[str, float] = {}
    
    def set_baseline(self, task: Task):
        """Set baseline performance metrics.
        
        Parameters
        ----------
        task : Task
            Task to evaluate baseline performance.
        """
        prediction = self.model.predict(task)
        
        for measure in self.measures:
            score = measure.score(prediction)
            self.baseline_metrics[measure.id] = score
            
            # Set default threshold if not provided
            if measure.id not in self.thresholds:
                # Alert if performance drops by more than 10%
                if measure.higher_better:
                    self.thresholds[measure.id] = score * 0.9
                else:
                    self.thresholds[measure.id] = score * 1.1
    
    def check(self, data: Union[Task, pd.DataFrame], **kwargs) -> List[Alert]:
        """Check model performance.
        
        Parameters
        ----------
        data : Union[Task, pd.DataFrame]
            Data to evaluate.
            
        Returns
        -------
        List[Alert]
            Performance alerts.
        """
        alerts = []
        
        # Convert to task if needed
        if isinstance(data, pd.DataFrame):
            # Assume last column is target
            from ..tasks import TaskClassif, TaskRegr
            if self.model.task_type == "classification":
                task = TaskClassif(data, target=data.columns[-1])
            else:
                task = TaskRegr(data, target=data.columns[-1])
        else:
            task = data
        
        # Get predictions and evaluate
        prediction = self.model.predict(task)
        
        for measure in self.measures:
            score = measure.score(prediction)
            self.metric_history[measure.id].append(score)
            
            # Keep only recent history
            if len(self.metric_history[measure.id]) > self.window_size:
                self.metric_history[measure.id].pop(0)
            
            # Check threshold
            if measure.id in self.thresholds:
                threshold = self.thresholds[measure.id]
                
                if measure.higher_better:
                    if score < threshold:
                        alerts.append(Alert(
                            level=AlertLevel.WARNING,
                            message=f"Performance degradation detected for {measure.id}",
                            metric=measure.id,
                            value=score,
                            threshold=threshold,
                            details={
                                "baseline": self.baseline_metrics.get(measure.id),
                                "window_avg": np.mean(self.metric_history[measure.id])
                            }
                        ))
                else:
                    if score > threshold:
                        alerts.append(Alert(
                            level=AlertLevel.WARNING,
                            message=f"Performance degradation detected for {measure.id}",
                            metric=measure.id,
                            value=score,
                            threshold=threshold,
                            details={
                                "baseline": self.baseline_metrics.get(measure.id),
                                "window_avg": np.mean(self.metric_history[measure.id])
                            }
                        ))
        
        # Add alerts to history
        self.alerts.extend(alerts)
        
        # Record in history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": {m.id: self.metric_history[m.id][-1] for m in self.measures},
            "n_alerts": len(alerts)
        })
        
        return alerts
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Summary statistics for each metric.
        """
        summary = {}
        
        for measure in self.measures:
            history = self.metric_history[measure.id]
            if history:
                summary[measure.id] = {
                    "current": history[-1],
                    "mean": np.mean(history),
                    "std": np.std(history),
                    "min": np.min(history),
                    "max": np.max(history),
                    "baseline": self.baseline_metrics.get(measure.id),
                    "threshold": self.thresholds.get(measure.id)
                }
        
        return summary


class DataQualityMonitor(ModelMonitor):
    """Monitor data quality issues.
    
    Detects missing values, outliers, and data drift.
    """
    
    def __init__(
        self,
        model: Learner,
        drift_detectors: Optional[Dict[str, DataDriftDetector]] = None,
        missing_threshold: float = 0.1,
        outlier_threshold: float = 3.0,
        name: str = "data_quality_monitor"
    ):
        """Initialize data quality monitor.
        
        Parameters
        ----------
        model : Learner
            Model to monitor.
        drift_detectors : Optional[Dict[str, DataDriftDetector]]
            Drift detectors for each feature.
        missing_threshold : float
            Threshold for missing value alerts.
        outlier_threshold : float
            Z-score threshold for outlier detection.
        name : str
            Name of the monitor.
        """
        super().__init__(model, name)
        self.drift_detectors = drift_detectors or {}
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.reference_stats: Dict[str, Dict[str, float]] = {}
    
    def set_reference(self, data: pd.DataFrame):
        """Set reference data statistics.
        
        Parameters
        ----------
        data : pd.DataFrame
            Reference data.
        """
        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                self.reference_stats[col] = {
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max()
                }
        
        # Fit drift detectors
        for col, detector in self.drift_detectors.items():
            if col in data.columns:
                detector.fit(data[col].values)
    
    def check(self, data: Union[Task, pd.DataFrame], **kwargs) -> List[Alert]:
        """Check data quality.
        
        Parameters
        ----------
        data : Union[Task, pd.DataFrame]
            Data to check.
            
        Returns
        -------
        List[Alert]
            Data quality alerts.
        """
        alerts = []
        
        # Convert to DataFrame if needed
        if isinstance(data, Task):
            df = data.data
        else:
            df = data
        
        # Check missing values
        missing_rates = df.isnull().mean()
        for col, rate in missing_rates.items():
            if rate > self.missing_threshold:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    message=f"High missing value rate in {col}",
                    metric="missing_rate",
                    value=rate,
                    threshold=self.missing_threshold,
                    details={"column": col}
                ))
        
        # Check for outliers using Z-score
        for col in df.columns:
            if col in self.reference_stats:
                stats = self.reference_stats[col]
                z_scores = np.abs((df[col] - stats["mean"]) / (stats["std"] + 1e-10))
                outlier_rate = (z_scores > self.outlier_threshold).mean()
                
                if outlier_rate > 0.01:  # More than 1% outliers
                    alerts.append(Alert(
                        level=AlertLevel.INFO,
                        message=f"Outliers detected in {col}",
                        metric="outlier_rate",
                        value=outlier_rate,
                        threshold=0.01,
                        details={
                            "column": col,
                            "n_outliers": int(outlier_rate * len(df))
                        }
                    ))
        
        # Check for drift
        for col, detector in self.drift_detectors.items():
            if col in df.columns and detector.fitted:
                drift_result = detector.detect(df[col].values)
                
                if drift_result.is_drift:
                    alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        message=f"Data drift detected in {col}",
                        metric="drift",
                        value=drift_result.statistic,
                        threshold=detector.threshold,
                        details={
                            "column": col,
                            "method": drift_result.method,
                            "p_value": drift_result.p_value
                        }
                    ))
        
        # Add alerts to history
        self.alerts.extend(alerts)
        
        # Record in history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "missing_rate": df.isnull().mean().mean(),
            "n_alerts": len(alerts)
        })
        
        return alerts
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get data quality summary.
        
        Returns
        -------
        Dict[str, Any]
            Data quality summary.
        """
        if not self.history:
            return {}
        
        recent = self.history[-1]
        
        # Count alerts by type
        alert_counts = {
            "missing": 0,
            "outlier": 0,
            "drift": 0
        }
        
        for alert in self.get_alerts(since=datetime.now() - timedelta(hours=1)):
            if alert.metric == "missing_rate":
                alert_counts["missing"] += 1
            elif alert.metric == "outlier_rate":
                alert_counts["outlier"] += 1
            elif alert.metric == "drift":
                alert_counts["drift"] += 1
        
        return {
            "last_check": recent["timestamp"],
            "n_rows": recent["n_rows"],
            "n_cols": recent["n_cols"],
            "overall_missing_rate": recent["missing_rate"],
            "alert_counts": alert_counts,
            "total_alerts": sum(alert_counts.values())
        }