"""
Model Monitoring and Drift Detection
=====================================

Monitor model performance and detect data/concept drift.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import logging
import json
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection."""
    
    feature_name: str
    drift_detected: bool
    drift_score: float
    p_value: float
    method: str
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "method": self.method,
            "threshold": self.threshold,
            "timestamp": self.timestamp
        }


@dataclass
class PerformanceMetrics:
    """Model performance metrics over time."""
    
    timestamp: str
    model_id: str
    metric_name: str
    value: float
    num_predictions: int
    window: str  # "hourly", "daily", "weekly"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "num_predictions": self.num_predictions,
            "window": self.window
        }


class DriftDetector:
    """Detects distribution drift in features and predictions."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        method: str = "ks",
        threshold: float = 0.05,
        window_size: int = 1000
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference/training data distribution
            method: Detection method ("ks", "chi2", "psi", "wasserstein")
            threshold: P-value threshold for drift detection
            window_size: Size of sliding window for monitoring
        """
        self.reference_data = reference_data
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self.feature_stats = self._calculate_reference_stats()
        self.drift_history: List[DriftResult] = []
    
    def _calculate_reference_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for reference data."""
        stats = {}
        
        for col in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                stats[col] = {
                    "type": "numeric",
                    "mean": self.reference_data[col].mean(),
                    "std": self.reference_data[col].std(),
                    "min": self.reference_data[col].min(),
                    "max": self.reference_data[col].max(),
                    "quantiles": self.reference_data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                stats[col] = {
                    "type": "categorical",
                    "unique_values": self.reference_data[col].nunique(),
                    "value_counts": self.reference_data[col].value_counts().to_dict()
                }
        
        return stats
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, DriftResult]:
        """
        Detect drift in current data compared to reference.
        
        Args:
            current_data: Current/production data
            
        Returns:
            Dictionary of drift results per feature
        """
        results = {}
        
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                logger.warning(f"Column {col} not found in current data")
                continue
            
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                result = self._detect_numeric_drift(col, current_data[col])
            else:
                result = self._detect_categorical_drift(col, current_data[col])
            
            results[col] = result
            self.drift_history.append(result)
        
        return results
    
    def _detect_numeric_drift(self, feature_name: str, current_values: pd.Series) -> DriftResult:
        """Detect drift in numeric features."""
        reference_values = self.reference_data[feature_name].dropna()
        current_values = current_values.dropna()
        
        if self.method == "ks":
            # Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(reference_values, current_values)
            drift_detected = p_value < self.threshold
            
        elif self.method == "wasserstein":
            # Wasserstein distance
            from scipy.stats import wasserstein_distance
            distance = wasserstein_distance(reference_values, current_values)
            # Normalize by reference std
            ref_std = reference_values.std()
            drift_score = distance / ref_std if ref_std > 0 else distance
            p_value = 1.0 - min(drift_score, 1.0)  # Pseudo p-value
            drift_detected = drift_score > 0.1
            statistic = drift_score
            
        elif self.method == "psi":
            # Population Stability Index
            psi = self._calculate_psi(reference_values, current_values)
            drift_detected = psi > 0.2  # Common threshold
            statistic = psi
            p_value = 1.0 - min(psi, 1.0)  # Pseudo p-value
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=statistic,
            p_value=p_value,
            method=self.method,
            threshold=self.threshold
        )
    
    def _detect_categorical_drift(self, feature_name: str, current_values: pd.Series) -> DriftResult:
        """Detect drift in categorical features."""
        reference_counts = self.reference_data[feature_name].value_counts()
        current_counts = current_values.value_counts()
        
        # Align categories
        all_categories = set(reference_counts.index) | set(current_counts.index)
        ref_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
        curr_aligned = [current_counts.get(cat, 0) for cat in all_categories]
        
        if self.method in ["ks", "chi2"]:
            # Chi-square test
            chi2, p_value, _, _ = chi2_contingency([ref_aligned, curr_aligned])
            drift_detected = p_value < self.threshold
            statistic = chi2
            
        elif self.method == "psi":
            # Population Stability Index for categorical
            psi = self._calculate_psi_categorical(reference_counts, current_counts)
            drift_detected = psi > 0.2
            statistic = psi
            p_value = 1.0 - min(psi, 1.0)
            
        else:
            # Default to chi-square
            chi2, p_value, _, _ = chi2_contingency([ref_aligned, curr_aligned])
            drift_detected = p_value < self.threshold
            statistic = chi2
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=statistic,
            p_value=p_value,
            method="chi2" if self.method in ["ks", "chi2"] else self.method,
            threshold=self.threshold
        )
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on reference data
        _, bins = pd.qcut(reference, q=n_bins, retbins=True, duplicates='drop')
        
        # Calculate frequencies
        ref_freq = pd.cut(reference, bins=bins, include_lowest=True).value_counts(normalize=True).sort_index()
        curr_freq = pd.cut(current, bins=bins, include_lowest=True).value_counts(normalize=True).sort_index()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_freq = ref_freq + epsilon
        curr_freq = curr_freq + epsilon
        
        # Calculate PSI
        psi = np.sum((curr_freq - ref_freq) * np.log(curr_freq / ref_freq))
        return psi
    
    def _calculate_psi_categorical(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate PSI for categorical variables."""
        # Normalize to get probabilities
        ref_prob = reference / reference.sum()
        curr_prob = current / current.sum()
        
        # Align categories
        all_cats = set(ref_prob.index) | set(curr_prob.index)
        
        psi = 0
        epsilon = 1e-10
        
        for cat in all_cats:
            ref_p = ref_prob.get(cat, epsilon)
            curr_p = curr_prob.get(cat, epsilon)
            psi += (curr_p - ref_p) * np.log(curr_p / ref_p)
        
        return psi
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift report."""
        if not self.drift_history:
            return {"message": "No drift detection performed yet"}
        
        # Group by feature
        feature_drift = {}
        for result in self.drift_history:
            if result.feature_name not in feature_drift:
                feature_drift[result.feature_name] = []
            feature_drift[result.feature_name].append(result)
        
        report = {
            "summary": {
                "total_features": len(feature_drift),
                "features_with_drift": sum(
                    1 for results in feature_drift.values()
                    if any(r.drift_detected for r in results)
                ),
                "last_check": self.drift_history[-1].timestamp if self.drift_history else None
            },
            "features": {}
        }
        
        for feature, results in feature_drift.items():
            latest = results[-1]
            report["features"][feature] = {
                "drift_detected": latest.drift_detected,
                "latest_score": latest.drift_score,
                "latest_p_value": latest.p_value,
                "drift_frequency": sum(r.drift_detected for r in results) / len(results),
                "checks_performed": len(results)
            }
        
        return report


class PerformanceMonitor:
    """Monitor model performance over time."""
    
    def __init__(
        self,
        model_id: str,
        metrics_file: Optional[str] = None,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize performance monitor.
        
        Args:
            model_id: Identifier for the model
            metrics_file: Path to store metrics history
            alert_thresholds: Thresholds for alerting
        """
        self.model_id = model_id
        self.metrics_file = Path(metrics_file) if metrics_file else Path(f"{model_id}_metrics.json")
        self.alert_thresholds = alert_thresholds or {}
        self.metrics_history: List[PerformanceMetrics] = []
        self.load_metrics()
    
    def load_metrics(self):
        """Load metrics history from file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                self.metrics_history = [
                    PerformanceMetrics(**item) for item in data
                ]
    
    def save_metrics(self):
        """Save metrics history to file."""
        data = [m.to_dict() for m in self.metrics_history]
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        num_predictions: int = 1,
        window: str = "hourly"
    ):
        """Record a performance metric."""
        metric = PerformanceMetrics(
            timestamp=datetime.utcnow().isoformat(),
            model_id=self.model_id,
            metric_name=metric_name,
            value=value,
            num_predictions=num_predictions,
            window=window
        )
        
        self.metrics_history.append(metric)
        self.save_metrics()
        
        # Check for alerts
        if metric_name in self.alert_thresholds:
            threshold = self.alert_thresholds[metric_name]
            if value > threshold:
                logger.warning(
                    f"ALERT: {metric_name} ({value:.4f}) exceeded threshold ({threshold:.4f})"
                )
    
    def get_metrics_summary(
        self,
        metric_name: Optional[str] = None,
        window: Optional[str] = None,
        last_n_hours: int = 24
    ) -> Dict[str, Any]:
        """Get summary of metrics."""
        # Filter metrics
        cutoff_time = datetime.utcnow() - timedelta(hours=last_n_hours)
        
        filtered = self.metrics_history
        if metric_name:
            filtered = [m for m in filtered if m.metric_name == metric_name]
        if window:
            filtered = [m for m in filtered if m.window == window]
        
        filtered = [
            m for m in filtered
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        if not filtered:
            return {"message": "No metrics found for the specified criteria"}
        
        # Calculate statistics
        values = [m.value for m in filtered]
        
        return {
            "metric_name": metric_name or "all",
            "window": window or "all",
            "time_range_hours": last_n_hours,
            "num_records": len(filtered),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "last_value": filtered[-1].value if filtered else None,
            "last_timestamp": filtered[-1].timestamp if filtered else None,
            "trend": self._calculate_trend(filtered)
        }
    
    def _calculate_trend(self, metrics: List[PerformanceMetrics]) -> str:
        """Calculate trend direction."""
        if len(metrics) < 2:
            return "insufficient_data"
        
        # Simple linear regression on values
        x = np.arange(len(metrics))
        y = [m.value for m in metrics]
        
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def detect_anomalies(
        self,
        metric_name: str,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        # Filter metrics
        metrics = [m for m in self.metrics_history if m.metric_name == metric_name]
        
        if len(metrics) < 10:
            return []
        
        values = np.array([m.value for m in metrics])
        
        if method == "zscore":
            # Z-score method
            mean = np.mean(values)
            std = np.std(values)
            z_scores = np.abs((values - mean) / std)
            anomaly_indices = np.where(z_scores > threshold)[0]
            
        elif method == "iqr":
            # Interquartile range method
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            anomaly_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
            
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        anomalies = []
        for idx in anomaly_indices:
            metric = metrics[idx]
            anomalies.append({
                "timestamp": metric.timestamp,
                "value": metric.value,
                "expected_range": (
                    float(np.mean(values) - threshold * np.std(values)),
                    float(np.mean(values) + threshold * np.std(values))
                ) if method == "zscore" else None
            })
        
        return anomalies