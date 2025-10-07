"""
Tests for Model Monitoring and Drift Detection.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from mlpy.monitoring import (
    DataDriftDetector, KSDriftDetector, ChiSquaredDriftDetector,
    PSIDetector, MMDDriftDetector,
    ModelMonitor, PerformanceMonitor, DataQualityMonitor,
    Alert, AlertLevel,
    calculate_psi, calculate_kl_divergence,
    calculate_wasserstein_distance, calculate_jensen_shannon_divergence
)
from mlpy.monitoring.drift import DriftResult
from mlpy.monitoring.metrics import (
    calculate_hellinger_distance, calculate_total_variation_distance,
    calculate_chi_squared_statistic, calculate_cramer_von_mises,
    calculate_anderson_darling
)

from mlpy.learners import LearnerClassifSklearn
from mlpy.tasks import TaskClassif
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


class TestDriftDetectors:
    """Test drift detection algorithms."""
    
    @pytest.fixture
    def reference_data(self):
        """Create reference data."""
        np.random.seed(42)
        return np.random.normal(0, 1, 1000)
    
    @pytest.fixture
    def no_drift_data(self):
        """Create data without drift."""
        np.random.seed(43)
        return np.random.normal(0, 1, 1000)
    
    @pytest.fixture
    def drift_data(self):
        """Create data with drift."""
        np.random.seed(44)
        return np.random.normal(2, 1.5, 1000)  # Different mean and std
    
    def test_ks_drift_detector(self, reference_data, no_drift_data, drift_data):
        """Test Kolmogorov-Smirnov drift detector."""
        detector = KSDriftDetector(threshold=0.05)
        
        # Fit on reference data
        detector.fit(reference_data)
        assert detector.fitted
        
        # Test on no drift data
        result = detector.detect(no_drift_data)
        assert isinstance(result, DriftResult)
        assert not result.is_drift
        assert result.p_value > 0.05
        assert result.method == "Kolmogorov-Smirnov"
        
        # Test on drift data
        result = detector.detect(drift_data)
        assert result.is_drift
        assert result.p_value < 0.05
    
    def test_chi_squared_drift_detector(self, reference_data, no_drift_data, drift_data):
        """Test Chi-squared drift detector."""
        detector = ChiSquaredDriftDetector(threshold=0.05, n_bins=10)
        
        # Fit on reference data
        detector.fit(reference_data)
        assert detector.fitted
        assert detector.bin_edges is not None
        
        # Test on no drift data
        result = detector.detect(no_drift_data)
        assert isinstance(result, DriftResult)
        assert result.method == "Chi-Squared"
        
        # Test on drift data
        result = detector.detect(drift_data)
        # Chi-squared should detect the distribution difference
        assert result.statistic > 0
    
    def test_psi_detector(self, reference_data, no_drift_data, drift_data):
        """Test PSI drift detector."""
        detector = PSIDetector(threshold=0.1, n_bins=10)
        
        # Fit on reference data
        detector.fit(reference_data)
        assert detector.fitted
        assert detector.bin_edges is not None
        assert detector.reference_dist is not None
        
        # Test on no drift data
        result = detector.detect(no_drift_data)
        assert isinstance(result, DriftResult)
        assert not result.is_drift
        assert result.statistic < 0.1
        assert result.method == "PSI"
        assert "interpretation" in result.details
        
        # Test on drift data
        result = detector.detect(drift_data)
        assert result.is_drift
        assert result.statistic > 0.1
    
    def test_mmd_drift_detector(self, reference_data, no_drift_data, drift_data):
        """Test MMD drift detector."""
        detector = MMDDriftDetector(threshold=0.05, kernel="rbf")
        
        # Fit on reference data
        detector.fit(reference_data.reshape(-1, 1))  # MMD needs 2D data
        assert detector.fitted
        assert detector.gamma is not None  # Should be set by median heuristic
        
        # Test on no drift data
        result = detector.detect(no_drift_data.reshape(-1, 1))
        assert isinstance(result, DriftResult)
        assert result.method == "MMD"
        assert "n_permutations" in result.details
        
        # Test on drift data
        result = detector.detect(drift_data.reshape(-1, 1))
        # MMD should detect the distribution difference
        assert result.statistic > 0
    
    def test_detector_with_pandas(self, reference_data):
        """Test detector with pandas DataFrame."""
        detector = KSDriftDetector()
        
        # Create DataFrame
        df_ref = pd.DataFrame({"feature": reference_data})
        df_test = pd.DataFrame({"feature": reference_data + 0.1})
        
        # Should work with DataFrame
        detector.fit(df_ref)
        result = detector.detect(df_test)
        assert isinstance(result, DriftResult)
    
    def test_detector_not_fitted_error(self, reference_data):
        """Test error when detector not fitted."""
        detector = KSDriftDetector()
        
        with pytest.raises(ValueError, match="fitted"):
            detector.detect(reference_data)


class TestMonitoring:
    """Test model monitoring functionality."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
        df["target"] = y
        
        task = TaskClassif(df, target="target")
        learner = LearnerClassifSklearn(LogisticRegression(random_state=42))
        learner.train(task)
        learner.task_type = "classification"  # Set task type
        return learner
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task."""
        X, y = make_classification(n_samples=50, n_features=10, random_state=43)
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
        df["target"] = y
        return TaskClassif(df, target="target")
    
    def test_alert_creation(self):
        """Test Alert creation."""
        alert = Alert(
            level=AlertLevel.WARNING,
            message="Test alert",
            metric="accuracy",
            value=0.85,
            threshold=0.90
        )
        
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test alert"
        assert alert.metric == "accuracy"
        assert alert.value == 0.85
        assert alert.threshold == 0.90
        assert isinstance(alert.timestamp, datetime)
        
        # Test to_dict
        alert_dict = alert.to_dict()
        assert alert_dict["level"] == "warning"
        assert alert_dict["metric"] == "accuracy"
    
    def test_performance_monitor(self, sample_model, sample_task):
        """Test PerformanceMonitor."""
        measures = [MeasureClassifAccuracy(), MeasureClassifF1()]
        monitor = PerformanceMonitor(
            model=sample_model,
            measures=measures,
            thresholds={"classif.acc": 0.7},
            window_size=10
        )
        
        # Set baseline
        monitor.set_baseline(sample_task)
        assert len(monitor.baseline_metrics) == 2
        assert "classif.acc" in monitor.baseline_metrics
        
        # Check performance
        alerts = monitor.check(sample_task)
        assert isinstance(alerts, list)
        
        # Get metrics summary
        summary = monitor.get_metrics_summary()
        assert "classif.acc" in summary
        assert "current" in summary["classif.acc"]
        assert "mean" in summary["classif.acc"]
    
    def test_data_quality_monitor(self, sample_model):
        """Test DataQualityMonitor."""
        # Create monitor with drift detector
        drift_detector = KSDriftDetector(threshold=0.05)
        monitor = DataQualityMonitor(
            model=sample_model,
            drift_detectors={"feat_0": drift_detector},
            missing_threshold=0.1,
            outlier_threshold=3.0
        )
        
        # Create reference data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        df_ref = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
        df_ref["target"] = y
        
        # Set reference
        monitor.set_reference(df_ref)
        assert "feat_0" in monitor.reference_stats
        assert drift_detector.fitted
        
        # Create test data with issues
        df_test = df_ref.copy()
        # Add missing values
        df_test.loc[0:15, "feat_1"] = np.nan
        # Add outliers
        df_test.loc[20:25, "feat_2"] = 100
        
        # Check data quality
        alerts = monitor.check(df_test)
        assert isinstance(alerts, list)
        
        # Should have alerts for missing values
        missing_alerts = [a for a in alerts if a.metric == "missing_rate"]
        assert len(missing_alerts) > 0
        
        # Get summary
        summary = monitor.get_data_summary()
        assert "last_check" in summary
        assert "alert_counts" in summary
    
    def test_monitor_alert_filtering(self, sample_model):
        """Test alert filtering."""
        monitor = PerformanceMonitor(
            model=sample_model,
            measures=[MeasureClassifAccuracy()]
        )
        
        # Add some alerts
        now = datetime.now()
        monitor.alerts = [
            Alert(AlertLevel.INFO, "Info", "metric1", 0.5, 0.6),
            Alert(AlertLevel.WARNING, "Warning", "metric2", 0.7, 0.8),
            Alert(AlertLevel.CRITICAL, "Critical", "metric3", 0.3, 0.4)
        ]
        
        # Filter by level
        warnings = monitor.get_alerts(level=AlertLevel.WARNING)
        assert len(warnings) == 1
        assert warnings[0].level == AlertLevel.WARNING
        
        # Filter by time
        recent = monitor.get_alerts(since=now - timedelta(hours=1))
        assert len(recent) == 3
        
        # Clear alerts
        monitor.clear_alerts()
        assert len(monitor.alerts) == 0
    
    def test_monitor_state_persistence(self, sample_model, tmp_path):
        """Test saving and loading monitor state."""
        monitor = PerformanceMonitor(
            model=sample_model,
            measures=[MeasureClassifAccuracy()],
            name="test_monitor"
        )
        
        # Add some alerts and history
        monitor.alerts.append(
            Alert(AlertLevel.WARNING, "Test", "accuracy", 0.8, 0.9)
        )
        monitor.history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": {"accuracy": 0.8}
        })
        
        # Save state
        state_file = tmp_path / "monitor_state.json"
        monitor.save_state(state_file)
        assert state_file.exists()
        
        # Create new monitor and load state
        new_monitor = PerformanceMonitor(
            model=sample_model,
            measures=[MeasureClassifAccuracy()]
        )
        new_monitor.load_state(state_file)
        
        assert new_monitor.name == "test_monitor"
        assert len(new_monitor.alerts) == 1
        assert len(new_monitor.history) == 1
        assert new_monitor.alerts[0].metric == "accuracy"


class TestDriftMetrics:
    """Test drift detection metrics."""
    
    def test_calculate_psi(self):
        """Test PSI calculation."""
        reference = np.random.normal(0, 1, 1000)
        current_no_drift = np.random.normal(0, 1, 1000)
        current_drift = np.random.normal(2, 1, 1000)
        
        # No drift case
        psi_no_drift = calculate_psi(reference, current_no_drift)
        assert psi_no_drift < 0.1  # No significant drift
        
        # Drift case
        psi_drift = calculate_psi(reference, current_drift)
        assert psi_drift > 0.1  # Significant drift
    
    def test_calculate_kl_divergence(self):
        """Test KL divergence calculation."""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.25, 0.5, 0.25])
        
        kl_div = calculate_kl_divergence(p, q)
        assert kl_div >= 0  # KL divergence is always non-negative
        
        # Same distribution
        kl_same = calculate_kl_divergence(p, p)
        assert kl_same < 0.001  # Should be close to 0
    
    def test_calculate_jensen_shannon(self):
        """Test Jensen-Shannon divergence."""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.25, 0.5, 0.25])
        
        js_div = calculate_jensen_shannon_divergence(p, q)
        assert 0 <= js_div <= 1  # JS divergence is bounded
        
        # Symmetric property
        js_div_reversed = calculate_jensen_shannon_divergence(q, p)
        assert np.isclose(js_div, js_div_reversed)
    
    def test_calculate_wasserstein(self):
        """Test Wasserstein distance."""
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)
        
        w_dist = calculate_wasserstein_distance(reference, current)
        assert w_dist > 0  # Different distributions
        
        # Same distribution
        w_same = calculate_wasserstein_distance(reference, reference)
        assert w_same == 0
    
    def test_calculate_hellinger(self):
        """Test Hellinger distance."""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.25, 0.5, 0.25])
        
        h_dist = calculate_hellinger_distance(p, q)
        assert 0 <= h_dist <= 1  # Hellinger distance is bounded
        
        # Same distribution
        h_same = calculate_hellinger_distance(p, p)
        assert h_same < 0.001
    
    def test_calculate_total_variation(self):
        """Test Total Variation distance."""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.25, 0.5, 0.25])
        
        tv_dist = calculate_total_variation_distance(p, q)
        assert 0 <= tv_dist <= 1  # TV distance is bounded
        
        # Same distribution
        tv_same = calculate_total_variation_distance(p, p)
        assert tv_same == 0
    
    def test_calculate_chi_squared(self):
        """Test Chi-squared statistic."""
        observed = np.array([10, 15, 20, 25])
        expected = np.array([12, 14, 18, 26])
        
        chi2, df = calculate_chi_squared_statistic(observed, expected)
        assert chi2 > 0
        assert df == len(observed) - 1
    
    def test_calculate_cramer_von_mises(self):
        """Test Cramér-von Mises statistic."""
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)
        
        cvm = calculate_cramer_von_mises(reference, current)
        assert cvm > 0
    
    def test_calculate_anderson_darling(self):
        """Test Anderson-Darling statistic."""
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)
        
        ad = calculate_anderson_darling(reference, current)
        assert ad > 0