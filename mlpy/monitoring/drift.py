"""
Data drift detection algorithms.

This module provides various algorithms for detecting distribution
shifts in data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DriftResult:
    """Result of drift detection.
    
    Attributes
    ----------
    is_drift : bool
        Whether drift was detected.
    p_value : float
        P-value from statistical test.
    statistic : float
        Test statistic value.
    method : str
        Method used for detection.
    details : Dict[str, Any]
        Additional details about the drift.
    timestamp : datetime
        When the test was performed.
    """
    is_drift: bool
    p_value: float
    statistic: float
    method: str
    details: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.details is None:
            self.details = {}


class DataDriftDetector(ABC):
    """Abstract base class for drift detection."""
    
    def __init__(self, threshold: float = 0.05):
        """Initialize detector.
        
        Parameters
        ----------
        threshold : float
            Significance level for hypothesis testing.
        """
        self.threshold = threshold
        self.reference_data = None
        self.fitted = False
    
    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame]):
        """Fit detector on reference data.
        
        Parameters
        ----------
        reference_data : Union[np.ndarray, pd.DataFrame]
            Reference data to compare against.
        """
        self.reference_data = self._validate_data(reference_data)
        self.fitted = True
        self._fit_implementation(self.reference_data)
    
    @abstractmethod
    def _fit_implementation(self, data: np.ndarray):
        """Implementation-specific fitting logic."""
        pass
    
    @abstractmethod
    def detect(self, data: Union[np.ndarray, pd.DataFrame]) -> DriftResult:
        """Detect drift in new data.
        
        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            New data to test for drift.
            
        Returns
        -------
        DriftResult
            Result of drift detection.
        """
        pass
    
    def _validate_data(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert data to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")


class KSDriftDetector(DataDriftDetector):
    """Kolmogorov-Smirnov test for drift detection.
    
    Best for continuous univariate data.
    """
    
    def _fit_implementation(self, data: np.ndarray):
        """Store reference data statistics."""
        if data.ndim > 1:
            warnings.warn("KS test is univariate. Using first column.")
            self.reference_data = data[:, 0]
    
    def detect(self, data: Union[np.ndarray, pd.DataFrame]) -> DriftResult:
        """Detect drift using KS test."""
        if not self.fitted:
            raise ValueError("Detector must be fitted before detection")
        
        test_data = self._validate_data(data)
        if test_data.ndim > 1:
            test_data = test_data[:, 0]
        
        statistic, p_value = stats.ks_2samp(self.reference_data, test_data)
        
        return DriftResult(
            is_drift=p_value < self.threshold,
            p_value=p_value,
            statistic=statistic,
            method="Kolmogorov-Smirnov",
            details={
                "n_reference": len(self.reference_data),
                "n_test": len(test_data)
            }
        )


class ChiSquaredDriftDetector(DataDriftDetector):
    """Chi-squared test for drift detection.
    
    Best for categorical data.
    """
    
    def __init__(self, threshold: float = 0.05, n_bins: int = 10):
        """Initialize detector.
        
        Parameters
        ----------
        threshold : float
            Significance level.
        n_bins : int
            Number of bins for continuous data.
        """
        super().__init__(threshold)
        self.n_bins = n_bins
        self.bin_edges = None
        self.reference_freq = None
    
    def _fit_implementation(self, data: np.ndarray):
        """Calculate reference frequencies."""
        if data.ndim > 1:
            data = data[:, 0]
        
        # Create bins for continuous data
        if np.issubdtype(data.dtype, np.floating):
            self.bin_edges = np.histogram_bin_edges(data, bins=self.n_bins)
            binned_data = np.digitize(data, self.bin_edges[:-1])
        else:
            binned_data = data
            self.bin_edges = None
        
        # Calculate frequencies
        unique, counts = np.unique(binned_data, return_counts=True)
        self.reference_freq = dict(zip(unique, counts / len(data)))
    
    def detect(self, data: Union[np.ndarray, pd.DataFrame]) -> DriftResult:
        """Detect drift using chi-squared test."""
        if not self.fitted:
            raise ValueError("Detector must be fitted before detection")
        
        test_data = self._validate_data(data)
        if test_data.ndim > 1:
            test_data = test_data[:, 0]
        
        # Bin test data
        if self.bin_edges is not None:
            test_data = np.digitize(test_data, self.bin_edges[:-1])
        
        # Calculate test frequencies
        unique, counts = np.unique(test_data, return_counts=True)
        test_freq = dict(zip(unique, counts / len(test_data)))
        
        # Align frequencies
        all_categories = set(self.reference_freq.keys()) | set(test_freq.keys())
        ref_freqs = np.array([self.reference_freq.get(cat, 0) for cat in all_categories])
        test_freqs = np.array([test_freq.get(cat, 0) for cat in all_categories])
        
        # Chi-squared test
        ref_counts = ref_freqs * len(self.reference_data)
        test_counts = test_freqs * len(test_data)
        
        # Avoid division by zero
        expected = ref_counts + 1e-10
        statistic = np.sum((test_counts - expected) ** 2 / expected)
        p_value = 1 - stats.chi2.cdf(statistic, df=len(all_categories) - 1)
        
        return DriftResult(
            is_drift=p_value < self.threshold,
            p_value=p_value,
            statistic=statistic,
            method="Chi-Squared",
            details={
                "n_categories": len(all_categories),
                "n_reference": len(self.reference_data),
                "n_test": len(test_data)
            }
        )


class PSIDetector(DataDriftDetector):
    """Population Stability Index (PSI) detector.
    
    Commonly used in finance and credit scoring.
    """
    
    def __init__(self, threshold: float = 0.1, n_bins: int = 10):
        """Initialize detector.
        
        Parameters
        ----------
        threshold : float
            PSI threshold (0.1 = small, 0.25 = medium drift).
        n_bins : int
            Number of bins for discretization.
        """
        super().__init__(threshold)
        self.n_bins = n_bins
        self.bin_edges = None
        self.reference_dist = None
    
    def _fit_implementation(self, data: np.ndarray):
        """Calculate reference distribution."""
        if data.ndim > 1:
            data = data[:, 0]
        
        # Create bins
        self.bin_edges = np.quantile(data, np.linspace(0, 1, self.n_bins + 1))
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        
        # Calculate distribution
        hist, _ = np.histogram(data, bins=self.bin_edges)
        self.reference_dist = hist / len(data)
    
    def detect(self, data: Union[np.ndarray, pd.DataFrame]) -> DriftResult:
        """Detect drift using PSI."""
        if not self.fitted:
            raise ValueError("Detector must be fitted before detection")
        
        test_data = self._validate_data(data)
        if test_data.ndim > 1:
            test_data = test_data[:, 0]
        
        # Calculate test distribution
        hist, _ = np.histogram(test_data, bins=self.bin_edges)
        test_dist = hist / len(test_data)
        
        # Calculate PSI
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        psi = np.sum(
            (test_dist - self.reference_dist) * 
            np.log((test_dist + eps) / (self.reference_dist + eps))
        )
        
        return DriftResult(
            is_drift=psi > self.threshold,
            p_value=None,  # PSI doesn't provide p-value
            statistic=psi,
            method="PSI",
            details={
                "n_bins": self.n_bins,
                "n_reference": len(self.reference_data),
                "n_test": len(test_data),
                "interpretation": self._interpret_psi(psi)
            }
        )
    
    def _interpret_psi(self, psi: float) -> str:
        """Interpret PSI value."""
        if psi < 0.1:
            return "No significant drift"
        elif psi < 0.25:
            return "Small drift"
        else:
            return "Significant drift"


class MMDDriftDetector(DataDriftDetector):
    """Maximum Mean Discrepancy (MMD) drift detector.
    
    Non-parametric test for multivariate data.
    """
    
    def __init__(self, threshold: float = 0.05, kernel: str = "rbf", gamma: float = None):
        """Initialize detector.
        
        Parameters
        ----------
        threshold : float
            Significance level.
        kernel : str
            Kernel type ('rbf', 'linear', 'polynomial').
        gamma : float
            Kernel parameter for RBF.
        """
        super().__init__(threshold)
        self.kernel = kernel
        self.gamma = gamma
    
    def _fit_implementation(self, data: np.ndarray):
        """Store reference data."""
        if self.gamma is None and self.kernel == "rbf":
            # Use median heuristic for gamma
            pairwise_dists = np.sum((data[:, None] - data[None, :]) ** 2, axis=-1)
            self.gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])
    
    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """Compute kernel matrix."""
        if Y is None:
            Y = X
        
        if self.kernel == "rbf":
            # RBF kernel
            XX = np.sum(X ** 2, axis=1)[:, None]
            YY = np.sum(Y ** 2, axis=1)[None, :]
            XY = np.dot(X, Y.T)
            distances = XX + YY - 2 * XY
            return np.exp(-self.gamma * distances)
        elif self.kernel == "linear":
            return np.dot(X, Y.T)
        elif self.kernel == "polynomial":
            return (1 + np.dot(X, Y.T)) ** 3
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def detect(self, data: Union[np.ndarray, pd.DataFrame]) -> DriftResult:
        """Detect drift using MMD."""
        if not self.fitted:
            raise ValueError("Detector must be fitted before detection")
        
        test_data = self._validate_data(data)
        
        # Ensure 2D arrays
        if self.reference_data.ndim == 1:
            ref_data = self.reference_data.reshape(-1, 1)
        else:
            ref_data = self.reference_data
            
        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)
        
        n_ref = len(ref_data)
        n_test = len(test_data)
        
        # Compute kernel matrices
        K_ref = self._kernel_matrix(ref_data)
        K_test = self._kernel_matrix(test_data)
        K_cross = self._kernel_matrix(ref_data, test_data)
        
        # Compute MMD
        mmd = (np.sum(K_ref) / (n_ref * n_ref) + 
               np.sum(K_test) / (n_test * n_test) - 
               2 * np.sum(K_cross) / (n_ref * n_test))
        
        # Permutation test for p-value
        n_perms = 100
        combined_data = np.vstack([ref_data, test_data])
        mmd_perms = []
        
        for _ in range(n_perms):
            perm = np.random.permutation(n_ref + n_test)
            perm_ref = combined_data[perm[:n_ref]]
            perm_test = combined_data[perm[n_ref:]]
            
            K_perm_ref = self._kernel_matrix(perm_ref)
            K_perm_test = self._kernel_matrix(perm_test)
            K_perm_cross = self._kernel_matrix(perm_ref, perm_test)
            
            mmd_perm = (np.sum(K_perm_ref) / (n_ref * n_ref) + 
                       np.sum(K_perm_test) / (n_test * n_test) - 
                       2 * np.sum(K_perm_cross) / (n_ref * n_test))
            mmd_perms.append(mmd_perm)
        
        p_value = np.mean(np.array(mmd_perms) >= mmd)
        
        return DriftResult(
            is_drift=p_value < self.threshold,
            p_value=p_value,
            statistic=mmd,
            method="MMD",
            details={
                "kernel": self.kernel,
                "n_reference": n_ref,
                "n_test": n_test,
                "n_permutations": n_perms
            }
        )