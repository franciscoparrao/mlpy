"""
Improved Gaussian Process fallback implementation for MLPY.

This provides a more robust GP implementation when TGPY is not available
or not fully functional.
"""

import numpy as np
from typing import Tuple, Optional


class SimpleGP:
    """Simple Gaussian Process implementation with RBF kernel."""
    
    def __init__(
        self,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise: float = 0.1
    ):
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        
    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF (Squared Exponential) kernel matrix."""
        # Compute pairwise squared distances
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        # Ensure non-negative (numerical issues)
        sqdist = np.maximum(sqdist, 0)
        # Compute RBF kernel
        K = self.variance * np.exp(-0.5 * sqdist / self.lengthscale**2)
        return K
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Gaussian Process."""
        self.X_train = X
        self.y_train = y
        
        # Compute kernel matrix
        K = self.rbf_kernel(X, X)
        
        # Add noise to diagonal
        K += self.noise * np.eye(len(X))
        
        # Compute inverse using Cholesky decomposition for stability
        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.K_inv = None  # Don't store full inverse
            self.L = L
        except np.linalg.LinAlgError:
            # Fall back to standard inverse if Cholesky fails
            self.K_inv = np.linalg.pinv(K)
            self.alpha = self.K_inv @ y
            self.L = None
            
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation."""
        # Compute kernel between test and training points
        K_star = self.rbf_kernel(X, self.X_train)
        
        # Predict mean
        y_mean = K_star @ self.alpha
        
        # Predict variance
        K_star_star = self.rbf_kernel(X, X)
        
        if self.L is not None:
            # Use Cholesky decomposition
            v = np.linalg.solve(self.L, K_star.T)
            y_var = np.diag(K_star_star) - np.sum(v**2, axis=0)
        else:
            # Use direct computation
            y_var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
            
        # Ensure non-negative variance
        y_var = np.maximum(y_var, 0)
        y_std = np.sqrt(y_var + self.noise)
        
        return y_mean, y_std
        
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_restarts: int = 5
    ):
        """Simple hyperparameter optimization using grid search."""
        # Define search grid
        lengthscales = np.logspace(-1, 1, 10)
        variances = np.logspace(-1, 1, 10)
        noises = np.logspace(-3, 0, 10)
        
        best_nll = np.inf
        best_params = (self.lengthscale, self.variance, self.noise)
        
        for ls in lengthscales:
            for var in variances:
                for noise in noises:
                    try:
                        # Compute negative log likelihood
                        K = var * np.exp(-0.5 * self._compute_distances(X, X) / ls**2)
                        K += noise * np.eye(len(X))
                        
                        L = np.linalg.cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                        
                        # Negative log likelihood
                        nll = 0.5 * y.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y) * np.log(2 * np.pi)
                        
                        if nll < best_nll:
                            best_nll = nll
                            best_params = (ls, var, noise)
                            
                    except np.linalg.LinAlgError:
                        continue
                        
        # Update parameters
        self.lengthscale, self.variance, self.noise = best_params
        
    def _compute_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute squared distances between points."""
        return np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)