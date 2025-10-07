"""
Native Linear Regression implementation for MLPY.

This is a pure Python/NumPy implementation of linear regression
using the normal equation or gradient descent.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Literal
import warnings

from ...tasks import TaskRegr
from ..regression import LearnerRegr
from ...predictions import PredictionRegr


class LearnerLinearRegression(LearnerRegr):
    """Native Linear Regression implementation.
    
    This learner implements linear regression using either the normal equation
    (closed-form solution) or gradient descent optimization.
    
    Parameters
    ----------
    id : str, optional
        Identifier for the learner.
    method : {'normal', 'gradient_descent'}, default='normal'
        Method to use for fitting:
        - 'normal': Use normal equation (exact solution)
        - 'gradient_descent': Use gradient descent optimization
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    learning_rate : float, default=0.01
        Learning rate for gradient descent (ignored if method='normal').
    n_iterations : int, default=1000
        Number of iterations for gradient descent (ignored if method='normal').
    tolerance : float, default=1e-4
        Convergence tolerance for gradient descent.
    regularization : {'none', 'l2'}, default='none'
        Type of regularization to apply.
    alpha : float, default=1.0
        Regularization strength (ignored if regularization='none').
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        method: Literal['normal', 'gradient_descent'] = 'normal',
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-4,
        regularization: Literal['none', 'l2'] = 'none',
        alpha: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            id=id or "linear_regression_native",
            predict_types=["response", "se"],
            properties=["linear", "weights"],
            feature_types=["numeric"],
            **kwargs
        )
        
        self.method = method
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state
        
        # Model parameters
        self.coef_ = None
        self.intercept_ = None
        self.n_features_ = None
        self.loss_history_ = None
        self.sigma_ = None  # Residual standard error
        
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to feature matrix."""
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])
        
    def _normal_equation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve linear regression using normal equation."""
        # Add regularization term if needed
        if self.regularization == 'l2':
            n_features = X.shape[1]
            # Create regularization matrix (don't regularize intercept)
            lambda_matrix = self.alpha * np.eye(n_features)
            if self.fit_intercept:
                lambda_matrix[0, 0] = 0
            
            # Regularized normal equation: (X^T X + λI)^(-1) X^T y
            XtX = X.T @ X + lambda_matrix
        else:
            # Standard normal equation: (X^T X)^(-1) X^T y
            XtX = X.T @ X
            
        try:
            # Try to solve using Cholesky decomposition (more stable)
            L = np.linalg.cholesky(XtX)
            z = np.linalg.solve(L, X.T @ y)
            theta = np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if matrix is singular
            warnings.warn("Normal equation matrix is singular, using pseudo-inverse")
            theta = np.linalg.pinv(XtX) @ X.T @ y
            
        return theta
        
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve linear regression using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        rng = np.random.RandomState(self.random_state)
        theta = rng.randn(n_features) * 0.01
        
        self.loss_history_ = []
        
        for iteration in range(self.n_iterations):
            # Compute predictions
            y_pred = X @ theta
            
            # Compute loss
            loss = np.mean((y_pred - y) ** 2) / 2
            
            # Add regularization term to loss
            if self.regularization == 'l2':
                # Don't regularize intercept
                reg_term = self.alpha * np.sum(theta[1:]**2) / (2 * n_samples) if self.fit_intercept else \
                           self.alpha * np.sum(theta**2) / (2 * n_samples)
                loss += reg_term
                
            self.loss_history_.append(loss)
            
            # Compute gradients
            gradients = X.T @ (y_pred - y) / n_samples
            
            # Add regularization gradients
            if self.regularization == 'l2':
                reg_gradients = self.alpha * theta / n_samples
                if self.fit_intercept:
                    reg_gradients[0] = 0  # Don't regularize intercept
                gradients += reg_gradients
                
            # Update parameters
            theta_new = theta - self.learning_rate * gradients
            
            # Check convergence
            if np.linalg.norm(theta_new - theta) < self.tolerance:
                break
                
            theta = theta_new
            
        return theta
        
    def _train(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> "LearnerLinearRegression":
        """Train linear regression model."""
        # Get training data
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store number of features
        self.n_features_ = X.shape[1]
        
        # Add intercept if needed
        if self.fit_intercept:
            X = self._add_intercept(X)
            
        # Fit model using chosen method
        if self.method == 'normal':
            theta = self._normal_equation(X, y)
        else:  # gradient_descent
            theta = self._gradient_descent(X, y)
            
        # Extract coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
            
        # Calculate residual standard error for prediction intervals
        y_pred = X @ theta
        residuals = y - y_pred
        degrees_of_freedom = len(y) - len(theta)
        self.sigma_ = np.sqrt(np.sum(residuals**2) / degrees_of_freedom) if degrees_of_freedom > 0 else 0.0
        
        return self
        
    def _predict(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Make predictions with linear regression."""
        # Get prediction data
        X = task.data(rows=row_ids, cols=task.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Make predictions
        predictions = X @ self.coef_ + self.intercept_
        
        # Calculate standard errors if requested
        se = None
        if self.predict_type == "se":
            # For linear regression, the standard error is constant
            # In practice, it would depend on leverage (hat matrix)
            # Here we use a simplified version
            se = np.full(len(predictions), self.sigma_)
            
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            response=predictions,
            se=se,
            truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
        )
        
    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get model coefficients."""
        return self.coef_
        
    def importance(self) -> Optional[np.ndarray]:
        """Get feature importances (absolute standardized coefficients)."""
        if self.coef_ is None:
            return None
        # Return absolute values of coefficients as simple importance measure
        return np.abs(self.coef_)