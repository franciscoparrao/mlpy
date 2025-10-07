"""
Native Logistic Regression implementation for MLPY.

This is a pure Python/NumPy implementation of logistic regression
for binary and multiclass classification.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Literal
import warnings

from ...tasks import TaskClassif
from ..classification import LearnerClassif
from ...predictions import PredictionClassif


class LearnerLogisticRegression(LearnerClassif):
    """Native Logistic Regression implementation.
    
    This learner implements logistic regression using gradient descent
    with support for binary and multiclass classification (one-vs-rest).
    
    Parameters
    ----------
    id : str, optional
        Identifier for the learner.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_iterations : int, default=1000
        Maximum number of iterations.
    tolerance : float, default=1e-4
        Convergence tolerance.
    regularization : {'none', 'l2'}, default='l2'
        Type of regularization to apply.
    C : float, default=1.0
        Inverse of regularization strength (larger values mean less regularization).
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print convergence information.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-4,
        regularization: Literal['none', 'l2'] = 'l2',
        C: float = 1.0,
        random_state: Optional[int] = None,
        verbose: bool = False,
        predict_type: str = "response",
        **kwargs
    ):
        super().__init__(
            id=id or "logistic_regression_native",
            predict_type=predict_type,
            predict_types=["response", "prob"],
            properties=["twoclass", "multiclass", "linear", "weights"],
            feature_types=["numeric"],
            **kwargs
        )
        
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.C = C
        self.random_state = random_state
        self.verbose = verbose
        
        # Model parameters
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.loss_history_ = None
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
        
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax function for multiclass."""
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to feature matrix."""
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])
        
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Fit binary logistic regression."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        rng = np.random.RandomState(self.random_state)
        theta = rng.randn(n_features) * 0.01
        
        # Convert labels to 0/1
        y_binary = (y == self.classes_[1]).astype(float)
        
        loss_history = []
        
        for iteration in range(self.n_iterations):
            # Forward pass
            z = X @ theta
            h = self._sigmoid(z)
            
            # Compute loss (negative log-likelihood)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-7
            loss = -np.mean(y_binary * np.log(h + epsilon) + 
                           (1 - y_binary) * np.log(1 - h + epsilon))
            
            # Add regularization term
            if self.regularization == 'l2' and self.C > 0:
                # Lambda = 1/C (sklearn convention)
                lambda_reg = 1 / self.C
                # Don't regularize intercept
                reg_term = lambda_reg * np.sum(theta[1:]**2) / (2 * n_samples) if self.fit_intercept else \
                          lambda_reg * np.sum(theta**2) / (2 * n_samples)
                loss += reg_term
                
            loss_history.append(loss)
            
            # Compute gradients
            gradients = X.T @ (h - y_binary) / n_samples
            
            # Add regularization gradients
            if self.regularization == 'l2' and self.C > 0:
                lambda_reg = 1 / self.C
                reg_gradients = lambda_reg * theta / n_samples
                if self.fit_intercept:
                    reg_gradients[0] = 0  # Don't regularize intercept
                gradients += reg_gradients
                
            # Update parameters
            theta_new = theta - self.learning_rate * gradients
            
            # Check convergence
            if np.linalg.norm(theta_new - theta) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
                
            theta = theta_new
            
        return theta, loss_history
        
    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Fit multiclass logistic regression using one-vs-rest."""
        n_samples, n_features = X.shape
        n_classes = self.n_classes_
        
        # Initialize parameters for each class
        theta_all = np.zeros((n_classes, n_features))
        loss_history_all = []
        
        # Fit one classifier per class
        for i, class_label in enumerate(self.classes_):
            if self.verbose:
                print(f"Fitting class {class_label} ({i+1}/{n_classes})")
                
            # Create binary labels for this class
            y_binary = (y == class_label).astype(float)
            
            # Fit binary classifier
            theta, loss_history = self._fit_binary_class(X, y_binary)
            theta_all[i] = theta
            loss_history_all.append(loss_history)
            
        return theta_all, loss_history_all
        
    def _fit_binary_class(self, X: np.ndarray, y_binary: np.ndarray) -> tuple:
        """Fit a single binary classifier (used in one-vs-rest)."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        rng = np.random.RandomState(self.random_state)
        theta = rng.randn(n_features) * 0.01
        
        loss_history = []
        
        for iteration in range(self.n_iterations):
            # Forward pass
            z = X @ theta
            h = self._sigmoid(z)
            
            # Compute loss
            epsilon = 1e-7
            loss = -np.mean(y_binary * np.log(h + epsilon) + 
                           (1 - y_binary) * np.log(1 - h + epsilon))
            
            # Add regularization
            if self.regularization == 'l2' and self.C > 0:
                lambda_reg = 1 / self.C
                reg_term = lambda_reg * np.sum(theta[1:]**2) / (2 * n_samples) if self.fit_intercept else \
                          lambda_reg * np.sum(theta**2) / (2 * n_samples)
                loss += reg_term
                
            loss_history.append(loss)
            
            # Compute gradients
            gradients = X.T @ (h - y_binary) / n_samples
            
            # Add regularization gradients
            if self.regularization == 'l2' and self.C > 0:
                lambda_reg = 1 / self.C
                reg_gradients = lambda_reg * theta / n_samples
                if self.fit_intercept:
                    reg_gradients[0] = 0
                gradients += reg_gradients
                
            # Update parameters
            theta_new = theta - self.learning_rate * gradients
            
            # Check convergence
            if np.linalg.norm(theta_new - theta) < self.tolerance:
                break
                
            theta = theta_new
            
        return theta, loss_history
        
    def _train(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> "LearnerLogisticRegression":
        """Train logistic regression model."""
        # Get training data
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store metadata
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Add intercept if needed
        if self.fit_intercept:
            X = self._add_intercept(X)
            
        # Fit model
        if self.n_classes_ == 2:
            # Binary classification
            theta, loss_history = self._fit_binary(X, y)
            self.loss_history_ = loss_history
            
            # Extract coefficients and intercept
            if self.fit_intercept:
                self.intercept_ = np.array([theta[0]])
                self.coef_ = theta[1:].reshape(1, -1)
            else:
                self.intercept_ = np.array([0.0])
                self.coef_ = theta.reshape(1, -1)
        else:
            # Multiclass classification
            theta_all, loss_history_all = self._fit_multiclass(X, y)
            self.loss_history_ = loss_history_all
            
            # Extract coefficients and intercepts
            if self.fit_intercept:
                self.intercept_ = theta_all[:, 0]
                self.coef_ = theta_all[:, 1:]
            else:
                self.intercept_ = np.zeros(self.n_classes_)
                self.coef_ = theta_all
                
        return self
        
    def _predict(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> PredictionClassif:
        """Make predictions with logistic regression."""
        # Get prediction data
        X = task.data(rows=row_ids, cols=task.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Add intercept if needed
        if self.fit_intercept:
            X = self._add_intercept(X)
            
        n_samples = X.shape[0]
        
        if self.predict_type == "response":
            # Predict classes
            if self.n_classes_ == 2:
                # Binary classification
                theta = np.concatenate([self.intercept_, self.coef_[0]])
                z = X @ theta
                probs = self._sigmoid(z)
                predictions = self.classes_[1] if isinstance(self.classes_[0], str) else self.classes_[1]
                predictions = np.where(probs >= 0.5, predictions, self.classes_[0])
            else:
                # Multiclass
                theta_all = np.column_stack([self.intercept_, self.coef_])
                scores = X @ theta_all.T
                predictions = self.classes_[np.argmax(scores, axis=1)]
                
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                response=predictions,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )
        else:  # prob
            # Predict probabilities
            if self.n_classes_ == 2:
                # Binary classification
                theta = np.concatenate([self.intercept_, self.coef_[0]])
                z = X @ theta
                probs_positive = self._sigmoid(z)
                prob_matrix = np.column_stack([1 - probs_positive, probs_positive])
            else:
                # Multiclass using softmax
                theta_all = np.column_stack([self.intercept_, self.coef_])
                scores = X @ theta_all.T
                prob_matrix = self._softmax(scores)
                
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                prob=prob_matrix,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )
            
    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get model coefficients."""
        return self.coef_
        
    def importance(self) -> Optional[np.ndarray]:
        """Get feature importances (average absolute coefficients for multiclass)."""
        if self.coef_ is None:
            return None
        # For multiclass, average absolute coefficients across classes
        if self.n_classes_ > 2:
            return np.mean(np.abs(self.coef_), axis=0)
        else:
            return np.abs(self.coef_[0])