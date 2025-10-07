"""
Native Naive Bayes implementation for MLPY.

This is a pure Python/NumPy implementation of Gaussian Naive Bayes
for classification.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
import warnings

from ...tasks import TaskClassif
from ..classification import LearnerClassif
from ...predictions import PredictionClassif


class LearnerNaiveBayesGaussian(LearnerClassif):
    """Native Gaussian Naive Bayes classifier for MLPY.
    
    This classifier assumes that the features follow a Gaussian distribution.
    It's particularly effective for high-dimensional data and when the
    independence assumption approximately holds.
    
    Parameters
    ----------
    id : str, optional
        Identifier for the learner.
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features added to variances
        for calculation stability.
    predict_type : str, default='response'
        Type of prediction to make.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        var_smoothing: float = 1e-9,
        predict_type: str = "response",
        **kwargs
    ):
        super().__init__(
            id=id or "naive_bayes_gaussian_native",
            predict_type=predict_type,
            predict_types=["response", "prob"],
            properties=["twoclass", "multiclass"],
            feature_types=["numeric"],
            **kwargs
        )
        
        self.var_smoothing = var_smoothing
        
        # Model parameters
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.class_prior_ = None  # P(class)
        self.class_count_ = None  # Number of samples per class
        self.theta_ = None  # Mean of each feature per class
        self.sigma_ = None  # Variance of each feature per class
        self.epsilon_ = None  # Variance smoothing value
        
    def _calculate_class_stats(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate statistics for each class."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # Initialize parameter arrays
        self.theta_ = np.zeros((self.n_classes_, self.n_features_))
        self.sigma_ = np.zeros((self.n_classes_, self.n_features_))
        self.class_count_ = np.zeros(self.n_classes_)
        
        # Calculate statistics for each class
        for i, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]
            self.class_count_[i] = len(X_cls)
            
            # Calculate mean and variance for each feature
            self.theta_[i] = np.mean(X_cls, axis=0)
            self.sigma_[i] = np.var(X_cls, axis=0)
            
        # Calculate class priors
        self.class_prior_ = self.class_count_ / len(y)
        
        # Add smoothing to variance
        self.epsilon_ = self.var_smoothing * np.max(self.sigma_)
        self.sigma_ += self.epsilon_
        
    def _gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Calculate Gaussian probability density function.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        mean : array-like of shape (n_features,)
            Mean of the Gaussian distribution.
        var : array-like of shape (n_features,)
            Variance of the Gaussian distribution.
            
        Returns
        -------
        pdf : array-like of shape (n_samples,)
            Probability density values.
        """
        # Avoid numerical issues with very small variances
        var = np.maximum(var, 1e-10)
        
        # Calculate (x - mu)^2 / (2 * sigma^2)
        exponent = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
        
        # Calculate normalization constant: 1 / sqrt(2 * pi * sigma^2)
        # Use log for numerical stability
        log_norm = -0.5 * np.sum(np.log(2 * np.pi * var))
        
        # Return exp(log_norm + exponent)
        # Clip to prevent overflow
        log_pdf = log_norm + exponent
        log_pdf = np.clip(log_pdf, -500, 500)
        
        return np.exp(log_pdf)
        
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Calculate joint log-likelihood for each class.
        
        Returns
        -------
        log_likelihood : array-like of shape (n_samples, n_classes)
            Log-likelihood of samples for each class.
        """
        n_samples = X.shape[0]
        log_likelihood = np.zeros((n_samples, self.n_classes_))
        
        for i in range(self.n_classes_):
            # Calculate log P(X|class) for each feature (independence assumption)
            # P(X|class) = product of P(x_i|class) for all features
            # log P(X|class) = sum of log P(x_i|class)
            
            # For numerical stability, work in log space
            log_prior = np.log(self.class_prior_[i])
            
            # Calculate log likelihood for each feature
            for j in range(self.n_features_):
                # Extract feature column
                X_feature = X[:, j:j+1]
                
                # Calculate log of Gaussian PDF
                mean = self.theta_[i, j]
                var = self.sigma_[i, j]
                
                # log(1/sqrt(2*pi*var)) - 0.5*(x-mean)^2/var
                log_gaussian = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X_feature - mean) ** 2) / var
                
                log_likelihood[:, i] += log_gaussian.ravel()
                
            # Add log prior
            log_likelihood[:, i] += log_prior
            
        return log_likelihood
        
    def _train(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> "LearnerNaiveBayesGaussian":
        """Train Gaussian Naive Bayes classifier."""
        # Get training data
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Check for NaN values
        if np.any(np.isnan(X)):
            raise ValueError("Naive Bayes cannot handle NaN values in the input data")
            
        # Calculate class statistics
        self._calculate_class_stats(X, y)
        
        return self
        
    def _predict(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> PredictionClassif:
        """Make predictions with Gaussian Naive Bayes."""
        # Get prediction data
        X = task.data(rows=row_ids, cols=task.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Check for NaN values
        if np.any(np.isnan(X)):
            warnings.warn("Input contains NaN values. These will be replaced with feature means.")
            # Replace NaN with feature means (averaged across classes)
            feature_means = np.mean(self.theta_, axis=0)
            for j in range(self.n_features_):
                nan_mask = np.isnan(X[:, j])
                X[nan_mask, j] = feature_means[j]
        
        # Calculate joint log-likelihood
        log_likelihood = self._joint_log_likelihood(X)
        
        if self.predict_type == "response":
            # Predict class with highest log-likelihood
            predictions = self.classes_[np.argmax(log_likelihood, axis=1)]
            
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                response=predictions,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )
            
        else:  # prob
            # Calculate probabilities using softmax on log-likelihoods
            # P(class|X) = exp(log_likelihood) / sum(exp(log_likelihood))
            
            # Subtract max for numerical stability (softmax trick)
            log_likelihood_shifted = log_likelihood - np.max(log_likelihood, axis=1, keepdims=True)
            exp_log_likelihood = np.exp(log_likelihood_shifted)
            prob_matrix = exp_log_likelihood / np.sum(exp_log_likelihood, axis=1, keepdims=True)
            
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                prob=prob_matrix,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )
            
    def get_params(self) -> Dict[str, any]:
        """Get parameters of the fitted model."""
        if self.theta_ is None:
            return {}
            
        return {
            "classes": self.classes_,
            "class_prior": self.class_prior_,
            "theta": self.theta_,
            "sigma": self.sigma_,
            "epsilon": self.epsilon_
        }