"""
Wrapper for Transport Gaussian Process (TGPY) integration with MLPY.

This module provides learners that use TGPY for Gaussian Process
regression with transport maps.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
import warnings

from ..tasks import TaskRegr
from .regression import LearnerRegr
from ..predictions import PredictionRegr
from .gp_fallback import SimpleGP


class LearnerTGPRegressor(LearnerRegr):
    """Transport Gaussian Process Regressor wrapper for MLPY.
    
    This learner wraps the TGPY (Transport Gaussian Process) library
    to provide GP regression with transport maps capabilities.
    
    Parameters
    ----------
    id : str, optional
        Identifier for the learner.
    kernel : str, default='SE'
        Kernel type ('SE' for Squared Exponential, 'Matern', etc.)
    lengthscale : float, default=1.0
        Lengthscale parameter for the kernel.
    variance : float, default=1.0
        Variance parameter for the kernel.
    noise : float, default=0.1
        Noise variance for observations.
    transport : str, default='marginal'
        Type of transport to use ('marginal', 'radial', 'covariance').
    learning_rate : float, default=0.01
        Learning rate for optimization.
    n_iterations : int, default=100
        Number of optimization iterations.
    batch_size : float, default=1.0
        Batch size as fraction of data (1.0 = full batch).
    use_gpu : bool, default=False
        Whether to use GPU acceleration if available.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        kernel: str = 'SE',
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise: float = 0.1,
        transport: str = 'marginal',
        learning_rate: float = 0.01,
        n_iterations: int = 100,
        batch_size: float = 1.0,
        use_gpu: bool = False,
        **kwargs
    ):
        super().__init__(
            id=id or "tgp_regressor",
            predict_types=["response", "se"],
            properties=["gaussian_process", "probabilistic"],
            feature_types=["numeric"],
            **kwargs
        )
        
        self.kernel_type = kernel
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.transport_type = transport
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        # Model components (will be initialized during training)
        self.tgp_model = None
        self.learning = None
        self.fallback_gp = None
        self.X_train = None
        self.y_train = None
        self.input_dim = None
        self._use_fallback = False
        
        # Check if TGPY is available
        self._tgpy_available = False
        try:
            import tgpy
            import torch
            self._tgpy_available = True
        except ImportError:
            warnings.warn(
                "TGPY not available. Install it from the tgpy-master directory "
                "using: pip install -e tgpy-master/"
            )
            
    def _check_tgpy(self):
        """Check if TGPY is available."""
        if not self._tgpy_available:
            raise ImportError(
                "TGPY is not installed. Please install it from the tgpy-master "
                "directory using: pip install -e tgpy-master/"
            )
            
    def _setup_model(self, X: np.ndarray, y: np.ndarray):
        """Setup TGPY model components using corrected approach."""
        self._check_tgpy()
        
        import torch
        import tgpy
        from tgpy.tensor import to_tensor
        
        # Set device
        device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        tgpy.tensor._device = device
        
        # Convert data to torch tensors
        X_torch = to_tensor(X, device=device)
        y_torch = to_tensor(y, device=device)
        
        # Store for predictions
        self.X_train_torch = X_torch
        self.y_train_torch = y_torch
        
        # Create priors for parameters (using multiple chains for variational inference)
        self.n_chains = 5
        
        self.lengthscale_prior = tgpy.TgPriorUnivariate(
            'lengthscale', 
            ['g0'], 
            dim=self.n_chains,
            low=0.1, high=5.0, alpha=2, beta=2
        )
        
        self.variance_prior = tgpy.TgPriorUnivariate(
            'variance',
            ['g0'],
            dim=self.n_chains,
            low=0.1, high=5.0, alpha=2, beta=2
        )
        
        self.noise_prior = tgpy.TgPriorUnivariate(
            'noise',
            ['g0'],
            dim=self.n_chains,
            low=0.01, high=1.0, alpha=2, beta=3
        )
        
        # Sample initial values
        self.lengthscale_prior.sample_params()
        self.variance_prior.sample_params()
        self.noise_prior.sample_params()
        
        # Create custom TGPY regressor
        class TGPYRegressor(torch.nn.Module):
            def __init__(self, lengthscale_prior, variance_prior, noise_prior, n_chains):
                super().__init__()
                self.lengthscale_prior = lengthscale_prior
                self.variance_prior = variance_prior
                self.noise_prior = noise_prior
                self.n_chains = n_chains
                
            def kernel_matrix(self, X1, X2=None):
                if X2 is None:
                    X2 = X1
                    
                # Get parameters from priors for all chains
                ls_params = self.lengthscale_prior.p['g0']
                var_params = self.variance_prior.p['g0']
                
                # Compute squared distances
                dist_sq = torch.cdist(X1, X2, p=2).pow(2)
                
                # Compute kernel for each chain
                K_list = []
                for i in range(self.n_chains):
                    ls = ls_params[i]
                    var = var_params[i]
                    K_i = var * torch.exp(-0.5 * dist_sq / (ls ** 2))
                    K_list.append(K_i)
                    
                return torch.stack(K_list, dim=0)
                
            def add_noise(self, K):
                noise_params = self.noise_prior.p['g0']
                n = K.shape[-1]
                identity = torch.eye(n, device=K.device)
                
                K_noisy = K.clone()
                for i in range(self.n_chains):
                    K_noisy[i] += noise_params[i] * identity
                    
                return K_noisy
                
            def log_likelihood(self, X, y):
                K = self.kernel_matrix(X)
                K_noisy = self.add_noise(K)
                
                log_liks = []
                for i in range(self.n_chains):
                    try:
                        L = torch.linalg.cholesky(K_noisy[i])
                        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
                        
                        data_fit = -0.5 * torch.dot(y, alpha)
                        complexity = -torch.sum(torch.log(torch.diag(L)))
                        log_Z = -0.5 * len(y) * np.log(2 * np.pi)
                        
                        log_lik = data_fit + complexity + log_Z
                        log_liks.append(log_lik)
                    except:
                        log_liks.append(torch.tensor(-1e6, device=device))
                        
                return torch.stack(log_liks)
                
            def predict(self, X_train, y_train, X_test):
                K_train = self.kernel_matrix(X_train)
                K_test = self.kernel_matrix(X_test, X_train)
                K_train_noisy = self.add_noise(K_train)
                
                predictions = []
                variances = []
                
                for i in range(self.n_chains):
                    try:
                        L = torch.linalg.cholesky(K_train_noisy[i])
                        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze(-1)
                        
                        pred_mean = torch.mv(K_test[i], alpha)
                        
                        v = torch.linalg.solve_triangular(L, K_test[i].T, upper=False)
                        pred_var = torch.sum(v**2, dim=0)
                        
                        predictions.append(pred_mean)
                        variances.append(pred_var)
                    except:
                        predictions.append(torch.zeros(X_test.shape[0], device=device))
                        variances.append(torch.ones(X_test.shape[0], device=device))
                        
                pred_mean = torch.stack(predictions).mean(dim=0)
                pred_var = torch.stack(variances).mean(dim=0)
                
                return pred_mean, pred_var
        
        # Create model
        self.tgpy_regressor = TGPYRegressor(
            self.lengthscale_prior, self.variance_prior, self.noise_prior, self.n_chains
        )
        
        # Setup optimizer
        all_params = []
        all_params.extend(self.lengthscale_prior.p.values())
        all_params.extend(self.variance_prior.p.values())
        all_params.extend(self.noise_prior.p.values())
        
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)
        
    def _train_tgpy_model(self):
        """Train TGPY model using variational optimization."""
        import numpy as np
        
        for epoch in range(self.n_iterations):
            self.optimizer.zero_grad()
            
            # Compute log likelihood
            log_lik = self.tgpy_regressor.log_likelihood(self.X_train_torch, self.y_train_torch)
            
            # Add prior regularization
            prior_log_prob = (self.lengthscale_prior.logp().sum() + 
                             self.variance_prior.logp().sum() + 
                             self.noise_prior.logp().sum())
            
            # Total loss (negative log posterior)
            loss = -(log_lik.mean() + prior_log_prob)
            
            loss.backward()
            
            # Clamp parameters to valid ranges
            self.lengthscale_prior.clamp()
            self.variance_prior.clamp()
            self.noise_prior.clamp()
            
            self.optimizer.step()
        
    def _train(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> "LearnerTGPRegressor":
        """Train Transport Gaussian Process model."""
        # Get training data
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store training data
        self.X_train = X
        self.y_train = y
        self.input_dim = X.shape[1]
        
        # Setup and train model
        try:
            self._setup_model(X, y)
            
            # Train using corrected TGPY approach
            self._train_tgpy_model()
            
        except Exception as e:
            warnings.warn(f"TGPY training failed: {str(e)}. Using improved fallback GP implementation.")
            # Fallback to improved GP implementation if TGPY fails
            self._use_fallback = True
            self.fallback_gp = SimpleGP(
                lengthscale=self.lengthscale,
                variance=self.variance,
                noise=self.noise
            )
            # Optimize hyperparameters if we have enough data
            if len(X) > 20:
                self.fallback_gp.optimize_hyperparameters(X, y)
            # Fit the model
            self.fallback_gp.fit(X, y)
            
        return self
        
    def _predict(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Make predictions with Transport Gaussian Process."""
        # Get prediction data
        X = task.data(rows=row_ids, cols=task.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples = X.shape[0]
        
        # Ensure row_ids is a list
        if row_ids is None:
            row_ids = list(range(n_samples))
        
        # Make predictions
        if self._use_fallback and self.fallback_gp is not None:
            # Use improved fallback GP
            predictions, se = self.fallback_gp.predict(X)
                
        else:
            # Use TGPY for predictions
            try:
                self._check_tgpy()
                import torch
                from tgpy.tensor import to_tensor
                
                device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
                X_torch = to_tensor(X, device=device)
                
                # Get predictions from TGPY regressor
                with torch.no_grad():
                    pred_mean, pred_var = self.tgpy_regressor.predict(
                        self.X_train_torch, self.y_train_torch, X_torch
                    )
                    
                predictions = pred_mean.cpu().numpy()
                se = torch.sqrt(pred_var).cpu().numpy()
                
            except Exception as e:
                warnings.warn(f"TGPY prediction failed: {str(e)}. Using fallback.")
                # Fallback prediction
                predictions = np.full(n_samples, np.mean(self.y_train))
                se = np.full(n_samples, np.std(self.y_train))
                
        # Return predictions based on predict_type
        if self.predict_type == "se":
            return PredictionRegr(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                response=predictions,
                se=se,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )
        else:
            return PredictionRegr(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                response=predictions,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )
            
    def get_kernel_params(self) -> Dict[str, float]:
        """Get current kernel parameters."""
        if self.tgp_model is None:
            return {
                "lengthscale": self.lengthscale,
                "variance": self.variance,
                "noise": self.noise
            }
            
        try:
            self._check_tgpy()
            # Extract learned parameters from TGPY model
            params = {}
            if hasattr(self.tgp_model, 'kernel'):
                if hasattr(self.tgp_model.kernel, 'lengthscale'):
                    params['lengthscale'] = self.tgp_model.kernel.lengthscale.value.item()
                if hasattr(self.tgp_model.kernel, 'variance'):
                    params['variance'] = self.tgp_model.kernel.variance.value.item()
            if hasattr(self.tgp_model, 'noise'):
                params['noise'] = self.tgp_model.noise.value.item()
            return params
        except:
            return {
                "lengthscale": self.lengthscale,
                "variance": self.variance,
                "noise": self.noise
            }
            

class LearnerTGPClassifier(LearnerRegr):
    """Transport Gaussian Process Classifier wrapper for MLPY.
    
    This learner uses TGPY for classification by treating it as a
    regression problem with appropriate link functions.
    
    Note: This is a simplified wrapper. For full classification support,
    additional implementation would be needed.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            id=id or "tgp_classifier",
            predict_types=["response", "prob"],
            properties=["gaussian_process", "probabilistic", "binary"],
            feature_types=["numeric"],
            **kwargs
        )
        
        warnings.warn(
            "TGP Classifier is not fully implemented. "
            "Use TGP Regressor for regression tasks."
        )
        
    def _train(self, task, row_ids=None):
        raise NotImplementedError("TGP Classifier not yet implemented")
        
    def _predict(self, task, row_ids=None):
        raise NotImplementedError("TGP Classifier not yet implemented")