"""Early stopping callback for MLPY.

This callback implements early stopping to prevent overfitting
during iterative processes.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .base import Callback


class CallbackEarlyStopping(Callback):
    """Callback for early stopping based on performance metrics.
    
    Parameters
    ----------
    patience : int, default=10
        Number of iterations with no improvement to wait before stopping.
    min_delta : float, default=0.0
        Minimum change to qualify as an improvement.
    restore_best : bool, default=True
        Whether to restore the best configuration/model.
    measure : str, optional
        Measure to monitor. If None, uses first measure.
    minimize : bool, optional
        Whether to minimize the measure. If None, inferred from measure.
    verbose : bool, default=True
        Whether to print early stopping messages.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best: bool = True,
        measure: Optional[str] = None,
        minimize: Optional[bool] = None,
        verbose: bool = True
    ):
        super().__init__(id="early_stopping")
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.measure = measure
        self.minimize = minimize
        self.verbose = verbose
        
        # State
        self.best_score = None
        self.best_iteration = None
        self.best_config = None
        self.wait = 0
        self.stopped_iteration = None
        self.stopped_config = None
        
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.minimize:
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
            
    # Resample callbacks
    def on_resample_begin(self, task: Any, learner: Any, resampling: Any) -> None:
        """Reset state for new resample."""
        self.best_score = None
        self.best_iteration = None
        self.wait = 0
        self.stopped_iteration = None
        
    def on_iteration_end(
        self,
        iteration: int,
        scores: Dict[str, float],
        train_time: float,
        predict_time: float
    ) -> None:
        """Check for early stopping after iteration."""
        # Get score to monitor
        if self.measure:
            if self.measure not in scores:
                return
            current_score = scores[self.measure]
        elif scores:
            # Use first measure
            current_score = list(scores.values())[0]
        else:
            return
            
        # Initialize best score
        if self.best_score is None:
            self.best_score = current_score
            self.best_iteration = iteration
            if self.minimize is None:
                # Try to infer from measure name
                self.minimize = any(
                    term in self.measure.lower() if self.measure else False
                    for term in ['error', 'loss', 'mse', 'mae', 'rmse']
                )
            return
            
        # Check if improved
        if self._is_better(current_score, self.best_score):
            self.best_score = current_score
            self.best_iteration = iteration
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_iteration = iteration
                if self.verbose:
                    print(f"\nEarly stopping triggered at iteration {iteration}")
                    print(f"Best iteration: {self.best_iteration} "
                          f"with score: {self.best_score:.4f}")
                    
                # TODO: Implement actual stopping mechanism
                # This would require modifying the resample loop
                
    # Tuning callbacks
    def on_tune_begin(self, learner: Any, param_set: Any, n_configs: int) -> None:
        """Reset state for new tuning."""
        self.best_score = None
        self.best_config = None
        self.wait = 0
        self.stopped_config = None
        
    def on_config_end(self, config_num: int, score: float) -> None:
        """Check for early stopping after config evaluation."""
        # Initialize best score
        if self.best_score is None:
            self.best_score = score
            self.best_config = config_num
            if self.minimize is None:
                self.minimize = True  # Default for tuning
            return
            
        # Check if improved
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.best_config = config_num
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_config = config_num
                if self.verbose:
                    print(f"\nEarly stopping triggered at config {config_num}")
                    print(f"Best config: {self.best_config} "
                          f"with score: {self.best_score:.4f}")
                    
                # TODO: Implement actual stopping mechanism
                
    def should_stop(self) -> bool:
        """Check if early stopping has been triggered.
        
        Returns
        -------
        bool
            True if should stop, False otherwise.
        """
        return (self.stopped_iteration is not None or 
                self.stopped_config is not None)
                
    def get_best(self) -> Dict[str, Any]:
        """Get information about the best iteration/config.
        
        Returns
        -------
        dict
            Dictionary with best score and iteration/config.
        """
        result = {}
        
        if self.best_score is not None:
            result['best_score'] = self.best_score
            
        if self.best_iteration is not None:
            result['best_iteration'] = self.best_iteration
            
        if self.best_config is not None:
            result['best_config'] = self.best_config
            
        return result


__all__ = ["CallbackEarlyStopping"]