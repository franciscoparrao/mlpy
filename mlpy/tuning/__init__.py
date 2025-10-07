"""
Hyperparameter tuning module for MLPY.

This module provides hyperparameter optimization via Optuna.
"""

# Import Optuna tuner if available
try:
    from .optuna_tuner import OptunaTuner
    __all__ = ['OptunaTuner']
    OPTUNA_AVAILABLE = True
except ImportError:
    __all__ = []
    OPTUNA_AVAILABLE = False

# Convenience function
def create_tuner(method='optuna', **kwargs):
    """Create a tuner instance.
    
    Parameters
    ----------
    method : str, default='optuna'
        Tuning method: currently only 'optuna' is available.
    **kwargs
        Arguments passed to the tuner.
        
    Returns
    -------
    Tuner
        A tuner instance.
    """
    if method == 'optuna':
        if OPTUNA_AVAILABLE:
            return OptunaTuner(**kwargs)
        else:
            raise ImportError(
                "Optuna not installed. Install with: pip install optuna"
            )
    else:
        raise ValueError(f"Unknown tuning method: {method}. Currently only 'optuna' is available.")